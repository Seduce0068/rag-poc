//! RAG PoC の指標計算を Rust で再実装したライブラリ。
//!
//! 挙動の基準は次の Python 実装。
//! - `rag_poc/ingest.py`: `_split_long_text` / `_point_id`
//! - `rag_poc/evaluate.py`: `_corpus_fingerprint` / `_source_matches` / nDCG
//! - `rag_poc/compare_embeddings.py`: `_cosine` / `_distribution`
//!
//! `unsafe` ブロック・関数・トレイトは一切使用しない。

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// スコア分布（min / max / mean / stdev、6 桁丸め）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub stdev: f64,
}

/// レイテンシ統計（`count` はサンプル数、統計値は 6 桁丸め）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: usize,
    pub avg: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub stdev: f64,
}

/// コーパスの fingerprint（ファイル数と本文連結の SHA-256 hex）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusFingerprint {
    pub count: usize,
    pub sha256: String,
}

/// Python の `round(x, 6)` 相当の丸め。
fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

/// 文字（Unicode scalar value）単位のウィンドウで分割する。
///
/// Python `ingest._split_long_text` と同じウィンドウ幅・ステップを使う。
/// `overlap >= chunk_size` のときはステップが正にならないため `Err`。
/// 各断片は `trim` し、空文字は結果に含めない。
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Result<Vec<String>, String> {
    if overlap >= chunk_size {
        return Err(format!(
            "overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        ));
    }
    let step = chunk_size - overlap;
    let chars: Vec<char> = text.chars().collect();
    let length = chars.len();

    if length <= chunk_size {
        let piece = text.trim();
        return Ok(if piece.is_empty() {
            Vec::new()
        } else {
            vec![piece.to_string()]
        });
    }

    let mut pieces = Vec::new();
    let mut start = 0;
    while start < length {
        let end = (start + chunk_size).min(length);
        let window: String = chars[start..end].iter().collect();
        let piece = window.trim();
        if !piece.is_empty() {
            pieces.push(piece.to_string());
        }
        if end >= length {
            break;
        }
        start += step;
    }
    Ok(pieces)
}

/// コサイン類似度。次元不一致は `Err`、いずれかのノルムが 0 なら 0.0。
pub fn cosine(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!("dimension mismatch: {} vs {}", a.len(), b.len()));
    }
    let mut dot = 0.0;
    let mut norm_a_sq = 0.0;
    let mut norm_b_sq = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a_sq += x * x;
        norm_b_sq += y * y;
    }
    let denominator = norm_a_sq.sqrt() * norm_b_sq.sqrt();
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / denominator)
    }
}

/// min / max / mean / 母集団標準偏差（ddof=0）。空入力は `Err`。
pub fn distribution(scores: &[f64]) -> Result<Distribution, String> {
    if scores.is_empty() {
        return Err("scores must not be empty".to_string());
    }
    let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance =
        scores.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / scores.len() as f64;
    Ok(Distribution {
        min: round6(min),
        max: round6(max),
        mean: round6(mean),
        stdev: round6(variance.sqrt()),
    })
}

/// パス区切り `'/'` 以降を basename として取り出す。
fn basename(value: &str) -> &str {
    value.rsplit('/').next().unwrap_or(value)
}

/// 照合用に `\\` → `/`、前後空白除去、小文字化を行う。
fn normalize(value: &str) -> String {
    value.replace('\\', "/").trim().to_lowercase()
}

/// Python `evaluate._source_matches` 相当の照合。
/// 完全一致・相対パス suffix・basename の一致を許容する。
fn source_matches(source: &str, expected: &str) -> bool {
    let normalized = normalize(source);
    let target = normalize(expected);
    if normalized.is_empty() || target.is_empty() {
        return false;
    }
    if normalized == target {
        return true;
    }
    let suffix = format!("/{}", target.trim_start_matches('/'));
    if normalized.ends_with(&suffix) {
        return true;
    }
    basename(&normalized) == basename(&target)
}

/// query-level hit@k。先頭 `k` 件に正解が 1 件でもあれば 1.0、否则 0.0。
pub fn recall_at(relevant: &[String], retrieved: &[String], k: usize) -> f64 {
    let head = &retrieved[..retrieved.len().min(k)];
    let hit = head.iter().any(|source| {
        relevant
            .iter()
            .any(|expected| source_matches(source, expected))
    });
    if hit {
        1.0
    } else {
        0.0
    }
}

/// nDCG@k。gain=1 / discount=log2(rank+1)、同一正解は最初の 1 回だけ計上。
/// 丸めは行わない（呼び出し側で行う）。
pub fn ndcg_at(relevant: &[String], retrieved: &[String], k: usize) -> f64 {
    let head = &retrieved[..retrieved.len().min(k)];
    let mut dcg = 0.0;
    let mut credited: Vec<String> = Vec::new();
    for (index, source) in head.iter().enumerate() {
        let rank = index + 1;
        for expected in relevant {
            let key = expected.to_lowercase();
            if credited.contains(&key) {
                continue;
            }
            if source_matches(source, expected) {
                credited.push(key);
                dcg += 1.0 / (rank as f64 + 1.0).log2();
                break;
            }
        }
    }

    let ideal = relevant.len().min(k);
    if ideal == 0 {
        return 0.0;
    }
    let idcg: f64 = (1..=ideal)
        .map(|rank| 1.0 / (rank as f64 + 1.0).log2())
        .sum();
    if idcg > 0.0 {
        dcg / idcg
    } else {
        0.0
    }
}

/// レイテンシ統計。空入力は `Err`。median は偶数件で中央 2 件の平均。
pub fn latency_stats(samples: &[f64]) -> Result<LatencyStats, String> {
    if samples.is_empty() {
        return Err("samples must not be empty".to_string());
    }
    let count = samples.len();
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let avg = samples.iter().sum::<f64>() / count as f64;
    let median = if count % 2 == 1 {
        sorted[count / 2]
    } else {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    };
    let variance = samples.iter().map(|x| (x - avg) * (x - avg)).sum::<f64>() / count as f64;
    Ok(LatencyStats {
        count,
        avg: round6(avg),
        median: round6(median),
        min: round6(sorted[0]),
        max: round6(sorted[count - 1]),
        stdev: round6(variance.sqrt()),
    })
}

/// バイト列を小文字 hex に変換する（外部 crate なし）。
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// パス文字列でソートし、本文バイトを連結した SHA-256 hex と件数を返す。
pub fn corpus_fingerprint(paths: &[(String, Vec<u8>)]) -> CorpusFingerprint {
    let mut ordered: Vec<&(String, Vec<u8>)> = paths.iter().collect();
    ordered.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = Sha256::new();
    let mut count = 0;
    for (_, body) in ordered {
        hasher.update(body.as_slice());
        count += 1;
    }
    let digest = hasher.finalize();
    CorpusFingerprint {
        count,
        sha256: hex_encode(digest.as_slice()),
    }
}

/// `source + "\\0" + chunk_index` の BLAKE2b 64bit digest を big-endian u64 にする。
/// Python `ingest._point_id` と同一値。
pub fn point_id(source: &str, chunk_index: u64) -> u64 {
    use blake2::digest::{Update, VariableOutput};
    use blake2::Blake2bVar;

    let input = format!("{source}\0{chunk_index}");
    let mut hasher = Blake2bVar::new(8).expect("digest_size=8 is valid for BLAKE2b");
    hasher.update(input.as_bytes());
    let mut digest = [0u8; 8];
    hasher
        .finalize_variable(&mut digest)
        .expect("buffer length equals digest size");
    u64::from_be_bytes(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_uses_character_windows() {
        assert_eq!(
            chunk_text("abcdefghij", 4, 2).unwrap(),
            vec!["abcd", "cdef", "efgh", "ghij"]
        );
        assert_eq!(
            chunk_text("あいうえおかきく", 3, 1).unwrap(),
            vec!["あいう", "うえお", "おかき", "きく"]
        );
    }

    #[test]
    fn chunk_rejects_nonpositive_step_and_empty_pieces() {
        assert!(chunk_text("abc", 3, 3).is_err());
        assert!(chunk_text("abc", 0, 0).is_err());
        assert!(chunk_text("   ", 3, 1).unwrap().is_empty());
    }

    #[test]
    fn cosine_matches_expected_cases() {
        assert!((cosine(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]).unwrap(), 0.0);
        assert!(cosine(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn distribution_rounds_to_six_digits() {
        let dist = distribution(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(dist.min, 1.0);
        assert_eq!(dist.max, 4.0);
        assert_eq!(dist.mean, 2.5);
        assert_eq!(dist.stdev, 1.118034);
        assert!(distribution(&[]).is_err());
    }

    #[test]
    fn recall_accepts_basename_and_exact() {
        let relevant = vec!["doc.md".to_string()];
        let retrieved = vec!["dir/doc.md".to_string(), "other.md".to_string()];
        assert_eq!(recall_at(&relevant, &retrieved, 2), 1.0);
        assert_eq!(recall_at(&relevant, &retrieved, 0), 0.0);
        assert_eq!(recall_at(&relevant, &retrieved[1..], 1), 0.0);
    }

    #[test]
    fn ndcg_uses_log_discount() {
        let relevant = vec!["doc.md".to_string()];
        let hit = vec!["doc.md".to_string()];
        let miss = vec!["other.md".to_string()];
        assert_eq!(ndcg_at(&relevant, &hit, 3), 1.0);
        assert_eq!(ndcg_at(&relevant, &miss, 3), 0.0);
        let second = ndcg_at(&relevant, &[miss[0].clone(), hit[0].clone()], 2);
        assert!((second - 1.0 / 3.0f64.log2()).abs() < 1e-12);
    }

    #[test]
    fn latency_stats_covers_median() {
        let stats = latency_stats(&[0.1, 0.3, 0.5]).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, 0.1);
        assert_eq!(stats.median, 0.3);
        assert_eq!(stats.max, 0.5);
        assert!(latency_stats(&[]).is_err());
    }

    #[test]
    fn fingerprint_sorts_by_path() {
        let paths = vec![
            ("b.md".to_string(), b"BB".to_vec()),
            ("a.md".to_string(), b"A".to_vec()),
        ];
        let forward = corpus_fingerprint(&paths);
        let reversed = corpus_fingerprint(&[paths[1].clone(), paths[0].clone()]);
        assert_eq!(forward.count, 2);
        assert_eq!(forward.sha256.len(), 64);
        assert_eq!(forward.sha256, reversed.sha256);
    }

    #[test]
    fn point_id_is_stable_and_source_sensitive() {
        assert_eq!(point_id("doc.md", 0), point_id("doc.md", 0));
        assert_ne!(point_id("doc.md", 0), point_id("other.md", 0));
    }
}
