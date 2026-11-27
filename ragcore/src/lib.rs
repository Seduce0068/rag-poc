//! RAG PoC の指標計算を Rust で再実装したライブラリ。
//!
//! 挙動の基準は次の Python 実装。
//! - `rag_poc/ingest.py`: `_split_long_text` / `_point_id`
//! - `rag_poc/evaluate.py`: `_corpus_fingerprint` / `_source_matches` / nDCG
//! - `rag_poc/compare_embeddings.py`: `_cosine` / `_distribution`
//!
//! `unsafe` ブロック・関数・トレイトは一切使用しない。

pub mod cloud;
pub mod config;
pub mod evaluate;
pub mod ingest;
pub mod llama;
pub mod qdrant;
pub mod query;
pub mod search;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

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

/// Reciprocal Rank Fusion（k=60）。順位のみを統合し、各 point の score を上書きする。
/// Python `query.rrf_fuse` と同じく、タイブレークは `str(id)` の辞書順。
pub fn rrf_fuse(rankings: &[Vec<serde_json::Value>], top_k: usize) -> Vec<serde_json::Value> {
    const RRF_K: f64 = 60.0;
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut items: HashMap<String, serde_json::Value> = HashMap::new();

    for ranking in rankings {
        for (index, point) in ranking.iter().enumerate() {
            let rank = index + 1;
            let key = point
                .get("id")
                .map(serde_json::Value::to_string)
                .unwrap_or_else(|| "null".to_string());
            *scores.entry(key.clone()).or_insert(0.0) += 1.0 / (RRF_K + rank as f64);
            items.entry(key).or_insert_with(|| point.clone());
        }
    }

    for (key, score) in &scores {
        if let Some(item) = items.get_mut(key) {
            if let Some(object) = item.as_object_mut() {
                object.insert("score".to_string(), serde_json::json!(score));
            }
        }
    }

    let mut fused: Vec<serde_json::Value> = items.into_values().collect();
    fused.sort_by(|a, b| {
        let score_a = a
            .get("score")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let score_b = b
            .get("score")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let id_a = a
                    .get("id")
                    .map(serde_json::Value::to_string)
                    .unwrap_or_default();
                let id_b = b
                    .get("id")
                    .map(serde_json::Value::to_string)
                    .unwrap_or_default();
                id_a.cmp(&id_b)
            })
    });
    fused.truncate(top_k);
    fused
}

/// eval-results-*.json の公開ゲート結果。
///
/// `diagnostics` にはファイル名とキー名のみを格納し、JSON の値は一切含めない。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateReport {
    pub ok: bool,
    pub checked: usize,
    pub failed: usize,
    pub diagnostics: Vec<String>,
}

const EVALUATE_KEYS: &[&str] = &[
    "generated_at",
    "schema_version",
    "qrels_schema_version",
    "qrels_label",
    "collection",
    "embed_model",
    "summary",
    "results",
];
const CLOUD_BENCH_KEYS: &[&str] = &[
    "generated_at",
    "model",
    "corpus",
    "qrels_label",
    "chunk_size",
    "summary",
    "results",
    "notes",
];
const FORBIDDEN_KEYS: &[&str] = &["answer", "rerank_candidates", "rerank_reply"];
/// 再帰走査の深さ上限（ルートを 0 とし、この値に達した階層は走査しない）。
const MAX_SCAN_DEPTH: u32 = 8;
/// 1 つの配列あたりの走査要素数上限。
const MAX_SCAN_ARRAY_ITEMS: usize = 1000;

/// `serde_json::Value` を再帰走査し、見つかった禁止キー名だけを `present` に追加する。
///
/// 値は一切収集しない（diagnostics へ漏らさないため）。深さ・配列要素数の
/// 上限により、過大な入力でも走査量が抑えられる。
fn collect_forbidden_keys(value: &serde_json::Value, depth: u32, present: &mut Vec<&'static str>) {
    if depth >= MAX_SCAN_DEPTH {
        return;
    }
    match value {
        serde_json::Value::Object(map) => {
            for &forbidden in FORBIDDEN_KEYS {
                if map.contains_key(forbidden) && !present.contains(&forbidden) {
                    present.push(forbidden);
                }
            }
            for child in map.values() {
                collect_forbidden_keys(child, depth + 1, present);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items.iter().take(MAX_SCAN_ARRAY_ITEMS) {
                collect_forbidden_keys(item, depth + 1, present);
            }
        }
        _ => {}
    }
}

/// 成果物名ごとの必須キー集合（ci.yml の heredoc と同じ対応表）。
fn required_keys(name: &str) -> &'static [&'static str] {
    match name {
        "eval-results.json"
        | "eval-results-baseline.json"
        | "eval-results-other-domain.json"
        | "eval-results-ruri-subset.json"
        | "eval-results-ruri-full.json"
        | "eval-results-alt.json"
        | "eval-results-cloud-pipeline.json" => EVALUATE_KEYS,
        "eval-results-cloud-subset.json"
        | "eval-results-cloud-full.json"
        | "eval-results-cloud-small-subset.json"
        | "eval-results-cloud-qwen3-subset.json" => CLOUD_BENCH_KEYS,
        _ if name.starts_with("eval-results-cloud-") => CLOUD_BENCH_KEYS,
        _ => EVALUATE_KEYS,
    }
}

fn take_digits(bytes: &[u8], pos: &mut usize, count: usize) -> bool {
    let end = *pos + count;
    if end > bytes.len() || !bytes[*pos..end].iter().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    *pos = end;
    true
}

/// RFC3339 date-time かつ UTC オフセット（Z / +00:00 / -00:00）を要求する。
/// 依存追加なしで元の正規表現と同じ受理条件を再現する。
fn is_rfc3339_utc(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut pos = 0;
    if !take_digits(bytes, &mut pos, 4) {
        return false;
    }
    if bytes.get(pos).copied() != Some(b'-') {
        return false;
    }
    pos += 1;
    if !take_digits(bytes, &mut pos, 2) {
        return false;
    }
    if bytes.get(pos).copied() != Some(b'-') {
        return false;
    }
    pos += 1;
    if !take_digits(bytes, &mut pos, 2) {
        return false;
    }
    match bytes.get(pos).copied() {
        Some(b'T') | Some(b't') => pos += 1,
        _ => return false,
    }
    if !take_digits(bytes, &mut pos, 2) {
        return false;
    }
    if bytes.get(pos).copied() != Some(b':') {
        return false;
    }
    pos += 1;
    if !take_digits(bytes, &mut pos, 2) {
        return false;
    }
    if bytes.get(pos).copied() != Some(b':') {
        return false;
    }
    pos += 1;
    if !take_digits(bytes, &mut pos, 2) {
        return false;
    }
    if bytes.get(pos).copied() == Some(b'.') {
        pos += 1;
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos == start {
            return false;
        }
    }
    let tail = &bytes[pos..];
    tail == &b"Z"[..] || tail == &b"+00:00"[..] || tail == &b"-00:00"[..]
}

/// (ファイル名, JSON 文字列) の組を検証する。
///
/// 検査内容はci.yml の heredoc と同一: 必須キー、禁止キー不在、
/// 全成果物の `generated_at` UTC。禁止キーは JSON 全体を再帰走査して検出し、
/// 値は diagnostics に一切含めない。
pub fn check_eval_results(files: &[(String, String)]) -> GateReport {
    let mut diagnostics = Vec::new();
    let mut failed = 0usize;
    for (name, content) in files {
        let data: serde_json::Value = match serde_json::from_str(content) {
            Ok(data) => data,
            Err(_) => {
                diagnostics.push(format!("{name}: invalid JSON"));
                failed += 1;
                continue;
            }
        };
        let object = match data.as_object() {
            Some(object) => object,
            None => {
                diagnostics.push(format!("{name}: top-level object required"));
                failed += 1;
                continue;
            }
        };
        let mut file_failed = false;

        let mut missing: Vec<&str> = required_keys(name)
            .iter()
            .copied()
            .filter(|key| !object.contains_key(*key))
            .collect();
        missing.sort_unstable();
        if !missing.is_empty() {
            diagnostics.push(format!("{}: missing keys: {}", name, missing.join(", ")));
            file_failed = true;
        }

        // generated_at は全成果物の必須キー。cloud-* に限らず UTC オフセットを要求する。
        if required_keys(name).contains(&"generated_at") {
            let valid = object
                .get("generated_at")
                .and_then(serde_json::Value::as_str)
                .map(is_rfc3339_utc)
                .unwrap_or(false);
            if !valid {
                diagnostics.push(format!("{name}: generated_at requires RFC3339 UTC"));
                file_failed = true;
            }
        }

        // トップレベルと results 直下に限らず、JSON 全体を再帰走査する。
        let mut present: Vec<&'static str> = Vec::new();
        collect_forbidden_keys(&data, 0, &mut present);
        present.sort_unstable();
        if !present.is_empty() {
            diagnostics.push(format!(
                "ERROR: {}: forbidden keys present: {}",
                name,
                present.join(", ")
            ));
            file_failed = true;
        }

        if file_failed {
            failed += 1;
        }
    }
    GateReport {
        ok: failed == 0,
        checked: files.len(),
        failed,
        diagnostics,
    }
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

    #[test]
    fn eval_results_gate_accepts_valid_artifacts() {
        let evaluate = r#"{
            "generated_at":"2026-11-26T00:00:00Z",
            "schema_version":1,"qrels_schema_version":1,"qrels_label":"qrels.json",
            "collection":"c","embed_model":"m","summary":{},"results":[]
        }"#;
        let cloud = r#"{
            "generated_at":"2026-11-26T00:00:00+00:00",
            "model":"m","corpus":"c","qrels_label":"qrels.json",
            "chunk_size":512,"summary":{},"results":[],"notes":[]
        }"#;
        let files = vec![
            ("eval-results.json".to_string(), evaluate.to_string()),
            (
                "eval-results-cloud-subset.json".to_string(),
                cloud.to_string(),
            ),
        ];
        let report = check_eval_results(&files);
        assert!(report.ok);
        assert_eq!(report.checked, 2);
        assert_eq!(report.failed, 0);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn eval_results_gate_rejects_forbidden_keys_without_values() {
        let json = r#"{
            "generated_at":"2026-11-26T00:00:00Z",
            "schema_version":1,"qrels_schema_version":1,"qrels_label":"q",
            "collection":"c","embed_model":"m","summary":{},
            "results":[{"id":"q0","answer":"SECRET_VALUE"}],
            "rerank_reply":"ANOTHER_SECRET"
        }"#;
        let files = vec![("eval-results.json".to_string(), json.to_string())];
        let report = check_eval_results(&files);
        assert!(!report.ok);
        let joined = report.diagnostics.join("\n");
        assert!(joined.contains("eval-results.json"));
        assert!(joined.contains("answer"));
        assert!(joined.contains("rerank_reply"));
        assert!(!joined.contains("SECRET_VALUE"));
        assert!(!joined.contains("ANOTHER_SECRET"));
    }

    #[test]
    fn eval_results_gate_requires_offset_on_cloud_generated_at() {
        let json = r#"{
            "generated_at":"2026-11-26T00:00:00",
            "model":"m","corpus":"c","qrels_label":"q",
            "chunk_size":512,"summary":{},"results":[],"notes":[]
        }"#;
        let files = vec![("eval-results-cloud-full.json".to_string(), json.to_string())];
        let report = check_eval_results(&files);
        assert!(!report.ok);
        assert!(report
            .diagnostics
            .iter()
            .any(|line| { line.contains("generated_at requires RFC3339 UTC") }));
    }

    #[test]
    fn eval_results_gate_checks_forbidden_keys_on_non_cloud_artifacts() {
        let json = r#"{
            "generated_at":"2026-11-26T00:00:00Z",
            "schema_version":1,"qrels_schema_version":1,"qrels_label":"q",
            "collection":"c","embed_model":"m","summary":{},
            "results":[{"rerank_candidates":["SECRET_VALUE"]}]
        }"#;
        let files = vec![("eval-results-custom.json".to_string(), json.to_string())];
        let report = check_eval_results(&files);
        assert!(!report.ok);
        let joined = report.diagnostics.join("\n");
        assert!(joined.contains("eval-results-custom.json"));
        assert!(joined.contains("rerank_candidates"));
        assert!(!joined.contains("SECRET_VALUE"));
    }

    /// `summary` 配下に `wrappers` 段の入れ子を置き、最内側に `answer` を持つ成果物。
    fn artifact_with_nested_answer(wrappers: usize) -> String {
        let mut nested = r#"{"answer":"SECRET_VALUE"}"#.to_string();
        for _ in 0..wrappers {
            nested = format!("{{\"n\":{nested}}}");
        }
        format!(
            r#"{{"generated_at":"2026-11-26T00:00:00Z","schema_version":1,"qrels_schema_version":1,"qrels_label":"q","collection":"c","embed_model":"m","summary":{nested},"results":[]}}"#
        )
    }

    /// `results` の `at_index` 番目（0-based）だけに `answer` を持つ成果物。
    fn artifact_with_results_answer(at_index: usize) -> String {
        let entries: Vec<String> = (0..=1001)
            .map(|index| {
                if index == at_index {
                    r#"{"id":"q","answer":"SECRET_VALUE"}"#.to_string()
                } else {
                    r#"{"id":"q"}"#.to_string()
                }
            })
            .collect();
        format!(
            r#"{{"generated_at":"2026-11-26T00:00:00Z","schema_version":1,"qrels_schema_version":1,"qrels_label":"q","collection":"c","embed_model":"m","summary":{{}},"results":[{}]}}"#,
            entries.join(",")
        )
    }

    #[test]
    fn eval_results_gate_requires_offset_on_non_cloud_generated_at() {
        let json = r#"{
            "generated_at":"2026-11-26T00:00:00",
            "schema_version":1,"qrels_schema_version":1,"qrels_label":"q",
            "collection":"c","embed_model":"m","summary":{},"results":[]
        }"#;
        let files = vec![("eval-results-baseline.json".to_string(), json.to_string())];
        let report = check_eval_results(&files);
        assert!(!report.ok);
        assert!(report
            .diagnostics
            .iter()
            .any(|line| { line.contains("generated_at requires RFC3339 UTC") }));
    }

    #[test]
    fn eval_results_gate_accepts_utc_offset_on_non_cloud_generated_at() {
        let json = r#"{
            "generated_at":"2026-11-26T00:00:00+00:00",
            "schema_version":1,"qrels_schema_version":1,"qrels_label":"q",
            "collection":"c","embed_model":"m","summary":{},"results":[]
        }"#;
        let files = vec![(
            "eval-results-ruri-subset.json".to_string(),
            json.to_string(),
        )];
        let report = check_eval_results(&files);
        assert!(report.ok);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn eval_results_gate_detects_nested_forbidden_key() {
        let files = vec![(
            "eval-results.json".to_string(),
            artifact_with_nested_answer(1),
        )];
        let report = check_eval_results(&files);
        assert!(!report.ok);
        let joined = report.diagnostics.join("\n");
        assert!(joined.contains("answer"));
        assert!(!joined.contains("SECRET_VALUE"));
    }

    #[test]
    fn eval_results_gate_depth_limit_stops_recursion() {
        // summary(深さ1) + 6 段 = 禁止キーのあるオブジェクトが深さ7 → 検出される。
        let within = vec![(
            "eval-results-baseline.json".to_string(),
            artifact_with_nested_answer(6),
        )];
        assert!(!check_eval_results(&within).ok);

        // 7 段では深さ8 となり上限に達するため走査されない。
        let beyond = vec![(
            "eval-results-baseline.json".to_string(),
            artifact_with_nested_answer(7),
        )];
        let report = check_eval_results(&beyond);
        assert!(report.ok);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn eval_results_gate_array_item_limit_stops_recursion() {
        // 先頭 1000 要素（index 0..=999）は走査される。
        let within = vec![(
            "eval-results.json".to_string(),
            artifact_with_results_answer(999),
        )];
        assert!(!check_eval_results(&within).ok);

        // 1001 要素目（index 1000）は上限を超えるため走査されない。
        let beyond = vec![(
            "eval-results.json".to_string(),
            artifact_with_results_answer(1000),
        )];
        let report = check_eval_results(&beyond);
        assert!(report.ok);
        assert!(report.diagnostics.is_empty());
    }

    #[test]
    fn rrf_fuse_uses_k60_and_top_k() {
        let first = vec![serde_json::json!({"id": "x"})];
        let second = vec![
            serde_json::json!({"id": "y"}),
            serde_json::json!({"id": "x"}),
        ];
        let fused = rrf_fuse(&[first, second], 5);
        assert_eq!(fused.len(), 2);
        let x = fused
            .iter()
            .find(|item| item.get("id").and_then(serde_json::Value::as_str) == Some("x"))
            .unwrap();
        let score = x.get("score").and_then(serde_json::Value::as_f64).unwrap();
        assert!((score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);

        let five: Vec<serde_json::Value> = (0..5)
            .map(|index| serde_json::json!({"id": index}))
            .collect();
        assert_eq!(rrf_fuse(&[five], 2).len(), 2);
    }
}
