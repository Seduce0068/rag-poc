//! Python `query.py` の検索ロジック（HTTP を除く純粋計算）の Rust 実装。
//!
//! 基準となる Python 関数:
//! - `_query_terms` / `_clamp01` / `_text_of` / `_source_of`
//! - `search_keyword` / `rrf_fuse`
//! - `_extract_json` / `_parse_scores` / `rerank`
//!
//! HTTP 呼び出しは含まない（別担当）。`unsafe` は使用しない。

use std::cmp::Ordering;
use std::collections::HashSet;

use serde_json::{json, Value};

// 同一 crate の関数を Python 側と同じ `ragcore::` パスで参照する。
extern crate self as ragcore;
use ragcore::rrf_fuse;

const TEXT_KEYS: &[&str] = &["text", "content", "chunk", "body"];
const SOURCE_KEYS: &[&str] = &[
    "source",
    "title",
    "path",
    "file_path",
    "file",
    "filename",
    "url",
];

/// Unicode の `[^\W_]+` 相当（英数字＋日本語、アンダースコア除外）で分割し、
/// 小文字化・重複除去した語を出現順に返す（Python `_query_terms` 相当）。
pub fn query_terms(question: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut token = String::new();

    for ch in question.chars() {
        if ch.is_alphanumeric() {
            token.push(ch);
        } else if !token.is_empty() {
            let lowered = token.to_lowercase();
            token.clear();
            if seen.insert(lowered.clone()) {
                terms.push(lowered);
            }
        }
    }
    if !token.is_empty() {
        let lowered = token.to_lowercase();
        if seen.insert(lowered.clone()) {
            terms.push(lowered);
        }
    }
    terms
}

/// text を小文字化し、各 term が部分文字列として含まれる数を返す。
/// 同じ term は 1 カウント（Python `search_keyword` の `matched` 相当）。
pub fn keyword_score(text: &str, terms: &[String]) -> f64 {
    let lowered = text.to_lowercase();
    terms
        .iter()
        .filter(|term| lowered.contains(term.as_str()))
        .count() as f64
}

/// 本文キーを `text/content/chunk/body` の順で探す。
fn first_text(value: &Value) -> Option<String> {
    TEXT_KEYS
        .iter()
        .filter_map(|key| value.get(key).and_then(Value::as_str))
        .find(|text| !text.is_empty())
        .map(str::to_string)
}

/// payload 付きの生 point と、`text` をトップレベルに持つ正規化済み point の両方を扱う。
pub fn text_of(point: &Value) -> String {
    point
        .get("payload")
        .filter(|value| value.is_object())
        .and_then(first_text)
        .or_else(|| first_text(point))
        .unwrap_or_default()
}

/// Python の `str(value)` 相当（文字列は引用符なし、それ以外は JSON 表記）。
fn display_value(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => "null".to_string(),
    }
}

/// Python の `if value:` 相当の truthy 判定を伴う出典文字列。
fn truthy_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::Bool(false) => None,
        Value::String(text) if text.is_empty() => None,
        Value::Number(number) if number.as_f64() == Some(0.0) => None,
        Value::Array(items) if items.is_empty() => None,
        Value::Object(items) if items.is_empty() => None,
        Value::String(text) => Some(text.clone()),
        other => Some(display_value(Some(other))),
    }
}

fn first_source(value: &Value) -> Option<String> {
    SOURCE_KEYS
        .iter()
        .filter_map(|key| value.get(key).and_then(truthy_text))
        .next()
}

/// 出典を `source/title/path/file_path/file/filename/url` → metadata 内 →
/// `point:<id>` の順で取得する（Python `_source_of` 相当）。
pub fn source_of(point: &Value) -> String {
    if let Some(payload) = point.get("payload").filter(|value| value.is_object()) {
        if let Some(found) = first_source(payload) {
            return found;
        }
        if let Some(metadata) = payload.get("metadata").filter(|value| value.is_object()) {
            if let Some(found) = first_source(metadata) {
                return found;
            }
        }
    }
    if let Some(found) = first_source(point) {
        return found;
    }
    if let Some(metadata) = point.get("metadata").filter(|value| value.is_object()) {
        if let Some(found) = first_source(metadata) {
            return found;
        }
    }
    format!("point:{}", display_value(point.get("id")))
}

/// スコアが正の point だけを `(-score, str(id))` でソートし `top_k` 件に截断する。
/// 各 point に `score` / `text` / `source` を設定する（Python `search_keyword` 相当）。
pub fn rank_keyword(points: &[Value], terms: &[String], top_k: usize) -> Vec<Value> {
    let mut scored: Vec<Value> = Vec::new();
    for point in points {
        let text = text_of(point);
        let score = keyword_score(&text, terms);
        if score <= 0.0 {
            continue;
        }
        let mut item = point.clone();
        if let Some(object) = item.as_object_mut() {
            object.insert("score".to_string(), json!(score));
            object.insert("text".to_string(), json!(text));
            object.insert("source".to_string(), json!(source_of(point)));
        }
        scored.push(item);
    }

    scored.sort_by(|a, b| {
        let score_a = a.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        let score_b = b.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| display_value(a.get("id")).cmp(&display_value(b.get("id"))))
    });
    scored.truncate(top_k);
    scored
}

/// dense / keyword の順位を既存の `rrf_fuse` (k=60) で統合する。
pub fn rank_hybrid(dense: &[Value], keyword: &[Value], top_k: usize) -> Vec<Value> {
    rrf_fuse(&[dense.to_vec(), keyword.to_vec()], top_k)
}

/// 非有限値は 0.0、それ以外は 0..1 へ丸める（Python `_clamp01` 相当）。
pub fn clamp01(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}

/// LLM 出力から JSON を取り出す。コードフェンスを外し、失敗時は
/// `[`...`]` / `{`...`}` の最初と最後で切り出して再挑戦する
/// （Python `_extract_json` 相当）。
pub fn extract_json(text: &str) -> Option<Value> {
    let mut stripped = text.trim();
    if let Some(rest) = stripped.strip_prefix("```") {
        stripped = rest.strip_prefix("json").unwrap_or(rest).trim_start();
    }
    if let Some(rest) = stripped.strip_suffix("```") {
        stripped = rest.trim_end();
    }

    if let Ok(value) = serde_json::from_str(stripped) {
        return Some(value);
    }

    for (opener, closer) in [('[', ']'), ('{', '}')] {
        if let (Some(start), Some(end)) = (stripped.find(opener), stripped.rfind(closer)) {
            if end > start {
                if let Ok(value) = serde_json::from_str(&stripped[start..=end]) {
                    return Some(value);
                }
            }
        }
    }
    None
}

/// Python `int(value or 0)` 相当。不正値は `None`（欠損スロット扱い）。
fn to_index(value: &Value) -> Option<i64> {
    match value {
        Value::Null => Some(0),
        Value::Bool(false) => Some(0),
        Value::Bool(true) => Some(1),
        Value::String(text) if text.trim().is_empty() => Some(0),
        Value::String(text) => text.trim().parse::<i64>().ok(),
        Value::Number(number) => match number.as_f64() {
            Some(value) if value.is_finite() => Some(value.trunc() as i64),
            _ => None,
        },
        Value::Array(items) if items.is_empty() => Some(0),
        Value::Object(items) if items.is_empty() => Some(0),
        _ => None,
    }
}

/// Python `float(value)` 相当（bool は 1.0 / 0.0、数値文字列も許容）。
fn to_score(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::Bool(true) => Some(1.0),
        Value::Bool(false) => Some(0.0),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// JSON から期待件数分の 0..1 スコアを取り出す（Python `_parse_scores` 相当）。
/// dict は `scores/results/items` の順に配列を探し、長さが `expected` で
/// 全要素が bool 以外の数値ならそのまま clamp する。それ以外は各要素の
/// `index/id`（既定 0）を int 化し、`strict_1based` では 1..=expected のみ有効、
/// legacy では 0 が含まれれば 0-based と推定する。欠損スロットは `None`。
pub fn parse_scores(raw: &str, expected: usize, strict_1based: bool) -> Vec<Option<f64>> {
    let data: Value = match serde_json::from_str(raw) {
        Ok(value) => value,
        Err(_) => match extract_json(raw) {
            Some(value) => value,
            None => return vec![None; expected],
        },
    };

    let data = if data.is_object() {
        match data
            .get("scores")
            .or_else(|| data.get("results"))
            .or_else(|| data.get("items"))
        {
            Some(value) => value.clone(),
            None => return vec![None; expected],
        }
    } else {
        data
    };

    let array = match data.as_array() {
        Some(items) if !items.is_empty() => items,
        _ => return vec![None; expected],
    };

    // 数値のみの配列は位置そのものをスコアとして扱う（bool は除外）。
    if array.len() == expected && array.iter().all(Value::is_number) {
        return array
            .iter()
            .map(|item| Some(clamp01(item.as_f64().unwrap_or(0.0))))
            .collect();
    }

    let parsed_indices: Vec<Option<i64>> = array
        .iter()
        .map(|entry| match entry {
            Value::Object(_) => {
                let raw_index = entry
                    .get("index")
                    .or_else(|| entry.get("id"))
                    .unwrap_or(&Value::Null);
                to_index(raw_index)
            }
            _ => None,
        })
        .collect();

    let dict_indices: Vec<i64> = parsed_indices.iter().flatten().copied().collect();
    let has_zero = dict_indices.contains(&0);
    let inferred = !dict_indices.is_empty()
        && !has_zero
        && dict_indices.iter().copied().min().unwrap_or(0) >= 1
        && dict_indices.iter().copied().max().unwrap_or(0) <= expected as i64;
    // strict で 1-based を固定する。ただし index に 0 が含まれる古い形式は
    // 従来推定へ戻す（Python の移行期互換経路）。
    let one_based = (strict_1based && !dict_indices.is_empty() && !has_zero) || inferred;

    let mut scores: Vec<Option<f64>> = vec![None; expected];
    for (position, item) in array.iter().enumerate() {
        match item {
            Value::Object(_) => {
                let raw_score = item
                    .get("score")
                    .or_else(|| item.get("relevance"))
                    .or_else(|| item.get("value"));
                let score = match raw_score.and_then(to_score) {
                    Some(score) => score,
                    None => continue,
                };
                let index = match parsed_indices[position] {
                    Some(index) => index,
                    None => continue,
                };
                let index = if one_based {
                    index.saturating_sub(1)
                } else {
                    index
                };
                if index >= 0 && (index as usize) < expected {
                    scores[index as usize] = Some(clamp01(score));
                }
            }
            value if value.is_number() && position < expected => {
                scores[position] = Some(clamp01(value.as_f64().unwrap_or(0.0)));
            }
            _ => {}
        }
    }
    scores
}

/// 各 point に `base_score` / `llm_score` / `final_score` / `score` /
/// `reranked_by_llm` を設定する（Python `rerank.finalize` 相当）。
/// 欠損した llm_score の `final_score` は 0.0、`llm_score` は null とする。
pub fn rerank_scores(points: &mut [Value], scores: &[Option<f64>]) {
    for (position, point) in points.iter_mut().enumerate() {
        let base_score = point.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        let llm_score = scores.get(position).copied().flatten();
        let final_score = llm_score.map(clamp01).unwrap_or(0.0);
        if let Some(object) = point.as_object_mut() {
            object.insert("base_score".to_string(), json!(base_score));
            object.insert("llm_score".to_string(), json!(llm_score));
            object.insert("final_score".to_string(), json!(final_score));
            object.insert("score".to_string(), json!(final_score));
            object.insert("reranked_by_llm".to_string(), json!(true));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_terms_splits_unicode_and_drops_underscore() {
        assert_eq!(
            query_terms("Rust_Rust 日本語 テスト! RUST rust"),
            vec!["rust", "日本語", "テスト"]
        );
        assert_eq!(query_terms("___!!!"), Vec::<String>::new());
    }

    #[test]
    fn keyword_score_counts_each_term_once() {
        let terms = vec!["rust".to_string(), "漠".to_string()];
        assert_eq!(keyword_score("Rust is RUST and 日本語", &terms), 1.0);
        assert_eq!(keyword_score("rust 漠", &terms), 2.0);
        assert_eq!(keyword_score("", &terms), 0.0);
    }

    #[test]
    fn rank_keyword_filters_sorts_and_sets_fields() {
        let points = vec![
            json!({"id": "b", "payload": {"text": "rust rust"}}),
            json!({"id": "a", "payload": {"content": "rust"}}),
            json!({"id": "z", "payload": {"text": "none"}}),
            json!({"id": "c", "text": "rust", "source": "top"}),
        ];
        let ranked = rank_keyword(&points, &query_terms("rust"), 2);
        assert_eq!(ranked.len(), 2);
        // Python 仕様は語ごと 1 カウント。同点は str(id) でタイブレークする。
        assert_eq!(ranked[0].get("id").and_then(Value::as_str), Some("a"));
        assert_eq!(ranked[0].get("score").and_then(Value::as_f64), Some(1.0));
        assert_eq!(
            ranked[0].get("source").and_then(Value::as_str),
            Some("point:a")
        );
        assert_eq!(ranked[1].get("id").and_then(Value::as_str), Some("b"));
        assert_eq!(ranked[1].get("score").and_then(Value::as_f64), Some(1.0));
    }

    #[test]
    fn rank_keyword_breaks_ties_by_string_id() {
        let points = vec![
            json!({"id": 10, "payload": {"text": "rust"}}),
            json!({"id": 2, "payload": {"text": "rust"}}),
        ];
        let ranked = rank_keyword(&points, &query_terms("rust"), 2);
        // str(10) < str(2) なので 10 が先（数値比較ではない）。
        assert_eq!(ranked[0].get("id").and_then(Value::as_i64), Some(10));
        assert_eq!(ranked[1].get("id").and_then(Value::as_i64), Some(2));
    }

    #[test]
    fn rank_hybrid_matches_existing_rrf_fuse() {
        let dense = vec![json!({"id": "x"}), json!({"id": "y"})];
        let keyword = vec![json!({"id": "y"}), json!({"id": "x"})];
        let hybrid = rank_hybrid(&dense, &keyword, 5);
        let expected = rrf_fuse(&[dense.clone(), keyword.clone()], 5);
        assert_eq!(hybrid, expected);
        let score = hybrid[0].get("score").and_then(Value::as_f64).unwrap();
        assert!((score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
        assert_eq!(rank_hybrid(&dense, &keyword, 1).len(), 1);
    }

    #[test]
    fn clamp01_maps_nonfinite_and_out_of_range() {
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(-1.0), 0.0);
        assert_eq!(clamp01(2.0), 1.0);
        assert_eq!(clamp01(f64::NAN), 0.0);
        assert_eq!(clamp01(f64::INFINITY), 0.0);
    }

    #[test]
    fn parse_scores_handles_strict_legacy_and_out_of_range() {
        // strict: 1-based 固定
        let raw = r#"{"scores":[{"index":1,"score":0.9},{"index":2,"score":0.1}]}"#;
        assert_eq!(parse_scores(raw, 2, true), vec![Some(0.9), Some(0.1)]);
        // strict: 範囲外 index は無視（0-based へシフトしない）
        let raw = r#"[{"index":2,"score":0.8},{"index":9,"score":0.5}]"#;
        assert_eq!(parse_scores(raw, 2, true), vec![None, Some(0.8)]);
        // legacy: 0 を含むため 0-based と推定
        let raw = r#"[{"index":0,"score":0.7},{"index":1,"score":0.3}]"#;
        assert_eq!(parse_scores(raw, 2, false), vec![Some(0.7), Some(0.3)]);
        // strict でも 0-based 生データなら従来推定へフォールバック
        assert_eq!(parse_scores(raw, 2, true), vec![Some(0.7), Some(0.3)]);
    }

    #[test]
    fn parse_scores_accepts_plain_array_and_rejects_invalid_json() {
        assert_eq!(
            parse_scores("[0.5, 2.0, -1.0]", 3, false),
            vec![Some(0.5), Some(1.0), Some(0.0)]
        );
        assert_eq!(
            parse_scores(r#"{"items":[{"id":1,"relevance":0.4}]}"#, 2, true),
            vec![Some(0.4), None]
        );
        assert_eq!(parse_scores("not json", 2, false), vec![None, None]);
    }

    #[test]
    fn extract_json_strips_fences_and_slices_prose() {
        let fenced = "```json\n{\"scores\":[0.5]}\n```";
        assert_eq!(extract_json(fenced), Some(json!({"scores": [0.5]})));
        let prose = "結果: [{\"index\":1,\"score\":0.2}] です";
        assert_eq!(
            extract_json(prose),
            Some(json!([{"index": 1, "score": 0.2}]))
        );
        assert_eq!(extract_json("no json here"), None);
    }

    #[test]
    fn text_and_source_helpers_cover_payload_metadata_and_id() {
        let point = json!({
            "id": 7,
            "payload": {"chunk": "本文", "metadata": {"title": "メタタイトル"}}
        });
        assert_eq!(text_of(&point), "本文");
        assert_eq!(source_of(&point), "メタタイトル");
        let fallback = json!({"id": 7, "payload": {}});
        assert_eq!(source_of(&fallback), "point:7");
    }

    #[test]
    fn rerank_scores_marks_missing_llm_score_as_zero() {
        let mut points = vec![
            json!({"id": "a", "score": 0.8}),
            json!({"id": "b", "score": 0.4}),
        ];
        let scores = vec![Some(0.9), None];
        rerank_scores(&mut points, &scores);

        assert_eq!(
            points[0].get("base_score").and_then(Value::as_f64),
            Some(0.8)
        );
        assert_eq!(
            points[0].get("llm_score").and_then(Value::as_f64),
            Some(0.9)
        );
        assert_eq!(
            points[0].get("final_score").and_then(Value::as_f64),
            Some(0.9)
        );
        assert_eq!(points[0].get("score").and_then(Value::as_f64), Some(0.9));
        assert_eq!(
            points[0].get("reranked_by_llm").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(points[1].get("llm_score"), Some(&Value::Null));
        assert_eq!(
            points[1].get("final_score").and_then(Value::as_f64),
            Some(0.0)
        );
        assert_eq!(
            points[1].get("base_score").and_then(Value::as_f64),
            Some(0.4)
        );
    }
}
