//! qrels に基づく検索品質評価（Rust のみ）。

use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use crate::config::Config;
use crate::query::{search, SEARCH_MODES};
use crate::{ndcg_at, recall_at};

const RECALL_K: [usize; 4] = [1, 3, 5, 10];
const NDCG_K: [usize; 2] = [3, 10];

/// 検索中の部分失敗を旧 Python の record.error 相当の文字列へまとめる。
fn merge_errors(errors: &[crate::query::SearchError]) -> Value {
    if errors.is_empty() {
        Value::Null
    } else {
        Value::String(
            errors
                .iter()
                .map(|error| format!("{}: {}: {}", error.stage, error.error_type, error.message))
                .collect::<Vec<_>>()
                .join("; "),
        )
    }
}

fn string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

pub fn run(config: &Config, qrels_path: &Path, output: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(qrels_path).map_err(|e| format!("qrels read failed: {e}"))?;
    let qrels: Value =
        serde_json::from_str(&raw).map_err(|e| format!("invalid qrels JSON: {e}"))?;
    let simple = qrels["simple_queries"]
        .as_array()
        .ok_or("simple_queries must be an array")?;
    let heldout = qrels["heldout_queries"]
        .as_array()
        .ok_or("heldout_queries must be an array")?;

    let retrieval_k = config.top_k.max(10);
    let mut results = Vec::new();
    let mut heldout_results = Vec::new();

    for query in simple {
        let question = query["query"].as_str().unwrap_or_default();
        let expected = string_array(query.get("expected"));
        for mode in SEARCH_MODES {
            let result = search(config, question, mode, retrieval_k)?;
            let hits = result.hits;
            let error = merge_errors(&result.errors);
            let rerank_unchanged = crate::search::rerank_unchanged_status(result.unchanged);
            let sources: Vec<String> = hits
                .iter()
                .map(|hit| hit["source"].as_str().unwrap_or_default().to_string())
                .collect();
            let matched: Vec<String> = sources
                .iter()
                .filter(|source| {
                    expected
                        .iter()
                        .any(|word| source.to_lowercase().contains(&word.to_lowercase()))
                })
                .cloned()
                .collect();
            results.push(json!({
                "query": question,
                "expected": expected,
                "mode": mode,
                "hits": hits.len(),
                "error": error,
                "rerank_unchanged": rerank_unchanged,
                "recall": !matched.is_empty(),
                "matched_sources": matched,
                "sources": sources,
            }));
        }
    }

    for query in heldout {
        let question = query["query"].as_str().unwrap_or_default();
        let relevant = string_array(query.get("relevant_sources"));
        for mode in SEARCH_MODES {
            let result = search(config, question, mode, retrieval_k)?;
            let hits = result.hits;
            let error = merge_errors(&result.errors);
            let rerank_unchanged = crate::search::rerank_unchanged_status(result.unchanged);
            let sources: Vec<String> = hits
                .iter()
                .map(|hit| hit["source"].as_str().unwrap_or_default().to_string())
                .collect();
            let mut recall = serde_json::Map::new();
            for k in RECALL_K {
                recall.insert(
                    k.to_string(),
                    json!(recall_at(&relevant, &sources, k) == 1.0),
                );
            }
            let mut ndcg = serde_json::Map::new();
            for k in NDCG_K {
                let raw_score = ndcg_at(&relevant, &sources, k);
                ndcg.insert(
                    k.to_string(),
                    json!((raw_score * 1_000_000.0).round() / 1_000_000.0),
                );
            }
            heldout_results.push(json!({
                "id": query["id"].as_str().unwrap_or_default(),
                "query": question,
                "relevant_sources": relevant,
                "mode": mode,
                "retrieval_k": retrieval_k,
                "hits": hits.len(),
                "error": error,
                "rerank_unchanged": rerank_unchanged,
                "sources": sources,
                "recall_at": Value::Object(recall),
                "ndcg_at": Value::Object(ndcg),
            }));
        }
    }

    let artifact = json!({
        "generated_at": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "schema_version": 2,
        "qrels_schema_version": qrels["schema_version"].as_u64().unwrap_or(1),
        "qrels_label": qrels_path.file_name().and_then(|v| v.to_str()).unwrap_or("qrels.json"),
        "collection": config.collection,
        "embed_model": config.embed_model,
        "chat_model": config.chat_model,
        "vector_size": config.vector_size,
        "modes": SEARCH_MODES,
        "top_k": config.top_k,
        "summary": {},
        "results": results,
        "heldout": { "results": heldout_results },
    });
    let text = serde_json::to_string_pretty(&artifact).map_err(|e| e.to_string())?;
    fs::write(output, text).map_err(|e| format!("output write failed: {e}"))?;
    eprintln!("[evaluate] wrote {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_array_ignores_non_strings() {
        let value = json!(["a", 1, "b"]);
        assert_eq!(string_array(Some(&value)), vec!["a", "b"]);
    }
}
