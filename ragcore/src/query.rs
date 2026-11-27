//! 検索モードの統合（HTTP クライアントと純粋計算の橋渡し）。

use serde_json::{json, Value};

use crate::config::Config;
use crate::ollama::OllamaClient;
use crate::qdrant::QdrantClient;
use crate::search::{keyword_score, query_terms, rank_hybrid, rank_keyword, source_of};

pub const SEARCH_MODES: [&str; 4] = ["dense", "keyword", "hybrid", "hybrid-rerank"];

fn payload_text(point: &Value) -> String {
    for key in ["text", "content", "chunk", "body"] {
        if let Some(value) = point["payload"][key].as_str() {
            return value.to_string();
        }
    }
    String::new()
}

fn to_hit(point: &Value) -> Value {
    json!({
        "id": point.get("id").cloned().unwrap_or(Value::Null),
        "score": point.get("score").cloned().unwrap_or(Value::Null),
        "source": source_of(point),
        "text": payload_text(point),
    })
}

fn dense_points(config: &Config, question: &str, top_k: usize) -> Result<Vec<Value>, String> {
    let ollama = OllamaClient::new(
        &config.ollama_url,
        &config.embed_model,
        config.ollama_timeout_secs,
    );
    let vectors = ollama.embed(&[question.to_string()], &config.embed_prefix_query, None)?;
    let vector = vectors.into_iter().next().ok_or("no embedding returned")?;
    let qdrant = QdrantClient::new(
        &config.qdrant_url,
        &config.collection,
        config.qdrant_timeout_secs,
    );
    qdrant.query_dense(&vector, top_k)
}

fn keyword_points(config: &Config, question: &str, top_k: usize) -> Result<Vec<Value>, String> {
    let qdrant = QdrantClient::new(
        &config.qdrant_url,
        &config.collection,
        config.qdrant_timeout_secs,
    );
    let points = qdrant.scroll_all()?;
    let terms = query_terms(question);
    let scored: Vec<Value> = points
        .into_iter()
        .map(|mut point| {
            let score = keyword_score(&payload_text(&point), &terms);
            point["score"] = json!(score);
            point
        })
        .collect();
    Ok(rank_keyword(&scored, &terms, top_k))
}

pub fn search(
    config: &Config,
    question: &str,
    mode: &str,
    top_k: usize,
) -> Result<Vec<Value>, String> {
    if !SEARCH_MODES.contains(&mode) {
        return Err(format!("unknown search mode: {mode}"));
    }
    let points = match mode {
        "dense" => dense_points(config, question, top_k)?,
        "keyword" => keyword_points(config, question, top_k)?,
        "hybrid" | "hybrid-rerank" => {
            let dense = dense_points(config, question, top_k).unwrap_or_default();
            let keyword = keyword_points(config, question, top_k).unwrap_or_default();
            rank_hybrid(&dense, &keyword, top_k)
        }
        _ => unreachable!(),
    };
    Ok(points.iter().map(to_hit).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_match_python() {
        assert_eq!(
            SEARCH_MODES,
            ["dense", "keyword", "hybrid", "hybrid-rerank"]
        );
    }

    #[test]
    fn unknown_mode_is_rejected() {
        use crate::config::Config;
        let mut config = Config {
            corpus_dir: std::path::PathBuf::from("docs"),
            collection: "c".to_string(),
            ollama_url: "http://127.0.0.1:11434".to_string(),
            qdrant_url: "http://127.0.0.1:6333".to_string(),
            embed_model: "qwen2.5:3b".to_string(),
            chat_model: "qwen2.5:3b".to_string(),
            vector_size: 2048,
            chunk_size: 800,
            chunk_overlap: 100,
            top_k: 10,
            question_max_chars: 2000,
            rerank_max_docs: 5,
            embed_prefix_query: String::new(),
            embed_prefix_passage: String::new(),
            ollama_timeout_secs: 300.0,
            qdrant_timeout_secs: 300.0,
            answer_timeout_secs: 60.0,
        };
        config.vector_size = 2048;
        assert!(search(&config, "q", "bogus", 10).is_err());
    }
}
