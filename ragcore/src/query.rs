//! 検索モードの統合（HTTP クライアントと純粋計算の橋渡し）。

use serde_json::{json, Value};

use crate::cloud::CloudClient;
use crate::config::Config;
use crate::llama::LlamaClient;
use crate::qdrant::QdrantClient;
use crate::search::{keyword_score, query_terms, rank_hybrid, rank_keyword, source_of};

pub const SEARCH_MODES: [&str; 4] = ["dense", "keyword", "hybrid", "hybrid-rerank"];

/// 埋め込み先の選択結果（HTTP を伴わない純粋判定の戻り値）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbedderKind {
    Local,
    Cloud,
}

/// `cloud.enabled` だけを根拠に埋め込み先を決める。
fn embedder_kind(config: &Config) -> EmbedderKind {
    if config.cloud.enabled {
        EmbedderKind::Cloud
    } else {
        EmbedderKind::Local
    }
}

/// 埋め込みクライアントを束ねる内部 enum（このファイル内で完結）。
enum Embedder<'a> {
    Local(&'a LlamaClient),
    Cloud(&'a CloudClient),
}

impl Embedder<'_> {
    /// llama は prefix を引数で渡し、cloud は self 側で付与してから渡す。
    fn embed(&self, texts: &[String], prefix: &str) -> Result<Vec<Vec<f64>>, String> {
        match self {
            Embedder::Local(client) => client.embed(texts, prefix, None),
            Embedder::Cloud(client) => {
                // 旧 Python 契約に合わせ、クラウド経路では prefix を付与しない。
                client.embed(texts)
            }
        }
    }
}

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
    let texts = [question.to_string()];
    let vectors = match embedder_kind(config) {
        EmbedderKind::Local => {
            let client = LlamaClient::new(
                &config.llama_server_url,
                &config.embed_model,
                config.llama_timeout_secs,
            );
            Embedder::Local(&client).embed(&texts, &config.embed_prefix_query)?
        }
        EmbedderKind::Cloud => {
            let client = CloudClient::from_config(config)?;
            Embedder::Cloud(&client).embed(&texts, &config.embed_prefix_query)?
        }
    };
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
    use crate::config::CloudConfig;

    fn test_config() -> Config {
        Config {
            corpus_dir: std::path::PathBuf::from("docs"),
            collection: "c".to_string(),
            llama_server_url: "http://127.0.0.1:8090".to_string(),
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
            llama_timeout_secs: 300.0,
            qdrant_timeout_secs: 300.0,
            answer_timeout_secs: 60.0,
            cloud: CloudConfig {
                enabled: false,
                chat_enabled: false,
                base_url: "https://opencode.ai/inference/go/openai/v1".to_string(),
                embed_model: String::new(),
                embed_dimension: None,
                model: "deepseek-v4.1-flash".to_string(),
                timeout_secs: 60.0,
                api_key: None,
                api_key_name: None,
            },
        }
    }

    #[test]
    fn modes_match_python() {
        assert_eq!(
            SEARCH_MODES,
            ["dense", "keyword", "hybrid", "hybrid-rerank"]
        );
    }

    #[test]
    fn unknown_mode_is_rejected() {
        let config = test_config();
        assert!(search(&config, "q", "bogus", 10).is_err());
    }

    #[test]
    fn embedder_kind_follows_cloud_enabled_flag() {
        let mut config = test_config();
        assert_eq!(embedder_kind(&config), EmbedderKind::Local);
        config.cloud.enabled = true;
        assert_eq!(embedder_kind(&config), EmbedderKind::Cloud);
    }
}
