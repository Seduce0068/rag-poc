//! 環境変数からの設定読み込みと検証（Python config.py の代替）。

use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub corpus_dir: PathBuf,
    pub collection: String,
    pub llama_server_url: String,
    pub qdrant_url: String,
    pub embed_model: String,
    pub chat_model: String,
    pub vector_size: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
    pub question_max_chars: usize,
    pub rerank_max_docs: usize,
    pub embed_prefix_query: String,
    pub embed_prefix_passage: String,
    pub llama_timeout_secs: f64,
    pub qdrant_timeout_secs: f64,
    pub answer_timeout_secs: f64,
}

fn env_str(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a non-negative integer")),
        Err(_) => Ok(default),
    }
}

fn env_f64(name: &str, default: f64) -> Result<f64, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<f64>()
            .map_err(|_| format!("{name} must be a number")),
        Err(_) => Ok(default),
    }
}

impl Config {
    pub fn from_env() -> Result<Self, String> {
        let embed_model = env_str("EMBED_MODEL", "qwen2.5:3b");
        let (default_query, default_passage) = if embed_model.to_lowercase().contains("ruri") {
            ("クエリ: ", "文章: ")
        } else {
            ("", "")
        };
        let config = Self {
            corpus_dir: PathBuf::from(env_str("RAG_CORPUS_DIR", "passkey-gateway/docs")),
            collection: env_str("RAG_COLLECTION", "rag_poc_docs"),
            llama_server_url: env_str("LLAMA_SERVER_URL", "http://127.0.0.1:8090"),
            qdrant_url: env_str("QDRANT_URL", "http://127.0.0.1:6333"),
            embed_model,
            chat_model: env_str("CHAT_MODEL", "qwen2.5:3b"),
            vector_size: env_usize("RAG_VECTOR_SIZE", 2048)?,
            chunk_size: env_usize("CHUNK_SIZE", 800)?,
            chunk_overlap: env_usize("CHUNK_OVERLAP", 100)?,
            top_k: env_usize("RAG_TOP_K", 10)?,
            question_max_chars: env_usize("RAG_QUESTION_MAX_CHARS", 2000)?,
            rerank_max_docs: env_usize("RAG_RERANK_MAX_DOCS", 5)?,
            embed_prefix_query: env_str("RAG_EMBED_PREFIX_QUERY", default_query),
            embed_prefix_passage: env_str("RAG_EMBED_PREFIX_PASSAGE", default_passage),
            llama_timeout_secs: env_f64("LLAMA_TIMEOUT", 300.0)?,
            qdrant_timeout_secs: env_f64("QDRANT_TIMEOUT", 300.0)?,
            answer_timeout_secs: env_f64("RAG_ANSWER_TIMEOUT", 60.0)?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.chunk_size == 0 {
            return Err("CHUNK_SIZE must be positive".to_string());
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err("CHUNK_OVERLAP must be smaller than CHUNK_SIZE".to_string());
        }
        if self.top_k == 0 {
            return Err("RAG_TOP_K must be positive".to_string());
        }
        if self.vector_size == 0 {
            return Err("RAG_VECTOR_SIZE must be positive".to_string());
        }
        if self.embed_model.to_lowercase().contains("ruri") {
            if self.vector_size != 768 {
                return Err("Ruri requires RAG_VECTOR_SIZE=768".to_string());
            }
            if self.chunk_size > 512 {
                return Err("Ruri requires CHUNK_SIZE<=512".to_string());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Config {
        Config {
            corpus_dir: PathBuf::from("docs"),
            collection: "c".to_string(),
            llama_server_url: "http://127.0.0.1:11434".to_string(),
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
        }
    }

    #[test]
    fn default_config_is_valid() {
        assert!(base().validate().is_ok());
    }

    #[test]
    fn ruri_requires_768_and_small_chunks() {
        let mut config = base();
        config.embed_model = "kun432/cl-nagoya-ruri-base".to_string();
        config.vector_size = 768;
        config.chunk_size = 300;
        assert!(config.validate().is_ok());

        config.chunk_size = 800;
        assert!(config.validate().is_err());
    }

    #[test]
    fn defaults_match_python_config() {
        let config = base();
        assert_eq!(config.rerank_max_docs, 5);
        assert_eq!(config.llama_timeout_secs, 300.0);
        assert_eq!(config.qdrant_timeout_secs, 300.0);
    }
}
