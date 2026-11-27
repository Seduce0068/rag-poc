//! 環境変数からの設定読み込みと検証（Python config.py の代替）。

use std::env;
use std::path::PathBuf;

/// opencodeGO（クラウド）の設定。既定では一切有効化されない。
#[derive(Clone)]
pub struct CloudConfig {
    pub enabled: bool,
    pub chat_enabled: bool,
    pub base_url: String,
    pub embed_model: String,
    pub embed_dimension: Option<usize>,
    pub model: String,
    pub timeout_secs: f64,
    pub api_key: Option<String>,
    pub api_key_name: Option<&'static str>,
}

impl std::fmt::Debug for CloudConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudConfig")
            .field("enabled", &self.enabled)
            .field("chat_enabled", &self.chat_enabled)
            .field("base_url", &self.base_url)
            .field("embed_model", &self.embed_model)
            .field("embed_dimension", &self.embed_dimension)
            .field("model", &self.model)
            .field("timeout_secs", &self.timeout_secs)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("api_key_name", &self.api_key_name)
            .finish()
    }
}

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
    pub cloud: CloudConfig,
}

fn env_str(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_flag(name: &str) -> bool {
    env::var(name).map(|v| v == "1").unwrap_or(false)
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a non-negative integer")),
        Err(_) => Ok(default),
    }
}

fn env_usize_opt(name: &str) -> Result<Option<usize>, String> {
    match env::var(name) {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => value
            .trim()
            .parse::<usize>()
            .map(Some)
            .map_err(|_| format!("{name} must be a positive integer")),
        Err(_) => Ok(None),
    }
}

fn env_f64(name: &str, default: f64) -> f64 {
    match env::var(name) {
        Ok(value) => value
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|number| number.is_finite())
            .unwrap_or(default),
        Err(_) => default,
    }
}

/// 既知のクラウド埋め込みモデルの期待次元。
fn known_cloud_dimension(model: &str) -> Option<usize> {
    match model {
        "text-embedding-3-small" => Some(1536),
        "qwen3-embedding-8b" => Some(4096),
        "gemini-embedding-2" => Some(3072),
        _ => None,
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

        // API キーは値をログへ出さず、参照した環境変数名だけを保持する。
        let (api_key, api_key_name) = match env::var("OPENCODE_GO_API_KEY") {
            Ok(value) if !value.is_empty() => (Some(value), Some("OPENCODE_GO_API_KEY")),
            _ => match env::var("OPENCODE_API_KEY") {
                Ok(value) if !value.is_empty() => (Some(value), Some("OPENCODE_API_KEY")),
                _ => (None, None),
            },
        };
        let cloud = CloudConfig {
            enabled: env_flag("RAG_CLOUD_ENABLED"),
            chat_enabled: env_flag("RAG_CLOUD_CHAT_ENABLED"),
            base_url: env_str(
                "RAG_CLOUD_BASE_URL",
                "https://opencode.ai/inference/go/openai/v1",
            ),
            embed_model: env_str("RAG_CLOUD_EMBED_MODEL", ""),
            embed_dimension: env_usize_opt("RAG_CLOUD_EMBED_DIMENSION")?
                .or(env_usize_opt("RAG_CLOUD_DIMENSIONS")?),
            model: env_str("RAG_CLOUD_MODEL", "deepseek-v4.1-flash"),
            timeout_secs: env_f64("RAG_CLOUD_TIMEOUT", 60.0).max(1.0),
            api_key,
            api_key_name,
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
            llama_timeout_secs: env_f64("LLAMA_TIMEOUT", 300.0),
            qdrant_timeout_secs: env_f64("QDRANT_TIMEOUT", 300.0),
            answer_timeout_secs: env_f64("RAG_ANSWER_TIMEOUT", 60.0),
            cloud,
        };
        config.validate()?;
        Ok(config)
    }

    /// クラウド埋め込みの期待次元を解決する（明示 > 既知モデル表）。
    pub fn cloud_embed_dimension(&self) -> Option<usize> {
        self.cloud
            .embed_dimension
            .or_else(|| known_cloud_dimension(&self.cloud.embed_model))
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
        self.validate_cloud_settings()?;
        Ok(())
    }

    /// クラウド経路の fail-closed 検証。既定（無効）では何も要求しない。
    fn validate_cloud_settings(&self) -> Result<(), String> {
        if !self.cloud.enabled && !self.cloud.chat_enabled {
            return Ok(());
        }
        if self.cloud.chat_enabled && !self.cloud.enabled {
            return Err("RAG_CLOUD_CHAT_ENABLED=1 requires RAG_CLOUD_ENABLED=1".to_string());
        }
        let endpoint = url::Url::parse(&self.cloud.base_url)
            .map_err(|_| "RAG_CLOUD_BASE_URL is invalid".to_string())?;
        if endpoint.scheme() != "https" {
            return Err("RAG_CLOUD_BASE_URL must use https".to_string());
        }
        if self.cloud.embed_model.is_empty() {
            return Err("RAG_CLOUD_EMBED_MODEL must be set explicitly".to_string());
        }
        let Some(dim) = self.cloud_embed_dimension() else {
            return Err(
                "unknown RAG_CLOUD_EMBED_MODEL requires RAG_CLOUD_EMBED_DIMENSION".to_string(),
            );
        };
        if dim != self.vector_size {
            return Err("RAG_VECTOR_SIZE does not match cloud embedding dimension".to_string());
        }
        if self.cloud.api_key.is_none() {
            let name = self.cloud.api_key_name.unwrap_or("OPENCODE_GO_API_KEY");
            return Err(format!("{name} is required for cloud mode"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cloud() -> CloudConfig {
        CloudConfig {
            enabled: false,
            chat_enabled: false,
            base_url: "https://opencode.ai/inference/go/openai/v1".to_string(),
            embed_model: String::new(),
            embed_dimension: None,
            model: "deepseek-v4.1-flash".to_string(),
            timeout_secs: 60.0,
            api_key: None,
            api_key_name: None,
        }
    }

    fn base() -> Config {
        Config {
            corpus_dir: PathBuf::from("docs"),
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
            cloud: base_cloud(),
        }
    }

    #[test]
    fn default_config_is_valid_and_cloud_is_off() {
        let config = base();
        assert!(config.validate().is_ok());
        assert!(!config.cloud.enabled);
        assert!(!config.cloud.chat_enabled);
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
    fn cloud_chat_alone_is_rejected() {
        let mut config = base();
        config.cloud.chat_enabled = true;
        let err = config.validate().unwrap_err();
        assert!(err.contains("RAG_CLOUD_ENABLED"));
    }

    #[test]
    fn cloud_requires_explicit_model_dimension_and_key() {
        let mut config = base();
        config.cloud.enabled = true;
        assert!(config.validate().is_err());

        config.cloud.embed_model = "text-embedding-3-small".to_string();
        config.vector_size = 1536;
        assert!(config.validate().is_err());

        config.cloud.api_key = Some("dummy-test-key".to_string());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn unknown_cloud_model_requires_explicit_dimension() {
        let mut config = base();
        config.cloud.enabled = true;
        config.cloud.embed_model = "custom-model".to_string();
        config.cloud.api_key = Some("dummy-test-key".to_string());
        config.vector_size = 512;
        assert!(config.validate().is_err());

        config.cloud.embed_dimension = Some(512);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn dimension_precedence_is_explicit_then_known_table() {
        let mut config = base();
        config.cloud.embed_model = "text-embedding-3-small".to_string();
        assert_eq!(config.cloud_embed_dimension(), Some(1536));
        config.cloud.embed_dimension = Some(512);
        assert_eq!(config.cloud_embed_dimension(), Some(512));
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let mut config = base();
        config.cloud.api_key = Some("DUMMY_SECRET_VALUE".to_string());
        let text = format!("{:?}", config);
        assert!(!text.contains("DUMMY_SECRET_VALUE"));
        assert!(text.contains("<redacted>"));
    }

    #[test]
    fn cloud_endpoint_must_use_https() {
        let mut config = base();
        config.cloud.enabled = true;
        config.cloud.base_url = "http://example.invalid/v1".to_string();
        config.cloud.embed_model = "text-embedding-3-small".to_string();
        config.vector_size = 1536;
        config.cloud.api_key = Some("dummy".to_string());
        assert!(config.validate().is_err());
    }
}
