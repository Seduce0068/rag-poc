//! 環境変数からの設定読み込みと検証（Python config.py の代替）。

use std::env;
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};

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
    /// コーパスディレクトリがリポジトリ外の場合だけ許可する（既定は拒否）。
    pub allow_external_corpus: bool,
    pub collection: String,
    pub llama_server_url: String,
    pub llama_chat_url: String,
    pub qdrant_url: String,
    pub allow_remote_endpoints: bool,
    pub allowed_endpoint_hosts: Vec<String>,
    pub embed_model: String,
    pub chat_model: String,
    pub vector_size: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
    pub question_max_chars: usize,
    pub rerank_max_docs: usize,
    pub rerank_prompt_style: String,
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

/// コンパイル時に埋め込まれた CARGO_MANIFEST_DIR（ragcore/）の親 = rag_poc ルート。
fn compile_time_repo_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap_or(manifest).to_path_buf()
}

/// Python の `Path.expanduser()` 相当（先頭の `~` を HOME に置換する）。
/// `~user` 形式は pwd 参照を避けるため未対応。
fn expand_tilde(path: &Path, home: Option<&OsStr>) -> PathBuf {
    let mut components = path.components();
    match (components.next(), home) {
        (Some(Component::Normal(first)), Some(home)) if first.to_str() == Some("~") => {
            let rest: PathBuf = components.collect();
            PathBuf::from(home).join(rest)
        }
        _ => path.to_path_buf(),
    }
}

/// ファイルシステムに触れずに `.` / `..` を解決する。
/// Python の `Path.resolve(strict=False)` 相当（シンボリックリンクは展開しない）。
fn normalize_lexically(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// リポジトリルート。CLI の CWD に依存しないよう、環境変数 `RAG_REPO_ROOT` を
/// 優先し、無ければコンパイル時の CARGO_MANIFEST_DIR の親を使う。
pub fn repo_root() -> PathBuf {
    let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"));
    match env::var_os("RAG_REPO_ROOT") {
        Some(value) if !value.is_empty() => {
            let expanded = expand_tilde(&PathBuf::from(value), home.as_deref());
            if expanded.is_absolute() {
                normalize_lexically(&expanded)
            } else {
                normalize_lexically(&env::current_dir().unwrap_or_default().join(expanded))
            }
        }
        _ => normalize_lexically(&compile_time_repo_root()),
    }
}

/// 旧 Python `resolve_corpus_dir()` 相当の純粋パス判定。
/// 相対パスは `root` 基準で絶対化し、`root` 配下でなければ
/// `allow_external` が true のときだけ許可する。エラーにはパスを出さない。
fn resolve_corpus_dir_inner(
    raw: &Path,
    root: &Path,
    allow_external: bool,
    home: Option<&OsStr>,
) -> Result<PathBuf, String> {
    let expanded = expand_tilde(raw, home);
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        root.join(expanded)
    };
    let resolved = normalize_lexically(&absolute);
    let root = normalize_lexically(root);
    if !resolved.starts_with(&root) && !allow_external {
        return Err(
            "RAG_CORPUS_DIR must be under the repository; set RAG_ALLOW_EXTERNAL_CORPUS=1"
                .to_string(),
        );
    }
    Ok(resolved)
}

impl Config {
    pub fn from_env() -> Result<Self, String> {
        let llama_server_url = env_str("LLAMA_SERVER_URL", "http://127.0.0.1:8090");
        let llama_chat_url = env_str("LLAMA_CHAT_URL", &llama_server_url);
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
            allow_external_corpus: env_flag("RAG_ALLOW_EXTERNAL_CORPUS"),
            collection: env_str("RAG_COLLECTION", "rag_poc_docs"),
            llama_server_url,
            llama_chat_url,
            qdrant_url: env_str("QDRANT_URL", "http://127.0.0.1:6333"),
            allow_remote_endpoints: env_flag("RAG_ALLOW_REMOTE_ENDPOINTS"),
            allowed_endpoint_hosts: env_str("RAG_ALLOWED_ENDPOINT_HOSTS", "")
                .split(',')
                .map(str::trim)
                .filter(|host| !host.is_empty())
                .map(str::to_lowercase)
                .collect(),
            embed_model,
            chat_model: env_str("CHAT_MODEL", "qwen2.5:3b"),
            vector_size: env_usize("RAG_VECTOR_SIZE", 2048)?,
            chunk_size: env_usize("CHUNK_SIZE", 800)?,
            chunk_overlap: env_usize("CHUNK_OVERLAP", 100)?,
            top_k: env_usize("RAG_TOP_K", 10)?,
            question_max_chars: env_usize("RAG_QUESTION_MAX_CHARS", 2000)?,
            rerank_max_docs: env_usize("RAG_RERANK_MAX_DOCS", 5)?,
            rerank_prompt_style: env_str("RAG_RERANK_PROMPT_STYLE", "strict"),
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

    /// 回答生成の実効タイムアウト（クラウド側の上限を超えない）。
    pub fn effective_answer_timeout(&self) -> f64 {
        self.answer_timeout_secs.min(self.cloud.timeout_secs)
    }

    /// クラウド埋め込みの期待次元を解決する（明示 > 既知モデル表）。
    pub fn cloud_embed_dimension(&self) -> Option<usize> {
        self.cloud
            .embed_dimension
            .or_else(|| known_cloud_dimension(&self.cloud.embed_model))
    }

    /// ingest 経路専用: コーパス dir を解決し、リポジトリ外を既定で拒否する。
    /// `validate()` は query/evaluate も通るため、ここには含めない。
    pub fn resolve_corpus_dir(&self, raw: &Path) -> Result<PathBuf, String> {
        let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"));
        let root = repo_root();
        let resolved =
            resolve_corpus_dir_inner(raw, &root, self.allow_external_corpus, home.as_deref())?;
        // 実体が存在するなら symlink を解決して範囲を再確認する。
        if let (Ok(real_path), Ok(real_root)) = (
            std::fs::canonicalize(&resolved),
            std::fs::canonicalize(&root),
        ) {
            if !real_path.starts_with(&real_root) && !self.allow_external_corpus {
                return Err(
                    "RAG_CORPUS_DIR must be under the repository; set RAG_ALLOW_EXTERNAL_CORPUS=1"
                        .to_string(),
                );
            }
            return Ok(real_path);
        }
        Ok(resolved)
    }

    /// 接続先 URL を検証する。loopback は既定許可、リモートは opt-in ＋ https ＋ 許可ホストが必須。
    fn validate_endpoint(&self, url: &str, name: &str) -> Result<(), String> {
        let parsed = url::Url::parse(url).map_err(|_| format!("{name} is invalid"))?;
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(format!("{name} must not contain credentials"));
        }
        let scheme = parsed.scheme();
        let host = parsed.host_str().unwrap_or_default().to_lowercase();
        let loopback = matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1" | "[::1]");
        if loopback {
            if scheme != "http" && scheme != "https" {
                return Err(format!("{name} has invalid scheme"));
            }
            return Ok(());
        }
        if scheme != "https" {
            return Err(format!("{name} must use https for remote hosts"));
        }
        if !self.allow_remote_endpoints {
            return Err(format!(
                "{name} remote host requires RAG_ALLOW_REMOTE_ENDPOINTS=1"
            ));
        }
        let host_port = match parsed.port() {
            Some(port) => format!("{host}:{port}"),
            None => host.clone(),
        };
        if !self
            .allowed_endpoint_hosts
            .iter()
            .any(|allowed| allowed == &host || allowed == &host_port)
        {
            return Err(format!("{name} host is not in RAG_ALLOWED_ENDPOINT_HOSTS"));
        }
        Ok(())
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
        if !self.answer_timeout_secs.is_finite() || self.answer_timeout_secs <= 0.0 {
            return Err("RAG_ANSWER_TIMEOUT must be positive".to_string());
        }
        if self.embed_model.to_lowercase().contains("ruri") {
            if self.vector_size != 768 {
                return Err("Ruri requires RAG_VECTOR_SIZE=768".to_string());
            }
            if self.chunk_size > 512 {
                return Err("Ruri requires CHUNK_SIZE<=512".to_string());
            }
        }
        self.validate_endpoint(&self.llama_server_url, "LLAMA_SERVER_URL")?;
        self.validate_endpoint(&self.llama_chat_url, "LLAMA_CHAT_URL")?;
        self.validate_endpoint(&self.qdrant_url, "QDRANT_URL")?;
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
            allow_external_corpus: false,
            collection: "c".to_string(),
            llama_server_url: "http://127.0.0.1:8090".to_string(),
            llama_chat_url: "http://127.0.0.1:8090".to_string(),
            qdrant_url: "http://127.0.0.1:6333".to_string(),
            allow_remote_endpoints: false,
            allowed_endpoint_hosts: Vec::new(),
            embed_model: "qwen2.5:3b".to_string(),
            chat_model: "qwen2.5:3b".to_string(),
            vector_size: 2048,
            chunk_size: 800,
            chunk_overlap: 100,
            top_k: 10,
            question_max_chars: 2000,
            rerank_max_docs: 5,
            rerank_prompt_style: "strict".to_string(),
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
    fn remote_endpoint_requires_opt_in_https_and_allowlist() {
        let mut config = base();
        config.llama_server_url = "http://example.invalid:8090".to_string();
        assert!(config.validate().is_err());

        config.llama_server_url = "https://example.invalid:8090".to_string();
        assert!(config.validate().is_err());

        config.allow_remote_endpoints = true;
        assert!(config.validate().is_err());

        config.allowed_endpoint_hosts = vec!["example.invalid:8090".to_string()];
        assert!(config.validate().is_ok());
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

    #[test]
    fn resolve_corpus_dir_anchors_relative_paths_to_repo_root() {
        let resolved =
            resolve_corpus_dir_inner(Path::new("docs"), Path::new("/repo"), false, None).unwrap();
        assert_eq!(resolved, PathBuf::from("/repo/docs"));
    }

    #[test]
    fn resolve_corpus_dir_normalizes_dot_and_parent_components() {
        let resolved = resolve_corpus_dir_inner(
            Path::new("/repo/a/../docs/./x"),
            Path::new("/repo"),
            false,
            None,
        )
        .unwrap();
        assert_eq!(resolved, PathBuf::from("/repo/docs/x"));
    }

    #[test]
    fn resolve_corpus_dir_rejects_external_path_without_opt_in() {
        let err = resolve_corpus_dir_inner(
            Path::new("/outside/corpus"),
            Path::new("/repo"),
            false,
            None,
        )
        .unwrap_err();
        assert!(err.contains("RAG_ALLOW_EXTERNAL_CORPUS=1"));
        assert!(!err.contains("/outside/corpus"));
    }

    #[test]
    fn resolve_corpus_dir_allows_external_path_with_opt_in() {
        let resolved =
            resolve_corpus_dir_inner(Path::new("/outside/corpus"), Path::new("/repo"), true, None)
                .unwrap();
        assert_eq!(resolved, PathBuf::from("/outside/corpus"));
    }

    #[test]
    fn resolve_corpus_dir_does_not_treat_sibling_prefix_as_inside() {
        let err = resolve_corpus_dir_inner(
            Path::new("/repo-evil/docs"),
            Path::new("/repo"),
            false,
            None,
        )
        .unwrap_err();
        assert!(err.contains("RAG_ALLOW_EXTERNAL_CORPUS"));
    }

    #[test]
    fn resolve_corpus_dir_expands_user_home() {
        let home = std::ffi::OsStr::new("/home/test-user");
        let resolved = resolve_corpus_dir_inner(
            Path::new("~/corpus"),
            Path::new("/home/test-user"),
            false,
            Some(home),
        )
        .unwrap();
        assert_eq!(resolved, PathBuf::from("/home/test-user/corpus"));
    }

    #[test]
    fn config_method_uses_env_backed_repo_root() {
        let config = base();
        let resolved = config.resolve_corpus_dir(Path::new("docs")).unwrap();
        assert_eq!(resolved, repo_root().join("docs"));
    }

    #[test]
    fn validate_does_not_apply_corpus_scope() {
        let mut config = base();
        config.corpus_dir = PathBuf::from("/outside/corpus");
        assert!(config.validate().is_ok());
    }
}
