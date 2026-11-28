//! opencodeGO の OpenAI 互換 `/embeddings` を呼ぶ blocking クライアント。
//!
//! secrecy に関する約束:
//! - API キー値は保持するが、エラー・ログ・`Debug` 出力には一切出さない。
//!   キーに言及する文言では `api_key_name`（環境変数名）のみを参照する。
//! - エラー文言に URL・レスポンス本文・入力テキストは含めない。
//!   HTTP エラーはステータスコードの数値のみを含める（reason phrase は使わない）。
//! - 429 / 500 / 502 / 503 / 504 のみ最大 3 回リトライ。
//!   Retry-After は 60 秒で cap し、解釈不能値も 60 秒待つ（fail-safe）。
//! - 接続先は https のみ。API キーを平文 HTTP へ送らない。

use crate::config::Config;
use serde::Serialize;
use serde_json::Value;
use std::fmt;
use std::time::Duration;

const EMBEDDINGS_PATH: &str = "/embeddings";
const DEFAULT_TIMEOUT_SECS: f64 = 60.0;
const MIN_TIMEOUT_SECS: f64 = 1.0;
const MAX_TIMEOUT_SECS: f64 = 86_400.0;

/// POST ボディ。`dimensions` は指定されたときだけ現れる。
///
/// `Debug` を実装しないのは、入力テキストがデバッグ出力へ漏れるのを防ぐため。
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

fn build_request_body<'a>(
    model: &'a str,
    texts: &'a [String],
    dimensions: Option<usize>,
) -> EmbeddingRequest<'a> {
    EmbeddingRequest {
        model,
        input: texts,
        dimensions,
    }
}

/// 数値列を検証して `f64` ベクトルに確定させる（空・次元不一致・非有限数を拒否）。
fn validate_finite_vector(
    values: Vec<f64>,
    expected_dim: Option<usize>,
) -> Result<Vec<f64>, String> {
    if values.is_empty() {
        return Err("embedding vector is empty".to_string());
    }
    if expected_dim.is_some_and(|dim| values.len() != dim) {
        return Err("embedding dimension mismatch".to_string());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("embedding vector contains a non-finite value".to_string());
    }
    Ok(values)
}

/// `data[].embedding` を配列順（＝入力順）で取り出す純粋ロジック。
///
/// エラー文言には `payload` の内容を一切含めない。
fn parse_embeddings(
    payload: &Value,
    expected_count: usize,
    expected_dim: Option<usize>,
) -> Result<Vec<Vec<f64>>, String> {
    let data = payload
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| "embeddings response has no data array".to_string())?;
    if data.is_empty() {
        return Err("embeddings response has an empty data array".to_string());
    }
    if data.len() != expected_count {
        return Err("embeddings response count mismatch".to_string());
    }

    let mut vectors = Vec::with_capacity(data.len());
    for (index, item) in data.iter().enumerate() {
        let values = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("embedding at index {index} is missing or not an array"))?;
        let mut numbers = Vec::with_capacity(values.len());
        for value in values {
            let number = value
                .as_f64()
                .ok_or_else(|| format!("embedding at index {index} has a non-numeric value"))?;
            numbers.push(number);
        }
        let vector = validate_finite_vector(numbers, expected_dim)
            .map_err(|error| format!("embedding at index {index}: {error}"))?;
        vectors.push(vector);
    }
    Ok(vectors)
}

/// 設定値を安全な `Duration` に丸める（NaN / 負 / 無限大でも panic しない）。
fn request_timeout(secs: f64) -> Duration {
    let secs = if secs.is_finite() && secs >= MIN_TIMEOUT_SECS {
        secs.min(MAX_TIMEOUT_SECS)
    } else {
        DEFAULT_TIMEOUT_SECS
    };
    Duration::from_secs_f64(secs)
}

/// ベース URL は `trim_end_matches('/')` 後に `/embeddings` を付与する。
/// base_url 側に `/v1` が含まれるため、ここでは `/v1` を足さない。
fn embeddings_endpoint(base_url: &str) -> String {
    format!("{}{EMBEDDINGS_PATH}", base_url.trim_end_matches('/'))
}

/// 再試行回数（総試行回数は 1 + この値）。
const RETRY_TOTAL: usize = 3;
/// backoff の基数（秒）。連続失敗 n 回目は `0.5 * 2^(n-1)` 秒。
const RETRY_BACKOFF_SECS: f64 = 0.5;
/// 再試行対象の HTTP ステータス。
const RETRY_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];
/// Retry-After の上限（秒）。無限待ちを避ける。
const MAX_RETRY_AFTER_SECS: u64 = 60;

fn retry_backoff(failure_count: usize) -> Duration {
    let exponent = failure_count.saturating_sub(1).min(16) as u32;
    let secs = RETRY_BACKOFF_SECS * 2f64.powi(exponent as i32);
    Duration::from_secs_f64(secs.min(MAX_RETRY_AFTER_SECS as f64))
}

/// Retry-After の秒数だけを解釈する（HTTP-date は無視して backoff を使う）。
fn retry_after(value: &str) -> Option<Duration> {
    value
        .trim()
        .parse::<u64>()
        .ok()
        .map(|secs| Duration::from_secs(secs.min(MAX_RETRY_AFTER_SECS)))
        .or(Some(Duration::from_secs(MAX_RETRY_AFTER_SECS)))
}

/// opencodeGO の `/embeddings` クライアント。フィールドはすべて private。
pub struct CloudClient {
    http: reqwest::blocking::Client,
    endpoint: String,
    model: String,
    dimensions: Option<usize>,
    api_key: String,
}

/// API キー値を出力しないカスタム `Debug`。
impl fmt::Debug for CloudClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CloudClient")
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl CloudClient {
    /// 無効時は外部クライアントを一切生成せず即座にエラーを返す。
    pub fn from_config(config: &Config) -> Result<Self, String> {
        let cloud = &config.cloud;
        if !cloud.enabled {
            return Err("cloud is disabled".to_string());
        }

        // キー値はエラー文言に出さず、参照名（環境変数名）だけを使う。
        let api_key = cloud.api_key.as_deref().ok_or_else(|| {
            format!(
                "{} is required for cloud embeddings",
                cloud.api_key_name.unwrap_or("cloud API key")
            )
        })?;
        let parsed = url::Url::parse(&cloud.base_url)
            .map_err(|_| "RAG_CLOUD_BASE_URL is invalid".to_string())?;
        if parsed.scheme() != "https" {
            return Err("RAG_CLOUD_BASE_URL must use https".to_string());
        }
        if cloud.embed_model.is_empty() {
            return Err("RAG_CLOUD_EMBED_MODEL must be set explicitly".to_string());
        }

        let http = reqwest::blocking::Client::builder()
            .timeout(request_timeout(cloud.timeout_secs))
            .build()
            .map_err(|_| "failed to build cloud HTTP client".to_string())?;

        Ok(Self {
            http,
            endpoint: embeddings_endpoint(&cloud.base_url),
            model: cloud.embed_model.clone(),
            dimensions: config.cloud_embed_dimension(),
            api_key: api_key.to_string(),
        })
    }

    /// 入力テキストを埋め込みベクトルへ変換し、入力順で返す。
    ///
    /// `reqwest::Error` の `Display` は URL を含む可能性があるため、そのまま
    /// エラー文言にせず固定文言へ変換する。レスポンス本文も一切保持しない。
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let body = build_request_body(&self.model, texts, self.dimensions);
        let mut failures = 0usize;
        loop {
            let response = self
                .http
                .post(&self.endpoint)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .map_err(|_| "cloud embeddings request failed".to_string());

            let response = match response {
                Ok(response) => response,
                Err(err) => {
                    if failures >= RETRY_TOTAL {
                        return Err(err);
                    }
                    failures += 1;
                    std::thread::sleep(retry_backoff(failures));
                    continue;
                }
            };

            let status = response.status();
            if status.is_success() {
                let payload: Value = response
                    .json()
                    .map_err(|_| "invalid cloud embeddings response".to_string())?;
                return parse_embeddings(&payload, texts.len(), self.dimensions);
            }

            let code = status.as_u16();
            if !RETRY_STATUSES.contains(&code) || failures >= RETRY_TOTAL {
                return Err(format!("cloud embeddings HTTP {code}"));
            }
            let delay = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(retry_after)
                .unwrap_or_else(|| retry_backoff(failures + 1));
            failures += 1;
            std::thread::sleep(delay);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CloudConfig, Config};
    use std::path::PathBuf;

    /// ダミーキー文字列のみを使うテスト用設定。
    fn test_config() -> Config {
        Config {
            corpus_dir: PathBuf::from("docs"),
            collection: "c".to_string(),
            llama_server_url: "http://127.0.0.1:8090".to_string(),
            qdrant_url: "http://127.0.0.1:6333".to_string(),
            allow_remote_endpoints: false,
            allowed_endpoint_hosts: Vec::new(),
            embed_model: "qwen2.5:3b".to_string(),
            chat_model: "qwen2.5:3b".to_string(),
            vector_size: 3,
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
                enabled: true,
                chat_enabled: false,
                base_url: "https://example.invalid/v1/".to_string(),
                embed_model: "embed-model".to_string(),
                embed_dimension: Some(3),
                model: "chat-model".to_string(),
                timeout_secs: 60.0,
                api_key: Some("dummy-test-key".to_string()),
                api_key_name: Some("DUMMY_KEY_ENV"),
            },
        }
    }

    #[test]
    fn request_body_has_model_input_and_optional_dimensions() {
        let texts = vec!["first".to_string(), "second".to_string()];

        let with_dim = serde_json::to_value(build_request_body("embed-model", &texts, Some(3)))
            .expect("serializable request");
        let object = with_dim.as_object().expect("request is an object");
        assert_eq!(object.len(), 3);
        assert_eq!(
            object.get("model").and_then(Value::as_str),
            Some("embed-model")
        );
        let input = object
            .get("input")
            .and_then(Value::as_array)
            .expect("input array");
        assert_eq!(
            input.iter().map(|value| value.as_str()).collect::<Vec<_>>(),
            vec![Some("first"), Some("second")]
        );
        assert_eq!(object.get("dimensions").and_then(Value::as_u64), Some(3));
        assert!(!object.contains_key("x-opencode-session"));

        let without_dim = serde_json::to_value(build_request_body("embed-model", &texts, None))
            .expect("serializable request");
        let object = without_dim.as_object().expect("request is an object");
        assert_eq!(object.len(), 2);
        assert!(!object.contains_key("dimensions"));
        assert!(!object.contains_key("x-opencode-session"));
    }

    #[test]
    fn embeddings_endpoint_trims_trailing_slashes_without_adding_v1() {
        assert_eq!(
            embeddings_endpoint("https://example.invalid/v1///"),
            "https://example.invalid/v1/embeddings"
        );
        assert_eq!(
            embeddings_endpoint("https://example.invalid/v1"),
            "https://example.invalid/v1/embeddings"
        );
    }

    #[test]
    fn response_embeddings_follow_input_order() {
        let payload = serde_json::json!({
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [1.1, 1.2, 1.3]},
                {"index": 2, "embedding": [2.1, 2.2, 2.3]}
            ]
        });
        let vectors = parse_embeddings(&payload, 3, Some(3)).expect("valid response");
        assert_eq!(
            vectors,
            vec![
                vec![0.1, 0.2, 0.3],
                vec![1.1, 1.2, 1.3],
                vec![2.1, 2.2, 2.3],
            ]
        );
    }

    #[test]
    fn response_rejects_invalid_vectors() {
        let wrong_count = serde_json::json!({"data": [{"embedding": [1.0, 2.0, 3.0]}]});
        assert!(parse_embeddings(&wrong_count, 2, None).is_err());

        let empty_data = serde_json::json!({"data": []});
        assert!(parse_embeddings(&empty_data, 0, None).is_err());

        let empty_vector = serde_json::json!({"data": [{"embedding": []}]});
        assert!(parse_embeddings(&empty_vector, 1, None).is_err());

        let bool_value = serde_json::json!({"data": [{"embedding": [true, 2.0, 3.0]}]});
        assert!(parse_embeddings(&bool_value, 1, None).is_err());

        let string_value = serde_json::json!({"data": [{"embedding": ["0.1", 2.0, 3.0]}]});
        assert!(parse_embeddings(&string_value, 1, None).is_err());

        let wrong_dim = serde_json::json!({"data": [{"embedding": [1.0, 2.0, 3.0]}]});
        assert!(parse_embeddings(&wrong_dim, 1, Some(2)).is_err());

        assert!(validate_finite_vector(vec![f64::NAN, 1.0], None).is_err());
        assert!(validate_finite_vector(vec![f64::INFINITY], None).is_err());
        assert!(validate_finite_vector(vec![f64::NEG_INFINITY], None).is_err());
        assert!(validate_finite_vector(vec![1.0, 2.0], None).is_ok());
    }

    #[test]
    fn disabled_config_returns_error_without_creating_client() {
        let mut config = test_config();
        config.cloud.enabled = false;
        // キーが設定されていても disabled 判定が優先される。
        config.cloud.api_key = Some("dummy-test-key".to_string());
        assert_eq!(
            CloudClient::from_config(&config).err().as_deref(),
            Some("cloud is disabled")
        );
    }

    #[test]
    fn missing_key_error_mentions_only_the_key_name() {
        let mut config = test_config();
        config.cloud.api_key = None;
        config.cloud.api_key_name = Some("DUMMY_KEY_ENV");
        let error = CloudClient::from_config(&config).unwrap_err();
        assert!(error.contains("DUMMY_KEY_ENV"));
        assert!(!error.contains("dummy-test-key"));
    }

    #[test]
    fn parse_errors_do_not_echo_payload_content() {
        let payload = serde_json::json!({
            "data": [{"embedding": ["SECRET_RESPONSE_BODY"]}],
            "note": "SECRET_RESPONSE_BODY"
        });
        let error = parse_embeddings(&payload, 1, None).unwrap_err();
        assert!(!error.contains("SECRET_RESPONSE_BODY"));
        assert!(!error.contains("example.invalid"));
    }

    #[test]
    fn retry_policy_matches_python_defaults() {
        assert_eq!(RETRY_TOTAL, 3);
        assert_eq!(RETRY_STATUSES, [429, 500, 502, 503, 504]);
        assert_eq!(retry_backoff(1).as_secs_f64(), 0.5);
        assert_eq!(retry_backoff(2).as_secs_f64(), 1.0);
        assert_eq!(retry_backoff(3).as_secs_f64(), 2.0);
        assert_eq!(retry_after("7"), Some(Duration::from_secs(7)));
        assert_eq!(
            retry_after("Wed, 21 Oct 2015 07:28:00 GMT"),
            Some(Duration::from_secs(MAX_RETRY_AFTER_SECS))
        );
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let config = test_config();
        let client = CloudClient::from_config(&config).expect("enabled config");
        let rendered = format!("{client:?}");
        assert!(rendered.contains("CloudClient"));
        assert!(!rendered.contains("dummy-test-key"));
        assert!(!rendered.contains("example.invalid"));
    }
}
