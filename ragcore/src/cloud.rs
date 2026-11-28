//! opencodeGO の OpenAI 互換 `/embeddings` / `/chat/completions` を呼ぶ blocking クライアント。
//!
//! secrecy に関する約束:
//! - API キー値は保持するが、エラー・ログ・`Debug` 出力には一切出さない。
//!   キーに言及する文言では `api_key_name`（環境変数名）のみを参照する。
//! - エラー文言に URL・レスポンス本文・入力テキストは含めない。
//!   HTTP エラーはステータスコードの数値のみを含める（reason phrase は使わない）。
//! - 429 / 500 / 502 / 503 / 504 のみ最大 3 回リトライ。
//!   Retry-After は 60 秒で cap し、解釈不能値も 60 秒待つ（fail-safe）。
//! - 接続先は https のみ。API キーを平文 HTTP へ送らない。
//! - chat リクエストだけに会話単位の `x-opencode-session` ヘッダを付与する。
//!   embed リクエストには付与しない。
//! - usage は有限数値と既知の billing_source 列挙値だけを保持し、
//!   生 usage はホワイトリストのキーのみを直近 1 リクエスト分だけ保持する。

use crate::config::Config;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

const EMBEDDINGS_PATH: &str = "/embeddings";
const CHAT_PATH: &str = "/chat/completions";
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

/// chat POST ボディ。`Debug` を実装しないのは message 本文を漏らさないため。
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Value],
    temperature: f64,
}

fn build_chat_body<'a>(model: &'a str, messages: &'a [Value], temperature: f64) -> ChatRequest<'a> {
    ChatRequest {
        model,
        messages,
        temperature,
    }
}

/// chat リクエストに付けるヘッダ名（埋め込みには付与しない）。
pub fn chat_header_name() -> &'static str {
    "x-opencode-session"
}

/// `choices[0].message.content` を取り出す。欠損時は本文をエコーしない固定文言。
fn parse_chat_content(payload: &Value) -> Result<String, String> {
    let content = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| "chat response has no message content".to_string())?;
    Ok(content.to_string())
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

/// ベース URL は `trim_end_matches('/')` 後にパスを付与する。
/// base_url 側に `/v1` が含まれるため、ここでは `/v1` を足さない。
fn embeddings_endpoint(base_url: &str) -> String {
    format!("{}{EMBEDDINGS_PATH}", base_url.trim_end_matches('/'))
}

fn chat_endpoint(base_url: &str) -> String {
    format!("{}{CHAT_PATH}", base_url.trim_end_matches('/'))
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

/// URL / `file://` / ホーム以下はじめの絶対パスを `<masked>` へ置換する。
fn mask_urls_and_paths(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let starts = [
            rest.find("http://"),
            rest.find("https://"),
            rest.find("file://"),
            rest.find("/home/"),
        ];
        let Some(start) = starts.into_iter().flatten().min() else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..start]);
        out.push_str("<masked>");
        let after = &rest[start..];
        let end = after
            .char_indices()
            .take_while(|(_, ch)| !ch.is_whitespace() && *ch != ',' && *ch != ';' && *ch != '"')
            .map(|(index, ch)| index + ch.len_utf8())
            .last()
            .unwrap_or(0);
        rest = &after[end..];
    }
    out
}

/// `bearer` の ASCII 大小文字を区別せず一致させ、続く半角スペース/タブを消費する。
/// 改行は別行として扱うため、ここでは行頭から行末までのスライスだけを見る。
fn bearer_prefix_len(line: &str) -> Option<usize> {
    const SCHEME: &[u8; 6] = b"bearer";
    let bytes = line.as_bytes();
    if bytes.len() < SCHEME.len() {
        return None;
    }
    if !bytes[..SCHEME.len()]
        .iter()
        .zip(SCHEME)
        .all(|(actual, expected)| actual.eq_ignore_ascii_case(expected))
    {
        return None;
    }
    let mut end = SCHEME.len();
    while end < bytes.len() && (bytes[end] == b' ' || bytes[end] == b'\t') {
        end += 1;
    }
    // 区切りが 1 個も無ければトークンが存在しない。
    (end > SCHEME.len()).then_some(end)
}

/// `sk-` がトークン境界にあり、直後にトークン文字が続くときだけ成立する。
fn sk_prefix_len(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if start > 0 {
        let previous = bytes[start - 1];
        if previous.is_ascii_alphanumeric() || previous == b'_' || previous == b'-' {
            return None;
        }
    }
    let after = &text[start + 3..];
    let has_token_char = after.bytes().next().is_some_and(|byte| {
        byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' || byte == b'.'
    });
    has_token_char.then_some(3)
}

/// 行内で最初に一致したマスク対象の開始位置と、消費すべき prefix 長を返す。
fn next_mask_hit(text: &str) -> Option<(usize, usize)> {
    let mut line_start = 0usize;
    while line_start <= text.len() {
        let line_end = text[line_start..]
            .find('\n')
            .map(|offset| line_start + offset)
            .unwrap_or(text.len());
        let line = &text[line_start..line_end];

        let mut best: Option<(usize, usize)> = None;
        for (offset, _) in line.char_indices() {
            let absolute = line_start + offset;
            let hit = bearer_prefix_len(&line[offset..]).map(|len| (absolute, len));
            let hit = hit.or_else(|| {
                line[offset..]
                    .strip_prefix("sk-")
                    .and_then(|_| sk_prefix_len(text, absolute).map(|len| (absolute, len)))
            });
            if let Some((start, len)) = hit {
                best = Some((start, len));
                break;
            }
        }
        if best.is_some() {
            return best;
        }
        if line_end == text.len() {
            break;
        }
        line_start = line_end + 1;
    }
    None
}

/// エラー文言に埋め込まれうる Bearer トークン / `sk-` トークンを落とす簡易マスク。
/// 実際の秘密情報はコードに置かない（テストもダミー文字列のみ）。
pub fn mask_error_message(message: &str) -> String {
    let message = mask_urls_and_paths(message);
    let mut out = String::with_capacity(message.len());
    let mut rest = &message[..];
    while !rest.is_empty() {
        let Some((start, prefix_len)) = next_mask_hit(rest) else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..start]);
        out.push_str("<masked>");
        let after = &rest[start + prefix_len..];
        let token_len = after
            .char_indices()
            .take_while(|(_, ch)| !ch.is_whitespace() && *ch != ',' && *ch != ';' && *ch != '"')
            .map(|(index, ch)| index + ch.len_utf8())
            .last()
            .unwrap_or(0);
        rest = &after[token_len..];
    }
    out
}

/// `billing_source` の既知列挙値。これ以外の文字列は破棄する。
const BILLING_SOURCES: [&str; 5] = ["free", "balance", "go", "go-plus", "byok"];

/// usage の蓄積状態。実レスポンスの値ではなく有限数値だけを保持する。
#[derive(Debug, Default)]
struct UsageSnapshot {
    prompt_tokens: f64,
    completion_tokens: f64,
    total_tokens: f64,
    cost: f64,
    cost_usd: f64,
    allowance: f64,
    list_cost: f64,
    billing_source: Option<String>,
    last_raw: Option<Value>,
}

/// chat / embed の使用量。`&self` から更新できるよう内部を `Mutex` にしている
/// （`CloudClient::usage()` が `&Usage` を返す契約を満たすため）。
#[derive(Debug, Default)]
pub struct Usage {
    inner: Mutex<UsageSnapshot>,
}

impl Usage {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(UsageSnapshot::default()),
        }
    }

    fn snapshot(&self) -> std::sync::MutexGuard<'_, UsageSnapshot> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn reset(&mut self) {
        *self.snapshot() = UsageSnapshot::default();
    }

    /// 呼び出し開始時に `&self` から状態をクリアする。
    fn begin_request(&self) {
        *self.snapshot() = UsageSnapshot::default();
    }

    /// 加算済み usage。`billing_source` は既知値があるときだけ現れる。
    pub fn last(&self) -> Value {
        let snapshot = self.snapshot();
        let mut out = Map::new();
        for (key, number) in [
            ("prompt_tokens", snapshot.prompt_tokens),
            ("completion_tokens", snapshot.completion_tokens),
            ("total_tokens", snapshot.total_tokens),
            ("cost", snapshot.cost),
            ("cost_usd", snapshot.cost_usd),
            ("allowance", snapshot.allowance),
            ("list_cost", snapshot.list_cost),
        ] {
            out.insert(key.to_string(), json!(number));
        }
        if let Some(source) = &snapshot.billing_source {
            out.insert("billing_source".to_string(), json!(source));
        }
        Value::Object(out)
    }

    /// 直近 1 リクエスト分のホワイトリスト済み生 usage。
    pub fn last_raw(&self) -> Option<Value> {
        self.snapshot().last_raw.clone()
    }

    fn accumulate(&self, value: &Value) {
        let mut snapshot = self.snapshot();
        if let Some(number) = top_number(value, "prompt_tokens") {
            snapshot.prompt_tokens += number;
        }
        if let Some(number) = top_number(value, "completion_tokens") {
            snapshot.completion_tokens += number;
        }
        if let Some(number) = top_number(value, "total_tokens") {
            snapshot.total_tokens += number;
        }
        if let Some(number) = top_number(value, "cost") {
            snapshot.cost += number;
        }
        if let Some(number) = top_number(value, "cost_usd") {
            snapshot.cost_usd += number;
        }
        if let Some(number) = special_number(value, "allowance") {
            snapshot.allowance += number;
        }
        if let Some(number) = special_number(value, "list_cost") {
            snapshot.list_cost += number;
        }
        if let Some(source) = billing_source(value) {
            snapshot.billing_source = Some(source.to_string());
        }
        snapshot.last_raw = Some(raw_usage(value));
    }
}

/// 公開 API の加算エントリポイント。`&mut Usage` を受け取る。
pub fn accumulate_usage(usage: &mut Usage, value: &Value) {
    usage.accumulate(value);
}

/// レスポンスの `usage` ラッパー（無ければルート）を返す。
fn usage_container(value: &Value) -> &Value {
    match value.get("usage") {
        Some(usage) if usage.is_object() => usage,
        _ => value,
    }
}

fn finite_number(value: Option<&Value>) -> Option<f64> {
    value
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
}

fn top_number(value: &Value, key: &str) -> Option<f64> {
    finite_number(usage_container(value).get(key)).or_else(|| finite_number(value.get(key)))
}

fn opencode_field<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    usage_container(value)
        .get("opencode")
        .and_then(|opencode| opencode.get(key))
        .or_else(|| value.get("opencode").and_then(|opencode| opencode.get(key)))
}

/// `opencode.allowance` / `opencode.list_cost` を優先し、平置きも許容する。
fn special_number(value: &Value, key: &str) -> Option<f64> {
    finite_number(opencode_field(value, key)).or_else(|| top_number(value, key))
}

fn billing_source(value: &Value) -> Option<&str> {
    let candidate = opencode_field(value, "billing_source")
        .or_else(|| usage_container(value).get("billing_source"))
        .or_else(|| value.get("billing_source"))
        .and_then(Value::as_str)?;
    BILLING_SOURCES.contains(&candidate).then_some(candidate)
}

/// 直近リクエストの生 usage をホワイトリストだけに絞る。
fn raw_usage(value: &Value) -> Value {
    let mut raw = Map::new();
    for key in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "cost_usd",
    ] {
        if let Some(number) = top_number(value, key) {
            raw.insert(key.to_string(), json!(number));
        }
    }
    for key in ["allowance", "list_cost"] {
        if let Some(number) = special_number(value, key) {
            raw.insert(key.to_string(), json!(number));
        }
    }
    if let Some(source) = billing_source(value) {
        raw.insert("billing_source".to_string(), json!(source));
    }
    Value::Object(raw)
}

/// opencodeGO の `/embeddings` と `/chat/completions` クライアント。フィールドはすべて private。
pub struct CloudClient {
    http: reqwest::blocking::Client,
    endpoint: String,
    chat_endpoint: String,
    model: String,
    chat_model: String,
    dimensions: Option<usize>,
    chat_enabled: bool,
    chat_timeout: Duration,
    api_key: String,
    usage: Usage,
}

/// API キー値と接続先 URL を出力しないカスタム `Debug`。
impl fmt::Debug for CloudClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CloudClient")
            .field("model", &self.model)
            .field("chat_model", &self.chat_model)
            .field("dimensions", &self.dimensions)
            .field("chat_enabled", &self.chat_enabled)
            .field("usage", &self.usage)
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
        // endpoint は Config が検証済みで返す（URL の再検証は不要）。
        let endpoint = config.cloud_endpoint()?;
        if cloud.embed_model.is_empty() {
            return Err("RAG_CLOUD_EMBED_MODEL must be set explicitly".to_string());
        }

        let http = reqwest::blocking::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(request_timeout(cloud.timeout_secs))
            .build()
            .map_err(|_| "failed to build cloud HTTP client".to_string())?;

        Ok(Self {
            http,
            endpoint: embeddings_endpoint(endpoint.url()),
            chat_endpoint: chat_endpoint(endpoint.url()),
            model: cloud.embed_model.clone(),
            chat_model: cloud.model.clone(),
            dimensions: config.cloud_embed_dimension(),
            chat_enabled: cloud.chat_enabled,
            chat_timeout: request_timeout(config.effective_answer_timeout()),
            api_key: api_key.to_string(),
            usage: Usage::new(),
        })
    }

    /// 入力テキストを埋め込みベクトルへ変換し、入力順で返す。
    ///
    /// `reqwest::Error` の `Display` は URL を含む可能性があるため、そのまま
    /// エラー文言にせず固定文言へ変換する。レスポンス本文も一切保持しない。
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, String> {
        self.usage.begin_request();
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
                let vectors = parse_embeddings(&payload, texts.len(), self.dimensions)?;
                self.usage.accumulate(&payload);
                return Ok(vectors);
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

    /// chat 補完を呼び、先頭 choice の message.content を返す。
    ///
    /// `chat_enabled=false` なら HTTP を一切送らず即座にエラーを返す。
    /// ヘッダは bearer_auth / Content-Type: application/json /
    /// `x-opencode-session: session_id` の 3 つだけ。
    pub fn chat(
        &self,
        messages: &[serde_json::Value],
        temperature: f64,
        session_id: &str,
    ) -> Result<String, String> {
        self.usage.begin_request();
        if !self.chat_enabled {
            return Err("cloud chat is disabled".to_string());
        }
        if session_id.is_empty()
            || session_id.len() > 128
            || !session_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        {
            return Err("invalid chat session id".to_string());
        }

        let body = build_chat_body(&self.chat_model, messages, temperature);
        let mut failures = 0usize;
        loop {
            let response = self
                .http
                .post(&self.chat_endpoint)
                .timeout(self.chat_timeout)
                .bearer_auth(&self.api_key)
                .header(
                    reqwest::header::CONTENT_TYPE,
                    reqwest::header::HeaderValue::from_static("application/json"),
                )
                .header(chat_header_name(), session_id)
                .json(&body)
                .send()
                .map_err(|_| mask_error_message("cloud chat request failed"));

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
                    .map_err(|_| "invalid cloud chat response".to_string())?;
                let content = parse_chat_content(&payload)?;
                self.usage.accumulate(&payload);
                return Ok(content);
            }

            let code = status.as_u16();
            if !RETRY_STATUSES.contains(&code) || failures >= RETRY_TOTAL {
                return Err(mask_error_message(&format!("cloud chat HTTP {code}")));
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

    /// 直近の chat / embed で蓄積した usage。
    pub fn usage(&self) -> &Usage {
        &self.usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CloudConfig, Config};
    use std::path::PathBuf;

    /// ダミーキー文字列のみを使うテスト用設定（HTTP は一切送らない）。
    fn test_config() -> Config {
        Config {
            corpus_dir: PathBuf::from("docs"),
            collection: "c".to_string(),
            llama_server_url: "http://127.0.0.1:8090".to_string(),
            llama_chat_url: "http://127.0.0.1:8090".to_string(),
            qdrant_url: "http://127.0.0.1:6333".to_string(),
            allow_remote_endpoints: true,
            allow_external_corpus: false,
            allowed_endpoint_hosts: vec!["example.invalid".to_string()],
            embed_model: "qwen2.5:3b".to_string(),
            chat_model: "qwen2.5:3b".to_string(),
            vector_size: 3,
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
    fn chat_header_name_is_x_opencode_session() {
        assert_eq!(chat_header_name(), "x-opencode-session");
    }

    #[test]
    fn chat_endpoint_trims_trailing_slashes_without_adding_v1() {
        assert_eq!(
            chat_endpoint("https://example.invalid/v1///"),
            "https://example.invalid/v1/chat/completions"
        );
        assert_eq!(
            chat_endpoint("https://example.invalid/v1"),
            "https://example.invalid/v1/chat/completions"
        );
    }

    #[test]
    fn chat_request_body_has_model_messages_and_temperature() {
        let messages = vec![json!({"role": "user", "content": "DUMMY_INPUT"})];
        let body = serde_json::to_value(build_chat_body("chat-model", &messages, 0.25))
            .expect("serializable chat request");
        let object = body.as_object().expect("chat request is an object");
        assert_eq!(object.len(), 3);
        assert_eq!(
            object.get("model").and_then(Value::as_str),
            Some("chat-model")
        );
        assert_eq!(
            object["messages"][0]["content"].as_str(),
            Some("DUMMY_INPUT")
        );
        assert_eq!(
            object.get("temperature").and_then(Value::as_f64),
            Some(0.25)
        );
        assert!(!object.contains_key(chat_header_name()));
    }

    #[test]
    fn chat_content_parse_uses_first_choice_and_fixed_error() {
        let payload = json!({
            "choices": [
                {"message": {"content": "DUMMY_ANSWER"}},
                {"message": {"content": "SECOND_ANSWER"}}
            ]
        });
        assert_eq!(
            parse_chat_content(&payload).expect("first content"),
            "DUMMY_ANSWER"
        );

        let missing = json!({"choices": [{"message": {}}], "note": "SECRET_BODY"});
        let error = parse_chat_content(&missing).unwrap_err();
        assert_eq!(error, "chat response has no message content");
        assert!(!error.contains("SECRET_BODY"));
    }

    #[test]
    fn disabled_chat_fails_immediately_without_secrets() {
        let config = test_config();
        let client = CloudClient::from_config(&config).expect("enabled cloud config");
        let messages = vec![json!({"role": "user", "content": "SECRET_INPUT"})];
        let error = client
            .chat(&messages, 0.0, "dummy-session")
            .expect_err("chat disabled");
        assert_eq!(error, "cloud chat is disabled");
        assert!(!error.contains("SECRET_INPUT"));
        assert!(!error.contains("example.invalid"));
        assert!(!error.contains("dummy-test-key"));
        assert_eq!(client.usage().last()["prompt_tokens"].as_f64(), Some(0.0));
        assert!(client.usage().last_raw().is_none());
    }

    #[test]
    fn usage_accumulates_finite_numbers_and_rejects_non_finite() {
        let mut usage = Usage::new();
        let first = json!({
            "usage": {
                "prompt_tokens": 2.0,
                "completion_tokens": 1.0,
                "total_tokens": 3.0,
                "cost": 0.0,
                "cost_usd": f64::INFINITY,
                "opencode": {"allowance": 0.25, "list_cost": 1.5},
                "secret_field": "SECRET_VALUE"
            }
        });
        accumulate_usage(&mut usage, &first);

        let second = json!({
            "usage": {
                "prompt_tokens": 4.0,
                "cost_usd": f64::NAN,
                "opencode": {"allowance": f64::NEG_INFINITY}
            }
        });
        accumulate_usage(&mut usage, &second);

        let last = usage.last();
        assert_eq!(last["prompt_tokens"].as_f64(), Some(6.0));
        assert_eq!(last["completion_tokens"].as_f64(), Some(1.0));
        assert_eq!(last["total_tokens"].as_f64(), Some(3.0));
        assert_eq!(last["cost"].as_f64(), Some(0.0));
        assert_eq!(last["cost_usd"].as_f64(), Some(0.0));
        assert_eq!(last["allowance"].as_f64(), Some(0.25));
        assert_eq!(last["list_cost"].as_f64(), Some(1.5));

        let raw = usage.last_raw().expect("last raw usage");
        let raw_text = raw.to_string();
        assert_eq!(raw["prompt_tokens"].as_f64(), Some(4.0));
        assert!(!raw_text.contains("secret_field"));
        assert!(!raw_text.contains("SECRET_VALUE"));
        assert!(!raw_text.contains("cost_usd"));
        let keys = raw.as_object().expect("raw is object");
        assert!(keys.keys().all(|key| [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost",
            "cost_usd",
            "allowance",
            "list_cost",
            "billing_source"
        ]
        .contains(&key.as_str())));
    }

    #[test]
    fn billing_source_unknown_values_are_discarded() {
        let mut usage = Usage::new();
        accumulate_usage(
            &mut usage,
            &json!({"usage": {"opencode": {"billing_source": "unknown-source"}}}),
        );
        assert!(usage.last().get("billing_source").is_none());
        assert!(!usage
            .last_raw()
            .expect("raw")
            .to_string()
            .contains("unknown-source"));

        accumulate_usage(
            &mut usage,
            &json!({"usage": {"opencode": {"billing_source": "go"}}}),
        );
        assert_eq!(usage.last()["billing_source"].as_str(), Some("go"));
        assert_eq!(
            usage.last_raw().expect("raw")["billing_source"].as_str(),
            Some("go")
        );

        // 未知値は既知の最後の値を上書きしない。
        accumulate_usage(
            &mut usage,
            &json!({"opencode": {"billing_source": "mystery"}}),
        );
        assert_eq!(usage.last()["billing_source"].as_str(), Some("go"));
        assert!(!usage
            .last_raw()
            .expect("raw")
            .to_string()
            .contains("mystery"));
    }

    #[test]
    fn usage_reset_clears_accumulated_state() {
        let mut usage = Usage::new();
        accumulate_usage(
            &mut usage,
            &json!({"usage": {"prompt_tokens": 7.0, "opencode": {"billing_source": "byok"}}}),
        );
        assert_eq!(usage.last()["prompt_tokens"].as_f64(), Some(7.0));
        usage.reset();
        assert_eq!(usage.last()["prompt_tokens"].as_f64(), Some(0.0));
        assert!(usage.last().get("billing_source").is_none());
        assert!(usage.last_raw().is_none());
    }

    #[test]
    fn mask_error_message_redacts_bearer_and_sk_tokens() {
        assert_eq!(
            mask_error_message("request failed: Bearer DUMMY_TOKEN"),
            "request failed: <masked>"
        );
        assert_eq!(
            mask_error_message("prefix sk-DUMMY and sk-OTHER"),
            "prefix <masked> and <masked>"
        );
        assert_eq!(mask_error_message("plain error"), "plain error");
    }

    #[test]
    fn mask_error_message_handles_bearer_case_tab_and_sk_word_boundary() {
        // 小文字 / 大文字 / 混在の bearer と、スペース・タブの複数区切り。
        assert_eq!(mask_error_message("low bearer DUMMY_TOKEN"), "low <masked>");
        assert_eq!(
            mask_error_message("loud BEARER DUMMY_TOKEN"),
            "loud <masked>"
        );
        assert_eq!(
            mask_error_message("mixed Bearer\t\tDUMMY_TOKEN"),
            "mixed <masked>"
        );
        assert_eq!(
            mask_error_message("tab bearer\tDUMMY_TOKEN"),
            "tab <masked>"
        );

        // 一般語の "task-force" / "ask-force" はマスクしない。
        assert_eq!(
            mask_error_message("task-force and ask-force are ordinary words"),
            "task-force and ask-force are ordinary words"
        );

        // 改行は跨がない: 区切りが無い行の Bearer はマスクしない。
        assert_eq!(
            mask_error_message("header Bearer\nDUMMY_TOKEN"),
            "header Bearer\nDUMMY_TOKEN"
        );
        assert_eq!(
            mask_error_message("first bearer X\nsecond sk-DUMMY"),
            "first <masked>\nsecond <masked>"
        );

        // 既存の sk- トークンは維持。
        assert_eq!(
            mask_error_message("prefix sk-DUMMY and sk-OTHER"),
            "prefix <masked> and <masked>"
        );
        assert_eq!(mask_error_message("plain error"), "plain error");
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
    fn mask_error_message_handles_multibyte_input() {
        let masked = mask_error_message("日本語のエラー Bearer DUMMY_TOKEN と sk-DUMMY を含みます");
        assert!(masked.contains("<masked>"));
        assert!(!masked.contains("DUMMY_TOKEN"));
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let mut config = test_config();
        config.cloud.chat_enabled = true;
        let client = CloudClient::from_config(&config).expect("enabled config");
        let rendered = format!("{client:?}");
        assert!(rendered.contains("CloudClient"));
        assert!(rendered.contains("chat_enabled"));
        assert!(!rendered.contains("dummy-test-key"));
        assert!(!rendered.contains("example.invalid"));
    }
}
