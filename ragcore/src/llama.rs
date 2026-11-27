//! llama-server の OpenAI 互換 `/v1/embeddings` を呼ぶブロッキングクライアント。
//!
//! - `POST /v1/embeddings` に `{"model": .., "input": [..]}` を送信。
//! - レスポンス `data[].embedding` を入力順に返す。
//! - prefix は送信直前に付与し、入力テキストは変更しない。
//! - 空入力は HTTP を送らず空ベクトル列を返す。
//!
//! エラー文言にはキー値・生レスポンス本文・URL を一切含めない。
//! `unsafe` は使用しない。

use std::time::Duration;

use serde_json::{json, Value};

/// タイムアウト未指定／不正値時の既定（秒）。
const DEFAULT_TIMEOUT_SECS: f64 = 300.0;
/// `Duration::from_secs_f64` のオーバーフローを避ける上限（秒）。
const MAX_TIMEOUT_SECS: f64 = 86_400.0;

fn request_timeout(secs: f64) -> Duration {
    if secs.is_finite() && secs > 0.0 && secs <= MAX_TIMEOUT_SECS {
        Duration::from_secs_f64(secs)
    } else {
        Duration::from_secs_f64(DEFAULT_TIMEOUT_SECS)
    }
}

/// llama-server 埋め込みクライアント。
pub struct LlamaClient {
    base_url: String,
    model: String,
    client: reqwest::blocking::Client,
}

impl LlamaClient {
    /// 共有のブロッキング HTTP クライアントをタイムアウト付きで生成する。
    pub fn new(base_url: &str, model: &str, timeout_secs: f64) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(request_timeout(timeout_secs))
            .build()
            .expect("llama-server HTTP client must be buildable");
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            client,
        }
    }

    /// JSON を POST し、ステータス確認済みの JSON 値を返す。
    fn post_json(&self, url: &str, body: &Value) -> Result<Value, String> {
        let response = self
            .client
            .post(url)
            .json(body)
            .send()
            .map_err(|_| "llama-server への接続に失敗しました".to_string())?;
        if !response.status().is_success() {
            return Err(format!(
                "llama-server がエラーを返しました: status {}",
                response.status().as_u16()
            ));
        }
        response
            .json::<Value>()
            .map_err(|_| "llama-server の応答を解析できませんでした".to_string())
    }

    /// テキスト列を埋め込み、入力と同じ順序でベクトルを返す。
    pub fn embed(
        &self,
        texts: &[String],
        prefix: &str,
        validate_dim: Option<usize>,
    ) -> Result<Vec<Vec<f64>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inputs = prefixed_inputs(texts, prefix);
        let url = format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'));
        let body = embed_body(&self.model, &inputs);
        let data = self.post_json(&url, &body)?;
        parse_embed_response(&data, inputs.len(), validate_dim)
    }
}

/// 送信直前に prefix を付与する（元の `texts` は変更しない）。
fn prefixed_inputs(texts: &[String], prefix: &str) -> Vec<String> {
    texts.iter().map(|text| format!("{prefix}{text}")).collect()
}

/// OpenAI 互換のリクエストボディ。
fn embed_body(model: &str, inputs: &[String]) -> Value {
    json!({ "model": model, "input": inputs })
}

/// 有限値だけを通す（NaN / ±inf を拒否する）。
fn finite_number(number: f64) -> Option<f64> {
    number.is_finite().then_some(number)
}

/// JSON の数値配列を f64 ベクトルへ変換する。
fn vector_from_values(items: &[Value]) -> Option<Vec<f64>> {
    if items.is_empty() {
        return None;
    }
    let mut numbers = Vec::with_capacity(items.len());
    for item in items {
        if item.is_boolean() {
            return None;
        }
        let number = item.as_f64()?;
        numbers.push(finite_number(number)?);
    }
    Some(numbers)
}

/// JSON 値が配列ならベクトルへ変換する。
fn vector_from_value(value: &Value) -> Option<Vec<f64>> {
    vector_from_values(value.as_array()?)
}

/// 次元指定がある場合だけベクトルの長さを検証する。
fn check_dimension(vector: &[f64], validate_dim: Option<usize>) -> Result<(), String> {
    if let Some(dim) = validate_dim {
        if vector.len() != dim {
            return Err("埋め込みベクトルの次元が一致しません".to_string());
        }
    }
    Ok(())
}

/// OpenAI 互換のレスポンスを解釈する（`data[].embedding`）。
fn parse_embed_response(
    data: &Value,
    expected: usize,
    validate_dim: Option<usize>,
) -> Result<Vec<Vec<f64>>, String> {
    let items = data
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| "embeddings レスポンスに data がありません".to_string())?;
    if items.len() != expected {
        return Err("埋め込み件数が入力と一致しません".to_string());
    }
    let mut result = Vec::with_capacity(items.len());
    for item in items {
        let vector = item
            .get("embedding")
            .and_then(vector_from_value)
            .ok_or_else(|| "embedding を取得できませんでした".to_string())?;
        check_dimension(&vector, validate_dim)?;
        result.push(vector);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_has_openai_shape() {
        let inputs = vec!["a".to_string(), "b".to_string()];
        assert_eq!(
            embed_body("model-x", &inputs),
            json!({ "model": "model-x", "input": ["a", "b"] })
        );
    }

    #[test]
    fn prefix_is_applied_at_send_time_only() {
        let texts = vec!["first".to_string(), "second".to_string()];
        assert_eq!(
            prefixed_inputs(&texts, "クエリ: "),
            vec!["クエリ: first", "クエリ: second"]
        );
        assert_eq!(prefixed_inputs(&texts, ""), texts);
    }

    #[test]
    fn vector_validation_rejects_empty_bool_non_number_and_non_finite() {
        assert_eq!(
            vector_from_values(&[json!(1), json!(2.5)]),
            Some(vec![1.0, 2.5])
        );
        assert!(vector_from_values(&[]).is_none());
        assert!(vector_from_values(&[json!(true), json!(1)]).is_none());
        assert!(vector_from_values(&[json!("1")]).is_none());
        assert!(finite_number(f64::NAN).is_none());
        assert!(finite_number(f64::INFINITY).is_none());
    }

    #[test]
    fn parse_reads_openai_embeddings_in_order() {
        let data = json!({
            "data": [
                {"embedding": [1.0, 2.0]},
                {"embedding": [3.0, 4.0]},
            ]
        });
        assert_eq!(
            parse_embed_response(&data, 2, None).unwrap(),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
    }

    #[test]
    fn dimension_mismatch_fails_without_leaking_values() {
        let data = json!({"data": [{"embedding": [1.0, 2.0, 3.0]}]});
        let err = parse_embed_response(&data, 1, Some(2)).unwrap_err();
        assert!(err.contains("次元"));
        assert!(!err.contains("3.0"));
        assert!(!err.contains("SECRET_MODEL"));
    }
}
