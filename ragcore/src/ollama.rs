//! Ollama の埋め込み API を呼ぶブロッキングクライアント。
//!
//! 挙動の基準は `rag_poc/ollama_client.py`。
//! - プライマリ: `POST /api/embed` に `{"model": .., "input": [..]}` を送信。
//! - 失敗／件数不正: 先頭から連続する有効ベクトルを再利用し、残りを
//!   `POST /api/embeddings` に `{"model": .., "prompt": ..}` で 1 件ずつ再送信。
//! - ベクトルは非空・数値のみ・有限値のみ。次元指定時は不一致を即エラーとする。
//!
//! エラー文言にはキー値・生レスポンス本文・URL を一切含めない。
//! `unsafe` は使用しない。

use std::time::Duration;

use serde_json::{json, Value};

/// タイムアウト未指定／不正値時の既定（秒）。
const DEFAULT_TIMEOUT_SECS: f64 = 300.0;
/// `Duration::from_secs_f64` のオーバーフローを避ける上限（秒）。
const MAX_TIMEOUT_SECS: f64 = 86_400.0;
/// フォールバックで `/api/embeddings` へ再送信する件数の上限。
const MAX_FALLBACK_RESENDS: usize = 64;

/// f64 秒を安全な `Duration` へ変換する（負・NaN・無限大・過大値は既定値）。
fn request_timeout(secs: f64) -> Duration {
    if secs.is_finite() && secs > 0.0 && secs <= MAX_TIMEOUT_SECS {
        Duration::from_secs_f64(secs)
    } else {
        Duration::from_secs_f64(DEFAULT_TIMEOUT_SECS)
    }
}

/// Ollama 埋め込みクライアント。
pub struct OllamaClient {
    base_url: String,
    model: String,
    timeout_secs: f64,
    client: reqwest::blocking::Client,
}

impl OllamaClient {
    /// 共有のブロッキング HTTP クライアントをタイムアウト付きで生成する。
    pub fn new(base_url: &str, model: &str, timeout_secs: f64) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(request_timeout(timeout_secs))
            .build()
            .expect("Ollama HTTP client must be buildable");
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            timeout_secs,
            client,
        }
    }

    /// JSON を POST し、ステータス確認済みの JSON 値を返す。
    ///
    /// 外部サービスの詳細（本文・URL・例外文言）はエラーへ含めない。
    fn post_json(&self, url: &str, body: &Value) -> Result<Value, String> {
        let response = self
            .client
            .post(url)
            .timeout(request_timeout(self.timeout_secs))
            .json(body)
            .send()
            .map_err(|_| "Ollama への接続に失敗しました".to_string())?;
        let status = response.status().as_u16();
        if status >= 400 {
            return Err(format!("Ollama へのリクエストに失敗: HTTP {status}"));
        }
        response
            .json::<Value>()
            .map_err(|_| "Ollama 応答の JSON 解析に失敗しました".to_string())
    }

    /// テキスト列を埋め込み、入力と同じ順序でベクトルを返す。
    ///
    /// `prefix` は送信直前に各テキストへ付与し、`texts` 自体は変更しない。
    /// 空入力のときは HTTP リクエストを送らず空ベクトル列を返す。
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
        let root = self.base_url.trim_end_matches('/');
        let primary_url = format!("{root}/api/embed");
        let legacy_url = format!("{root}/api/embeddings");
        let primary_body = embed_body(&self.model, &inputs);

        let (complete, reused) = match self.post_json(&primary_url, &primary_body) {
            Ok(data) => parse_embed_response(&data, inputs.len(), validate_dim)?,
            // プライマリ障害はフォールバックへ委譲する（詳細はエラーに出さない）。
            Err(_) => (None, Vec::new()),
        };
        if let Some(vectors) = complete {
            return Ok(vectors);
        }
        // 余分なベクトルが返った場合など、再利用分で入力件数が満たされれば完了。
        if reused.len() >= inputs.len() {
            return Ok(reused);
        }

        let missing = &inputs[reused.len()..];
        check_resend_limit(missing.len())?;

        let mut result = reused;
        for prompt in missing {
            let body = legacy_body(&self.model, prompt);
            let data = self.post_json(&legacy_url, &body).map_err(|_| {
                "Ollama embed に失敗: /api/embeddings への再送信も失敗しました".to_string()
            })?;
            let vector = data
                .get("embedding")
                .and_then(vector_from_value)
                .ok_or_else(|| {
                    "/api/embeddings から embedding を取得できませんでした".to_string()
                })?;
            check_dimension(&vector, validate_dim)?;
            result.push(vector);
        }
        if result.len() != inputs.len() {
            return Err("埋め込み件数が入力と一致しません".to_string());
        }
        Ok(result)
    }
}

/// 送信直前に prefix を付与する（元の `texts` は変更しない）。
fn prefixed_inputs(texts: &[String], prefix: &str) -> Vec<String> {
    texts.iter().map(|text| format!("{prefix}{text}")).collect()
}

/// プライマリ `/api/embed` のリクエストボディ。
fn embed_body(model: &str, inputs: &[String]) -> Value {
    json!({ "model": model, "input": inputs })
}

/// レガシー `/api/embeddings` のリクエストボディ。
fn legacy_body(model: &str, prompt: &str) -> Value {
    json!({ "model": model, "prompt": prompt })
}

/// 再送信件数の上限を検証する（64 件超はエラー）。
fn check_resend_limit(missing: usize) -> Result<(), String> {
    if missing > MAX_FALLBACK_RESENDS {
        return Err("埋め込みフォールバックの再送信上限を超過しました".to_string());
    }
    Ok(())
}

/// 有限値だけを通す（NaN / ±inf を拒否する）。
fn finite_number(number: f64) -> Option<f64> {
    number.is_finite().then_some(number)
}

/// JSON の数値配列を f64 ベクトルへ変換する。
/// 空配列・bool・非数値・非有限値は `None`。
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
/// 不一致は値をエラー文言へ出さず即座に失敗させる。
fn check_dimension(vector: &[f64], validate_dim: Option<usize>) -> Result<(), String> {
    if let Some(dim) = validate_dim {
        if vector.len() != dim {
            return Err("埋め込みベクトルの次元が一致しません".to_string());
        }
    }
    Ok(())
}

/// `/api/embed` のレスポンスを解釈する。
///
/// 戻り値は「件数も一致した完全なベクトル列」と「先頭から連続する有効ベクトル」。
/// 不正値が出た時点で打ち切り、後ろの有効ベクトルは再利用しない（入力との
/// 位置対応を保つため）。旧実装互換のフラット配列は入力 1 件のときだけ使う。
type ParsedEmbeddings = (Option<Vec<Vec<f64>>>, Vec<Vec<f64>>);

fn parse_embed_response(
    data: &Value,
    expected: usize,
    validate_dim: Option<usize>,
) -> Result<ParsedEmbeddings, String> {
    let Some(raw) = data.get("embeddings").and_then(Value::as_array) else {
        return Ok((None, Vec::new()));
    };
    if raw.is_empty() {
        return Ok((None, Vec::new()));
    }

    if raw.iter().all(Value::is_array) {
        let mut parsed = Vec::with_capacity(raw.len());
        for item in raw {
            let Some(vector) = vector_from_value(item) else {
                break;
            };
            check_dimension(&vector, validate_dim)?;
            parsed.push(vector);
        }
        let reused = parsed.iter().take(expected).cloned().collect();
        let complete = if parsed.len() == expected && raw.len() == expected {
            Some(parsed)
        } else {
            None
        };
        return Ok((complete, reused));
    }

    if expected == 1 {
        if let Some(vector) = vector_from_values(raw) {
            check_dimension(&vector, validate_dim)?;
            return Ok((Some(vec![vector.clone()]), vec![vector]));
        }
    }
    Ok((None, Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_bodies_have_expected_json_shape() {
        let inputs = vec!["a".to_string(), "b".to_string()];
        assert_eq!(
            embed_body("model-x", &inputs),
            json!({ "model": "model-x", "input": ["a", "b"] })
        );
        assert_eq!(
            legacy_body("model-x", "prompt-1"),
            json!({ "model": "model-x", "prompt": "prompt-1" })
        );
    }

    #[test]
    fn prefix_is_applied_at_send_time_only() {
        let texts = vec!["first".to_string(), "second".to_string()];
        assert_eq!(
            prefixed_inputs(&texts, "query: "),
            vec!["query: first", "query: second"]
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
        assert!(vector_from_values(&[json!([1.0])]).is_none());
        assert!(finite_number(f64::NAN).is_none());
        assert!(finite_number(f64::INFINITY).is_none());
        assert_eq!(finite_number(0.0), Some(0.0));
    }

    #[test]
    fn parse_reuses_only_contiguous_valid_vectors() {
        let data = json!({ "embeddings": [[1.0, 2.0], [true], [3.0, 4.0]] });
        let (complete, reused) = parse_embed_response(&data, 3, None).unwrap();
        assert!(complete.is_none());
        assert_eq!(reused, vec![vec![1.0, 2.0]]);
    }

    #[test]
    fn parse_accepts_exact_batch_and_flat_single_vector() {
        let data = json!({ "embeddings": [[1.0, 2.0], [3.0, 4.0]] });
        let (complete, reused) = parse_embed_response(&data, 2, None).unwrap();
        assert_eq!(complete, Some(reused.clone()));
        assert_eq!(reused.len(), 2);

        let flat = json!({ "embeddings": [1.0, 2.0] });
        let (complete, reused) = parse_embed_response(&flat, 1, None).unwrap();
        assert_eq!(complete, Some(vec![vec![1.0, 2.0]]));
        assert_eq!(reused.len(), 1);
    }

    #[test]
    fn dimension_mismatch_fails_immediately_without_leaking_values() {
        let data = json!({ "embeddings": [[1.0, 2.0, 3.0]] });
        let err = parse_embed_response(&data, 1, Some(2)).unwrap_err();
        assert!(err.contains("次元"));
        assert!(!err.contains("3.0"));
        assert!(!err.contains("SECRET_MODEL"));
    }

    #[test]
    fn resend_limit_accepts_64_and_rejects_more() {
        assert!(check_resend_limit(64).is_ok());
        assert!(check_resend_limit(65).is_err());
    }
}
