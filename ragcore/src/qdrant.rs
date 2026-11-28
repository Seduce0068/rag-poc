//! Qdrant のコレクション／points API を呼ぶブロッキングクライアント。
//!
//! 挙動の基準は `rag_poc/ingest.py` と `rag_poc/query.py`。
//! - コレクション名は percent-encoding でパスへ埋め込む（`quote(name, safe="")` 相当）。
//! - エラーは `Result<_, String>` とし、ステータス >=400 は短文のマスク済み
//!   メッセージのみ（レスポンス本文・URL・キー値は含めない）。
//!
//! - scroll は最大 100 ページで打ち切り、未取得分が残る場合は `ScrollResult`
//!   の `truncated` を true にして部分結果を成功として返す（旧 Python 契約）。
//!
//! `unsafe` は使用しない。

use crate::config::ValidatedEndpoint;
use std::time::Duration;

use reqwest::blocking::Response;
use reqwest::Method;
use serde_json::{json, Value};

/// タイムアウト未指定／不正値時の既定（秒）。
const DEFAULT_TIMEOUT_SECS: f64 = 300.0;
/// `Duration::from_secs_f64` のオーバーフローを避ける上限（秒）。
const MAX_TIMEOUT_SECS: f64 = 86_400.0;
/// scroll の 1 ページ件数。
const SCROLL_PAGE_SIZE: usize = 256;
/// scroll の最大ページ数。
const MAX_SCROLL_PAGES: usize = 100;

/// f64 秒を安全な `Duration` へ変換する（負・NaN・無限大・過大値は既定値）。
fn request_timeout(secs: f64) -> Duration {
    if secs.is_finite() && secs > 0.0 && secs <= MAX_TIMEOUT_SECS {
        Duration::from_secs_f64(secs)
    } else {
        Duration::from_secs_f64(DEFAULT_TIMEOUT_SECS)
    }
}

/// scroll の結果。上限到達で未取得分が残る場合は `truncated = true`。
#[derive(Debug, Clone)]
pub struct ScrollResult {
    pub points: Vec<serde_json::Value>,
    pub truncated: bool,
}

/// Qdrant コレクションを操作するクライアント。
pub struct QdrantClient {
    base_url: String,
    collection: String,
    timeout_secs: f64,
    client: reqwest::blocking::Client,
}

impl QdrantClient {
    /// 共有のブロッキング HTTP クライアントをタイムアウト付きで生成する。
    pub fn new(endpoint: &ValidatedEndpoint, collection: &str, timeout_secs: f64) -> Self {
        let client = reqwest::blocking::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(request_timeout(timeout_secs))
            .build()
            .expect("Qdrant HTTP client must be buildable");
        Self {
            base_url: endpoint.url().to_string(),
            collection: collection.to_string(),
            timeout_secs,
            client,
        }
    }

    /// `/collections/{name}` の URL を組み立てる。
    fn collection_url(&self) -> String {
        format!(
            "{}/collections/{}",
            self.base_url.trim_end_matches('/'),
            encode_segment(&self.collection)
        )
    }

    /// リクエストを送り、成功時に JSON ボディを返す（空ボディは `None`）。
    fn send(
        &self,
        method: Method,
        url: &str,
        body: Option<&Value>,
        context: &str,
    ) -> Result<Option<Value>, String> {
        let mut builder = self
            .client
            .request(method, url)
            .timeout(request_timeout(self.timeout_secs));
        if let Some(body) = body {
            builder = builder.json(body);
        }
        let response = builder
            .send()
            .map_err(|_| "Qdrant への接続に失敗しました".to_string())?;
        let status = response.status().as_u16();
        if status >= 400 {
            return Err(status_error(context, status));
        }
        read_json(response)
    }

    /// コレクションが存在するか（200 系 → true、404 → false）。
    pub fn collection_exists(&self) -> Result<bool, String> {
        let response = self
            .client
            .get(self.collection_url())
            .timeout(request_timeout(self.timeout_secs))
            .send()
            .map_err(|_| "Qdrant への接続に失敗しました".to_string())?;
        exists_from_status(response.status().as_u16())
    }

    /// 指定次元の Cosine コレクションを作る。
    pub fn create_collection(&self, size: usize) -> Result<(), String> {
        let body = create_collection_body(size);
        self.send(
            Method::PUT,
            &self.collection_url(),
            Some(&body),
            "コレクション作成",
        )?;
        Ok(())
    }

    /// コレクションを削除する（404 は既に存在しないとして成功扱い）。
    pub fn delete_collection(&self) -> Result<(), String> {
        let response = self
            .client
            .delete(self.collection_url())
            .timeout(request_timeout(self.timeout_secs))
            .send()
            .map_err(|_| "Qdrant への接続に失敗しました".to_string())?;
        delete_from_status(response.status().as_u16())
    }

    /// points を投入し、実際に送信した点数を返す。
    pub fn upsert(&self, points: &[Value]) -> Result<usize, String> {
        let url = format!("{}/points?wait=true", self.collection_url());
        let body = upsert_body(points);
        self.send(Method::PUT, &url, Some(&body), "points upsert")?;
        Ok(points.len())
    }

    /// scroll API をページングして全 point を返す（最大 100 ページ）。
    ///
    /// 不正なページ形状はエラーとする。上限ページ取得後も offset が残る場合は
    /// 部分結果を成功として返し、`truncated` を true にする（旧 Python 契約）。
    pub fn scroll_all(&self) -> Result<ScrollResult, String> {
        let url = format!("{}/points/scroll", self.collection_url());
        let mut offset: Option<Value> = None;
        let mut points = Vec::new();
        // ループ最後の offset 状態だけが truncated 判定に使う値。
        let mut final_offset: Option<Value> = None;
        for _ in 0..MAX_SCROLL_PAGES {
            let body = scroll_body(offset.as_ref());
            let data = self
                .send(Method::POST, &url, Some(&body), "points scroll")?
                .ok_or_else(points_parse_error)?;
            let result = data.get("result").unwrap_or(&data);
            let batch = scroll_points(result)?;
            if batch.is_empty() {
                // 旧 Python 契約: 空ページで即終端（offset が残っても続行しない）。
                final_offset = None;
                break;
            }
            points.extend(batch);
            final_offset = next_offset(result);
            if final_offset.is_none() {
                break;
            }
            offset = final_offset.clone();
        }
        Ok(ScrollResult {
            points,
            truncated: scroll_truncated(&final_offset),
        })
    }

    /// 密ベクトルで points/query を呼び、point の配列を返す。
    pub fn query_dense(&self, vector: &[f64], limit: usize) -> Result<Vec<Value>, String> {
        let url = format!("{}/points/query", self.collection_url());
        let body = query_body(vector, limit);
        let data = self
            .send(Method::POST, &url, Some(&body), "points query")?
            .ok_or_else(points_parse_error)?;
        points_from_response(&data)
    }

    /// exact count を返す。
    pub fn count_exact(&self) -> Result<usize, String> {
        let url = format!("{}/points/count", self.collection_url());
        let body = count_body();
        let data = self
            .send(Method::POST, &url, Some(&body), "points count")?
            .unwrap_or(Value::Null);
        count_from_response(&data)
    }
}

/// レスポンス本文を読み、空なら `None` を返す。
fn read_json(response: Response) -> Result<Option<Value>, String> {
    let text = response
        .text()
        .map_err(|_| "Qdrant 応答の読み取りに失敗しました".to_string())?;
    if text.trim().is_empty() {
        return Ok(None);
    }
    serde_json::from_str::<Value>(&text)
        .map(Some)
        .map_err(|_| "Qdrant 応答の JSON 解析に失敗しました".to_string())
}

/// パス segment の percent-encoding（unreserved 以外を %XX、英大文字 hex）。
/// `url::form_urlencoded` と違い、空白は `+` ではなく `%20` になる。
fn encode_segment(value: &str) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char)
            }
            other => {
                let _ = write!(out, "%{other:02X}");
            }
        }
    }
    out
}

/// ステータスだけを含む短文エラー（本文・URL は渡さない）。
fn status_error(context: &str, status: u16) -> String {
    format!("Qdrant {context} に失敗: HTTP {status}")
}

/// GET /collections/{name} のステータス解釈。
fn exists_from_status(status: u16) -> Result<bool, String> {
    match status {
        200..=299 => Ok(true),
        404 => Ok(false),
        other => Err(status_error("コレクション確認", other)),
    }
}

/// DELETE /collections/{name} のステータス解釈（404 は成功）。
fn delete_from_status(status: u16) -> Result<(), String> {
    match status {
        404 => Ok(()),
        400..=u16::MAX => Err(status_error("コレクション削除", status)),
        _ => Ok(()),
    }
}

/// コレクション作成ボディ。
fn create_collection_body(size: usize) -> Value {
    json!({ "vectors": { "size": size, "distance": "Cosine" } })
}

/// points upsert ボディ。
fn upsert_body(points: &[Value]) -> Value {
    json!({ "points": points })
}

/// scroll ボディ（offset がある場合だけキーを追加）。
fn scroll_body(offset: Option<&Value>) -> Value {
    let mut body = json!({
        "limit": SCROLL_PAGE_SIZE,
        "with_payload": true,
        "with_vector": false,
    });
    if let Some(offset) = offset {
        body["offset"] = (*offset).clone();
    }
    body
}

/// dense query ボディ。
fn query_body(vector: &[f64], limit: usize) -> Value {
    json!({ "query": vector, "limit": limit, "with_payload": true })
}

/// exact count ボディ。
fn count_body() -> Value {
    json!({ "exact": true })
}

/// points 抽出失敗時の固定文言エラー（URL・本文は含まない）。
fn points_parse_error() -> String {
    "Qdrant points 応答を解析できませんでした".to_string()
}

/// Qdrant レスポンスから point 配列を取り出す。
///
/// 受理するのは `result` 配列 / `result.points` 配列 / トップレベル `points`
/// 配列 / bare 配列（旧 Python 契約）。正規の空配列は `Ok(vec![])`、
/// 欠落・型不正は固定文言の `Err`。
fn points_from_response(data: &Value) -> Result<Vec<Value>, String> {
    let result = data.get("result").unwrap_or(data);
    if let Some(array) = result.as_array() {
        return Ok(array.clone());
    }
    if let Some(array) = result.get("points").and_then(Value::as_array) {
        return Ok(array.clone());
    }
    if let Some(array) = data.get("points").and_then(Value::as_array) {
        return Ok(array.clone());
    }
    Err(points_parse_error())
}

/// scroll 1 ページの `result.points` を配列として取り出す。
/// scroll では `result.points` のみを受理し、それ以外はエラーとする。
fn scroll_points(result: &Value) -> Result<Vec<Value>, String> {
    result
        .get("points")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(points_parse_error)
}

/// `next_page_offset` を取得する（null は終端）。
fn next_offset(result: &Value) -> Option<Value> {
    result
        .get("next_page_offset")
        .cloned()
        .filter(|value| !value.is_null())
}

/// ループ最後の offset 状態だけから truncated を判定する純粋関数。
/// offset が残っていれば上限到達による部分結果とみなす。
fn scroll_truncated(final_offset: &Option<Value>) -> bool {
    final_offset
        .as_ref()
        .map(|value| !value.is_null())
        .unwrap_or(false)
}

/// count レスポンスから件数を取り出す。
fn count_from_response(data: &Value) -> Result<usize, String> {
    let value = data
        .get("result")
        .and_then(|result| result.get("count"))
        .or_else(|| data.get("count"))
        .ok_or_else(|| "Qdrant count 応答を解析できませんでした".to_string())?;
    if let Some(count) = value.as_u64() {
        return usize::try_from(count).map_err(|_| "Qdrant count が大きすぎます".to_string());
    }
    if let Some(count) = value.as_f64() {
        if count.is_finite() && count >= 0.0 && count.fract() == 0.0 && count < usize::MAX as f64 {
            return Ok(count as usize);
        }
    }
    Err("Qdrant count 応答を解析できませんでした".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_name_uses_percent_encoding_not_form_encoding() {
        assert_eq!(encode_segment("plain_name-1.ok~"), "plain_name-1.ok~");
        assert_eq!(encode_segment("a b/c?d#e&f"), "a%20b%2Fc%3Fd%23e%26f");
        assert_eq!(encode_segment("100%"), "100%25");
        assert_eq!(encode_segment("あ"), "%E3%81%82");
    }

    #[test]
    fn request_bodies_match_qdrant_shape() {
        assert_eq!(
            create_collection_body(768),
            json!({ "vectors": { "size": 768, "distance": "Cosine" } })
        );
        let points = vec![json!({ "id": 1, "vector": [0.1], "payload": {} })];
        assert_eq!(
            upsert_body(&points),
            json!({ "points": vec![json!({ "id": 1, "vector": [0.1], "payload": {} })] })
        );
        assert_eq!(count_body(), json!({ "exact": true }));
        assert_eq!(
            query_body(&[0.5, 0.25], 7),
            json!({ "query": [0.5, 0.25], "limit": 7, "with_payload": true })
        );
    }

    #[test]
    fn scroll_body_uses_256_limit_and_optional_offset() {
        let first = scroll_body(None);
        assert_eq!(
            first,
            json!({ "limit": 256, "with_payload": true, "with_vector": false })
        );
        let second = scroll_body(Some(&json!(42)));
        assert_eq!(second["offset"], json!(42));
        assert_eq!(second["limit"], json!(256));
    }

    #[test]
    fn collection_exists_maps_200_and_404() {
        assert!(exists_from_status(200).unwrap());
        assert!(!exists_from_status(404).unwrap());
        assert!(exists_from_status(500).is_err());
    }

    #[test]
    fn delete_treats_404_as_success() {
        assert!(delete_from_status(200).is_ok());
        assert!(delete_from_status(404).is_ok());
        assert!(delete_from_status(500).is_err());
    }

    #[test]
    fn status_errors_stay_short_and_do_not_include_body() {
        let error = status_error("コレクション作成", 503);
        assert!(error.contains("503"));
        assert!(!error.contains("SECRET_BODY"));
        assert!(!error.contains("http://"));
    }

    #[test]
    fn points_and_count_are_extracted_from_nested_results() {
        let query = json!({ "result": { "points": [{ "id": 1 }] } });
        assert_eq!(
            points_from_response(&query).unwrap(),
            vec![json!({ "id": 1 })]
        );
        let scroll = json!({ "result": { "points": [{ "id": 2 }], "next_page_offset": null } });
        assert_eq!(
            points_from_response(&scroll).unwrap(),
            vec![json!({ "id": 2 })]
        );
        // `{}` は正規の空配列ではないため Err。
        assert!(points_from_response(&json!({})).is_err());

        let count = json!({ "result": { "count": 12 } });
        assert_eq!(count_from_response(&count).unwrap(), 12);
        assert!(count_from_response(&json!({ "result": {} })).is_err());
    }

    #[test]
    fn points_from_response_accepts_empty_array_and_rejects_malformed_shapes() {
        // 正規の空配列は Ok(empty)。
        assert_eq!(
            points_from_response(&json!({ "result": [] })).unwrap(),
            Vec::<Value>::new()
        );
        assert_eq!(
            points_from_response(&json!({ "result": { "points": [] } })).unwrap(),
            Vec::<Value>::new()
        );
        assert_eq!(
            points_from_response(&json!({ "points": [] })).unwrap(),
            Vec::<Value>::new()
        );
        // result 配列形式。
        assert_eq!(
            points_from_response(&json!({ "result": [{ "id": 7 }] })).unwrap(),
            vec![json!({ "id": 7 })]
        );
        // トップレベル points 配列形式。
        assert_eq!(
            points_from_response(&json!({ "points": [{ "id": 8 }] })).unwrap(),
            vec![json!({ "id": 8 })]
        );

        // 欠落・型不正・{}・null・プリミティブは固定文言の Err。
        for bad in [
            json!({}),
            json!(null),
            json!(42),
            json!("points"),
            json!({ "result": {} }),
            json!({ "result": null }),
            json!({ "result": 7 }),
            json!({ "result": { "points": null } }),
            json!({ "result": { "points": "oops" } }),
            json!({ "points": {} }),
            json!({ "points": 1 }),
        ] {
            let err = points_from_response(&bad).unwrap_err();
            assert_eq!(err, points_parse_error());
            assert!(!err.contains("http://"));
        }
    }

    #[test]
    fn scroll_truncated_depends_only_on_final_offset() {
        // 100 ページ目に offset が残る → truncated。
        let last_offset = Some(json!(123));
        assert!(scroll_truncated(&last_offset));
        // offset が null / 無し → 正常終端。
        assert!(!scroll_truncated(&None));
        assert!(!scroll_truncated(&Some(Value::Null)));
        // 途中で offset が無くなった場合（final が None）は truncated にならない。
        let final_offset = next_offset(&json!({
            "points": [],
            "next_page_offset": null
        }));
        assert!(!scroll_truncated(&final_offset));

        // null 以外の JSON はそのまま次の offset として扱われる。
        let final_offset = next_offset(&json!({
            "points": [{ "id": 3 }],
            "next_page_offset": { "id": 3 }
        }));
        assert!(scroll_truncated(&final_offset));
    }

    #[test]
    fn scroll_points_rejects_missing_or_non_array_points() {
        assert_eq!(
            scroll_points(&json!({ "points": [{ "id": 1 }] })).unwrap(),
            vec![json!({ "id": 1 })]
        );
        assert_eq!(
            scroll_points(&json!({ "points": [] })).unwrap(),
            Vec::<Value>::new()
        );
        for bad in [
            json!({}),
            json!(null),
            json!({ "points": null }),
            json!({ "points": {} }),
            json!({ "points": 3 }),
        ] {
            assert_eq!(scroll_points(&bad).unwrap_err(), points_parse_error());
        }
    }

    #[test]
    fn query_dense_malformed_shape_is_error_via_pure_parser() {
        // query_dense と同じ経路（points_from_response の ? 伝播）を検証。
        // HTTP を使わず pure パーサの Err を確認する。
        let malformed = json!({ "result": { "points": "not-an-array" } });
        let propagated = points_from_response(&malformed).map(|_| ());
        assert_eq!(propagated.unwrap_err(), points_parse_error());

        let valid = json!({ "result": { "points": [{ "id": 9 }] } });
        assert_eq!(
            points_from_response(&valid).unwrap(),
            vec![json!({ "id": 9 })]
        );
    }
}
