//! stdin JSON → stdout JSON の薄い CLI ラッパー。
//!
//! 入力: `{"op":"chunk","text":"...","chunk_size":300,"overlap":50}`
//! 入力: `{"op":"cosine_batch","pairs":[[[1,0],[1,0]],[[1,0],[0,1]]]}`
//! 出力: op ごとの結果値、失敗時は `{"error":"..."}` を出して終了コード 1。

use std::io::{self, Read};

use ragcore::{
    chunk_text, corpus_fingerprint, cosine, distribution, latency_stats, ndcg_at, point_id,
    recall_at,
};
use serde_json::{json, Value};

fn main() {
    let mut input = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut input) {
        fail(&format!("stdin の読み込みに失敗: {err}"));
    }
    let value: Value = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(err) => fail(&format!("入力 JSON の解析に失敗: {err}")),
    };
    let op = value
        .get("op")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let result = match op.as_str() {
        "chunk" => run_chunk(&value),
        "cosine" => run_cosine(&value),
        "cosine_batch" => run_cosine_batch(&value),
        "distribution" => run_distribution(&value),
        "recall" => run_recall(&value),
        "ndcg" => run_ndcg(&value),
        "latency" => run_latency(&value),
        "fingerprint" => run_fingerprint(&value),
        "point_id" => run_point_id(&value),
        other => Err(format!("未知の op: {other}")),
    };

    match result {
        Ok(output) => println!("{output}"),
        Err(message) => fail(&message),
    }
}

/// エラー JSON を stdout に出し、終了コード 1 で終了する。
fn fail(message: &str) -> ! {
    println!("{}", json!({ "error": message }));
    std::process::exit(1);
}

fn require_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid string field: {key}"))
}

fn require_usize(value: &Value, key: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| format!("missing or invalid non-negative integer field: {key}"))
}

fn require_f64_array(value: &Value, key: &str) -> Result<Vec<f64>, String> {
    let array = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid array field: {key}"))?;
    array
        .iter()
        .map(|item| {
            item.as_f64()
                .ok_or_else(|| format!("{key} must contain only numbers"))
        })
        .collect()
}

fn require_string_array(value: &Value, key: &str) -> Result<Vec<String>, String> {
    let array = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid array field: {key}"))?;
    array
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{key} must contain only strings"))
        })
        .collect()
}

fn to_json_string<T: serde::Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string(value).map_err(|err| err.to_string())
}

fn run_chunk(value: &Value) -> Result<String, String> {
    let text = require_str(value, "text")?;
    let chunk_size = require_usize(value, "chunk_size")?;
    let overlap = require_usize(value, "overlap")?;
    to_json_string(&chunk_text(text, chunk_size, overlap)?)
}

fn run_cosine(value: &Value) -> Result<String, String> {
    let a = require_f64_array(value, "a")?;
    let b = require_f64_array(value, "b")?;
    to_json_string(&cosine(&a, &b)?)
}

/// `Value` を数値配列として解釈する（`require_f64_array` のキー無し版）。
fn f64_array_from(value: &Value, context: &str) -> Result<Vec<f64>, String> {
    let array = value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?;
    array
        .iter()
        .map(|item| {
            item.as_f64()
                .ok_or_else(|| format!("{context} must contain only numbers"))
        })
        .collect()
}

/// 複数のコサイン類似度を一括計算し、配列そのものを返す（丸めなし）。
/// 不正なペアがあれば 1 件も返さずエラーとする。
fn run_cosine_batch(value: &Value) -> Result<String, String> {
    let pairs = value
        .get("pairs")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing or invalid array field: pairs".to_string())?;
    let mut scores = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let elements = pair
            .as_array()
            .ok_or_else(|| format!("pairs[{index}] must be an array"))?;
        if elements.len() != 2 {
            return Err(format!("pairs[{index}] must contain exactly two vectors"));
        }
        let a = f64_array_from(&elements[0], &format!("pairs[{index}][0]"))?;
        let b = f64_array_from(&elements[1], &format!("pairs[{index}][1]"))?;
        if a.is_empty() || b.is_empty() {
            return Err(format!("pairs[{index}] vectors must not be empty"));
        }
        scores.push(cosine(&a, &b).map_err(|err| format!("pairs[{index}]: {err}"))?);
    }
    to_json_string(&scores)
}

fn run_distribution(value: &Value) -> Result<String, String> {
    let scores = require_f64_array(value, "scores")?;
    to_json_string(&distribution(&scores)?)
}

fn run_recall(value: &Value) -> Result<String, String> {
    let relevant = require_string_array(value, "relevant")?;
    let retrieved = require_string_array(value, "retrieved")?;
    let k = require_usize(value, "k")?;
    to_json_string(&recall_at(&relevant, &retrieved, k))
}

fn run_ndcg(value: &Value) -> Result<String, String> {
    let relevant = require_string_array(value, "relevant")?;
    let retrieved = require_string_array(value, "retrieved")?;
    let k = require_usize(value, "k")?;
    // metrics 側は丸めない。CLI は呼び出し側として 6 桁に揃える。
    let raw = ndcg_at(&relevant, &retrieved, k);
    to_json_string(&((raw * 1_000_000.0).round() / 1_000_000.0))
}

fn run_latency(value: &Value) -> Result<String, String> {
    let samples = require_f64_array(value, "samples")?;
    to_json_string(&latency_stats(&samples)?)
}

fn run_fingerprint(value: &Value) -> Result<String, String> {
    let paths = value
        .get("paths")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing or invalid array field: paths".to_string())?;
    let mut parsed: Vec<(String, Vec<u8>)> = Vec::new();
    for (index, item) in paths.iter().enumerate() {
        let path = item
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("paths[{index}].path must be a string"))?
            .to_string();
        let bytes = if let Some(text) = item.get("content").and_then(Value::as_str) {
            text.as_bytes().to_vec()
        } else if let Some(text) = item.get("text").and_then(Value::as_str) {
            text.as_bytes().to_vec()
        } else if let Some(array) = item.get("bytes").and_then(Value::as_array) {
            array
                .iter()
                .map(|number| {
                    number
                        .as_u64()
                        .and_then(|value| u8::try_from(value).ok())
                        .ok_or_else(|| format!("paths[{index}].bytes must contain 0..255"))
                })
                .collect::<Result<Vec<u8>, String>>()?
        } else {
            return Err(format!("paths[{index}] needs content / text / bytes"));
        };
        parsed.push((path, bytes));
    }
    to_json_string(&corpus_fingerprint(&parsed))
}

fn run_point_id(value: &Value) -> Result<String, String> {
    let source = require_str(value, "source")?;
    let index = require_usize(value, "chunk_index")?;
    let chunk_index =
        u64::try_from(index).map_err(|_| "chunk_index is too large for u64".to_string())?;
    to_json_string(&point_id(source, chunk_index))
}
