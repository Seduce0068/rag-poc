//! Rust のみの RAG CLI。stdin JSON の op モードも後方互換で維持する。

use std::io::{self, Read};
use std::path::Path;

use clap::{Parser, Subcommand};
use serde_json::{json, Value};

use ragcore::{
    check_eval_results, chunk_text, corpus_fingerprint, cosine, distribution, latency_stats,
    ndcg_at, point_id, recall_at, rrf_fuse,
};

#[derive(Parser)]
#[command(name = "ragcore", about = "Rust-only RAG CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// コーパスを埋め込みして Qdrant へ投入する
    Ingest {
        #[arg(long, env = "RAG_CORPUS_DIR")]
        corpus_dir: Option<String>,
        #[arg(long, env = "RAG_COLLECTION")]
        collection: Option<String>,
        #[arg(long)]
        recreate: bool,
        #[arg(long)]
        yes: bool,
    },
    /// 質問して検索結果を返す
    Query {
        question: String,
        #[arg(long, default_value = "hybrid")]
        mode: String,
        #[arg(long, env = "RAG_TOP_K")]
        top_k: Option<usize>,
        #[arg(long)]
        json: bool,
    },
    /// qrels で検索品質を評価する
    Evaluate {
        #[arg(long)]
        qrels: String,
        #[arg(long)]
        output: String,
    },
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let first = args.get(1).map(String::as_str).unwrap_or_default();
    let is_cli = matches!(
        first,
        "ingest" | "query" | "evaluate" | "--help" | "-h" | "help"
    );
    if is_cli {
        match Cli::try_parse() {
            Ok(cli) => {
                if let Err(err) = run_cli(cli) {
                    eprintln!("error: {err}");
                    std::process::exit(1);
                }
            }
            Err(err) => err.exit(),
        }
        return;
    }
    run_op_mode();
}

fn run_cli(cli: Cli) -> Result<(), String> {
    let mut config = ragcore::config::Config::from_env()?;
    match cli.command {
        Command::Ingest {
            corpus_dir,
            collection,
            recreate,
            yes,
        } => {
            if let Some(dir) = corpus_dir {
                config.corpus_dir = std::path::PathBuf::from(dir);
            }
            if let Some(name) = collection {
                config.collection = name;
            }
            ragcore::ingest::run(&config, &config.corpus_dir, recreate, yes)
        }
        Command::Query {
            question,
            mode,
            top_k,
            json: as_json,
        } => {
            if question.chars().count() > config.question_max_chars {
                return Err("question too long".to_string());
            }
            let k = top_k.filter(|&value| value > 0).unwrap_or(config.top_k);
            let result = ragcore::query::search(&config, &question, &mode, k)?;
            let hits = result.hits;
            let (answer, answer_error) = ragcore::query::generate_answer(&config, &question, &hits);
            let mut seen = std::collections::HashSet::new();
            let sources: Vec<Value> = hits
                .iter()
                .filter_map(|hit| hit["source"].as_str())
                .filter(|source| seen.insert((*source).to_string()))
                .map(|source| json!(source))
                .collect();
            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "mode": mode,
                        "hits": hits,
                        "errors": result.errors,
                        "answer": answer,
                        "sources": sources,
                        "answer_error": answer_error,
                        "rerank_unchanged": ragcore::search::rerank_unchanged_status(result.unchanged),
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                for hit in hits {
                    println!("{}\t{}\t{}", hit["score"], hit["source"], hit["text"]);
                }
                println!("\n{}", answer);
            }
            Ok(())
        }
        Command::Evaluate { qrels, output } => {
            ragcore::evaluate::run(&config, Path::new(&qrels), Path::new(&output))
        }
    }
}

fn run_op_mode() {
    let mut input = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut input) {
        fail(&format!("stdin read failed: {err}"));
    }
    let value: Value = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(err) => fail(&format!("invalid JSON: {err}")),
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
        "check_eval_results" => run_check_eval_results(&value),
        "rrf_fuse" => run_rrf_fuse(&value),
        other => Err(format!("unknown op: {other}")),
    };
    match result {
        Ok(output) => println!("{output}"),
        Err(message) => fail(&message),
    }
}

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
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid array field: {key}"))?
        .iter()
        .map(|item| {
            item.as_f64()
                .ok_or_else(|| format!("{key} must contain only numbers"))
        })
        .collect()
}

fn require_string_array(value: &Value, key: &str) -> Result<Vec<String>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid array field: {key}"))?
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

fn f64_array_from(value: &Value, context: &str) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{context} must be an array"))?
        .iter()
        .map(|item| {
            item.as_f64()
                .ok_or_else(|| format!("{context} must contain only numbers"))
        })
        .collect()
}

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
        scores.push(cosine(&a, &b).map_err(|err| format!("pairs[{index}]: {err}"))?);
    }
    to_json_string(&scores)
}

fn run_distribution(value: &Value) -> Result<String, String> {
    to_json_string(&distribution(&require_f64_array(value, "scores")?)?)
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
    let raw = ndcg_at(&relevant, &retrieved, k);
    to_json_string(&((raw * 1_000_000.0).round() / 1_000_000.0))
}

fn run_latency(value: &Value) -> Result<String, String> {
    to_json_string(&latency_stats(&require_f64_array(value, "samples")?)?)
}

fn run_fingerprint(value: &Value) -> Result<String, String> {
    let paths = value
        .get("paths")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing or invalid array field: paths".to_string())?;
    let mut parsed = Vec::with_capacity(paths.len());
    for item in paths {
        let path = item
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .or_else(|| item.get("text").and_then(Value::as_str))
            .unwrap_or_default()
            .as_bytes()
            .to_vec();
        parsed.push((path, content));
    }
    to_json_string(&corpus_fingerprint(&parsed))
}

fn run_point_id(value: &Value) -> Result<String, String> {
    let source = require_str(value, "source")?;
    let index = require_usize(value, "chunk_index")?;
    to_json_string(&point_id(source, u64::try_from(index).unwrap_or(u64::MAX)))
}

fn run_check_eval_results(value: &Value) -> Result<String, String> {
    let paths = value
        .get("paths")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing or invalid array field: paths".to_string())?;
    let mut files = Vec::with_capacity(paths.len());
    for item in paths {
        let path = item
            .as_str()
            .ok_or_else(|| "paths must contain strings".to_string())?;
        let content =
            std::fs::read_to_string(path).map_err(|_| "path could not be read".to_string())?;
        let name = Path::new(path)
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or(path)
            .to_string();
        files.push((name, content));
    }
    to_json_string(&check_eval_results(&files))
}

fn run_rrf_fuse(value: &Value) -> Result<String, String> {
    let rankings_value = value
        .get("rankings")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing or invalid array field: rankings".to_string())?;
    let rankings: Vec<Vec<Value>> = rankings_value
        .iter()
        .map(|item| {
            item.as_array()
                .cloned()
                .ok_or_else(|| "rankings must contain arrays".to_string())
        })
        .collect::<Result<_, _>>()?;
    let top_k = require_usize(value, "top_k")?;
    to_json_string(&rrf_fuse(&rankings, top_k))
}
