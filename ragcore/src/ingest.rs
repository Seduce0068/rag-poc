//! コーパスをチャンク化し llama-server で埋め込み Qdrant へ投入する。

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::config::Config;
use crate::llama::LlamaClient;
use crate::qdrant::QdrantClient;
use crate::{chunk_text, point_id};

const EMBED_BATCH_SIZE: usize = 16;
const UPSERT_BATCH_SIZE: usize = 64;
const HEADING_RE: &str = "#";

fn is_heading(line: &str) -> bool {
    line.starts_with(HEADING_RE)
}

fn heading_text(line: &str) -> &str {
    line.trim_start_matches('#').trim()
}

fn collect_markdown(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|e| format!("read_dir failed: {e}"))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|v| v.to_str()) == Some("md") {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// 1 ファイルを見出し区切り＋文字ウィンドウで分割する。
pub fn iter_chunks(path: &Path, root: &Path, chunk_size: usize, overlap: usize) -> Vec<Value> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return Vec::new(),
    };
    let source = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let title = text
        .lines()
        .find(|line| is_heading(line))
        .map(|line| heading_text(line).to_string())
        .unwrap_or_else(|| {
            path.file_stem()
                .map(|v| v.to_string_lossy().to_string())
                .unwrap_or_default()
        });

    let mut sections: Vec<String> = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    for line in text.lines() {
        if is_heading(line) {
            if !buf.is_empty() {
                let body = buf.join("\n").trim().to_string();
                if !body.is_empty() {
                    sections.push(body);
                }
            }
            buf = vec![line.to_string()];
        } else {
            buf.push(line.to_string());
        }
    }
    if !buf.is_empty() {
        let body = buf.join("\n").trim().to_string();
        if !body.is_empty() {
            sections.push(body);
        }
    }

    let mut points = Vec::new();
    let mut index = 0u64;
    for section in sections {
        let first_line = section.lines().next().unwrap_or("");
        let (prefix, body) = if is_heading(first_line) {
            let rest = section.strip_prefix(first_line).unwrap_or("").trim();
            (Some(first_line.to_string()), rest.to_string())
        } else {
            (None, section)
        };
        let mut pieces = chunk_text(&body, chunk_size, overlap).unwrap_or_default();
        if pieces.is_empty() {
            pieces.push(String::new());
        }
        for piece in pieces {
            let text_chunk = match &prefix {
                Some(p) if !piece.is_empty() => format!("{p}\n{piece}"),
                _ => piece.clone(),
            };
            points.push(json!({
                "id": point_id(&source, index),
                "payload": {
                    "source": source,
                    "title": title,
                    "text": text_chunk,
                    "chunk_index": index,
                },
                "vector": Vec::<f64>::new(),
            }));
            index += 1;
        }
    }
    points
}

pub fn run(config: &Config, corpus_dir: &Path, recreate: bool, yes: bool) -> Result<(), String> {
    if recreate && !yes {
        return Err("--recreate には --yes が必要です".to_string());
    }
    if !corpus_dir.is_dir() {
        return Err("corpus dir not found".to_string());
    }

    let llama = LlamaClient::new(
        &config.llama_server_url,
        &config.embed_model,
        config.llama_timeout_secs,
    );
    let qdrant = QdrantClient::new(
        &config.qdrant_url,
        &config.collection,
        config.qdrant_timeout_secs,
    );

    if recreate {
        qdrant.delete_collection()?;
    }
    if !qdrant.collection_exists()? {
        qdrant.create_collection(config.vector_size)?;
    }

    let files = collect_markdown(corpus_dir)?;
    let mut points = Vec::new();
    for path in &files {
        points.extend(iter_chunks(
            path,
            corpus_dir,
            config.chunk_size,
            config.chunk_overlap,
        ));
    }
    eprintln!("[ingest] files={} chunks={}", files.len(), points.len());

    for batch in points.chunks_mut(EMBED_BATCH_SIZE) {
        let texts: Vec<String> = batch
            .iter()
            .map(|point| {
                point["payload"]["text"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        let vectors = llama.embed(&texts, &config.embed_prefix_passage, None)?;
        if vectors.len() != batch.len() {
            return Err("embedded vector count mismatch".to_string());
        }
        for (point, vector) in batch.iter_mut().zip(vectors) {
            point["vector"] = json!(vector);
        }
    }
    for batch in points.chunks(UPSERT_BATCH_SIZE) {
        qdrant.upsert(batch)?;
    }
    eprintln!("[ingest] done: {} points", points.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter_chunks_uses_character_windows() {
        let dir = std::env::temp_dir().join("ragcore-ingest-test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("doc.md");
        fs::write(&path, "# Title\nあいうえおかきくけこ").unwrap();
        let points = iter_chunks(&path, &dir, 4, 1);
        assert!(!points.is_empty());
        assert!(points[0]["payload"]["text"]
            .as_str()
            .unwrap()
            .starts_with("# Title"));
        let _ = fs::remove_dir_all(&dir);
    }
}
