//! コーパスをチャンク化し埋め込み Qdrant へ投入する。
//! 埋め込み器はローカル llama.cpp / クラウド opencodeGO を設定で切り替える。

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::config::Config;
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
            if entry
                .file_type()
                .map(|kind| kind.is_symlink())
                .unwrap_or(false)
            {
                continue;
            }
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

/// 埋め込み器の経路。`cloud.enabled` で一度だけ選び、バッチ間で使い回す。
enum Embedder {
    Local(crate::llama::LlamaClient),
    Cloud(Box<crate::cloud::CloudClient>),
}

/// クラウド経路を選ぶかどうか（IO を伴わない純粋判定）。
fn use_cloud_embedder(config: &Config) -> bool {
    config.cloud.enabled
}

/// Qdrant コレクションの次元。クラウド経路では解決済み次元を優先する。
fn resolved_vector_size(config: &Config) -> usize {
    if use_cloud_embedder(config) {
        config.cloud_embed_dimension().unwrap_or(config.vector_size)
    } else {
        config.vector_size
    }
}

impl Embedder {
    fn from_config(config: &Config) -> Result<Self, String> {
        if use_cloud_embedder(config) {
            Ok(Self::Cloud(Box::new(
                crate::cloud::CloudClient::from_config(config)?,
            )))
        } else {
            Ok(Self::Local(crate::llama::LlamaClient::new(
                &config.llama_server_url,
                &config.embed_model,
                config.llama_timeout_secs,
            )))
        }
    }

    fn embed(
        &self,
        texts: &[String],
        prefix: &str,
        config: &Config,
    ) -> Result<Vec<Vec<f64>>, String> {
        match self {
            Self::Local(client) => client.embed(texts, prefix, None),
            Self::Cloud(client) => {
                // 旧 Python 契約に合わせ、クラウド経路では prefix を付与しない。
                let inputs: Vec<String> = texts.to_vec();
                let vectors = client.embed(&inputs)?;
                if config
                    .cloud_embed_dimension()
                    .is_some_and(|dim| vectors.iter().any(|vector| vector.len() != dim))
                {
                    return Err("埋め込みベクトルの次元が一致しません".to_string());
                }
                Ok(vectors)
            }
        }
    }
}

pub fn run(config: &Config, corpus_dir: &Path, recreate: bool, yes: bool) -> Result<(), String> {
    // 旧 Python resolve_corpus_dir() 相当の範囲制限。main.rs は --corpus-dir の
    // 上書き値を config.corpus_dir へ反映してから渡すため、実効パスをここで検証
    // すれば環境変数由来・CLI 上書き由来の両方をカバーできる。
    let corpus_dir = config.resolve_corpus_dir(corpus_dir)?;
    if recreate && !yes {
        return Err("--recreate には --yes が必要です".to_string());
    }
    if !corpus_dir.is_dir() {
        return Err("corpus dir not found".to_string());
    }

    let embedder = Embedder::from_config(config)?;
    let qdrant = QdrantClient::new(
        &config.qdrant_url,
        &config.collection,
        config.qdrant_timeout_secs,
    );

    if recreate {
        qdrant.delete_collection()?;
    }
    if !qdrant.collection_exists()? {
        qdrant.create_collection(resolved_vector_size(config))?;
    }

    let files = collect_markdown(&corpus_dir)?;
    let mut points = Vec::new();
    for path in &files {
        points.extend(iter_chunks(
            path,
            &corpus_dir,
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
        let vectors = embedder.embed(&texts, &config.embed_prefix_passage, config)?;
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
    use crate::config::CloudConfig;

    fn test_config() -> Config {
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
            cloud: CloudConfig {
                enabled: false,
                chat_enabled: false,
                base_url: "https://opencode.ai/inference/go/openai/v1".to_string(),
                embed_model: String::new(),
                embed_dimension: None,
                model: "deepseek-v4.1-flash".to_string(),
                timeout_secs: 60.0,
                api_key: None,
                api_key_name: None,
            },
        }
    }

    #[test]
    fn embedder_selection_follows_cloud_enabled_flag() {
        let mut config = test_config();
        assert!(!use_cloud_embedder(&config));
        // ローカル経路は HTTP を送らずに生成できる。
        assert!(matches!(
            Embedder::from_config(&config),
            Ok(Embedder::Local(_))
        ));

        config.cloud.enabled = true;
        assert!(use_cloud_embedder(&config));
    }

    #[test]
    fn cloud_path_uses_resolved_embed_dimension() {
        let mut config = test_config();
        assert_eq!(resolved_vector_size(&config), 2048);

        config.cloud.enabled = true;
        config.cloud.embed_model = "text-embedding-3-small".to_string();
        config.vector_size = 1536;
        assert_eq!(resolved_vector_size(&config), 1536);

        config.cloud.embed_dimension = Some(4096);
        assert_eq!(resolved_vector_size(&config), 4096);
    }

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

    /// リポジトリ外のディレクトリを構築する（実在しなくてよい）。
    fn outside_repo_dir() -> PathBuf {
        let root = crate::config::repo_root();
        root.parent().unwrap_or(&root).join("outside-corpus-dir")
    }

    #[test]
    fn run_rejects_external_corpus_dir_before_any_io() {
        let config = test_config();
        let outside = outside_repo_dir();
        let err = run(&config, &outside, false, false).unwrap_err();
        assert!(err.contains("RAG_ALLOW_EXTERNAL_CORPUS=1"));
        assert!(!err.contains(outside.to_string_lossy().as_ref()));
    }

    #[test]
    fn run_allows_external_corpus_dir_only_with_opt_in() {
        let mut config = test_config();
        config.allow_external_corpus = true;
        // 存在しないパスなので、HTTP に進む前の is_dir チェックで止まる。
        let err = run(&config, &outside_repo_dir(), false, false).unwrap_err();
        assert_eq!(err, "corpus dir not found");
    }

    #[test]
    fn run_accepts_relative_corpus_dir_under_repo_root() {
        let config = test_config();
        let err = run(&config, Path::new("missing-in-repo-corpus"), false, false).unwrap_err();
        assert_eq!(err, "corpus dir not found");
    }
}
