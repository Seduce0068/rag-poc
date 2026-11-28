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

/// 文件内容・パス・認証情報を出さずに、失敗件数だけを伝える固定文言。
fn read_dir_error(count: usize) -> String {
    format!("ディレクトリ走査に失敗しました（{count} 件）")
}

fn file_type_error(count: usize) -> String {
    format!("ファイル種別の判定に失敗しました（{count} 件）")
}

fn read_file_error(count: usize) -> String {
    format!("ファイルの読み取りに失敗しました（{count} 件）")
}

fn chunk_error(count: usize) -> String {
    format!("チャンク化に失敗しました（{count} 件）")
}

fn is_heading(line: &str) -> bool {
    line.starts_with(HEADING_RE)
}

fn heading_text(line: &str) -> &str {
    line.trim_start_matches('#').trim()
}

fn collect_markdown(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    let mut read_dir_errors = 0usize;
    let mut file_type_errors = 0usize;
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => {
                read_dir_errors += 1;
                continue;
            }
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => {
                    read_dir_errors += 1;
                    continue;
                }
            };
            let path = entry.path();
            match entry.file_type() {
                Ok(kind) if kind.is_symlink() => continue,
                Ok(_) => {}
                Err(_) => {
                    file_type_errors += 1;
                    continue;
                }
            }
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|v| v.to_str()) == Some("md") {
                files.push(path);
            }
        }
    }
    // 一件でも失敗があれば、内容を返さず明示的に失敗を伝播させる。
    if read_dir_errors > 0 {
        return Err(read_dir_error(read_dir_errors));
    }
    if file_type_errors > 0 {
        return Err(file_type_error(file_type_errors));
    }
    files.sort();
    Ok(files)
}

/// 1 ファイルを見出し区切り＋文字ウィンドウで分割する。
/// 読み取り・チャンク化の失敗は黙って空にせず `Err` で伝播する。
pub fn iter_chunks(
    path: &Path,
    root: &Path,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<Value>, String> {
    let text = fs::read_to_string(path).map_err(|_| read_file_error(1))?;
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
    let mut chunk_errors = 0usize;
    for section in sections {
        let first_line = section.lines().next().unwrap_or("");
        let (prefix, body) = if is_heading(first_line) {
            let rest = section.strip_prefix(first_line).unwrap_or("").trim();
            (Some(first_line.to_string()), rest.to_string())
        } else {
            (None, section)
        };
        let mut pieces = match chunk_text(&body, chunk_size, overlap) {
            Ok(pieces) => pieces,
            Err(_) => {
                chunk_errors += 1;
                continue;
            }
        };
        // 見出しのみのセクションで空チャンクを残す挙動は旧 Python 契約のため維持する。
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
    if chunk_errors > 0 {
        return Err(chunk_error(chunk_errors));
    }
    Ok(points)
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

    // 走査・読み取り・チャンク化は HTTP リクエストより前に完了させる。
    // 失敗した場合は collection 変更も埋め込みも行わない。
    let files = collect_markdown(&corpus_dir)?;
    let mut points = Vec::new();
    for path in &files {
        points.extend(iter_chunks(
            path,
            &corpus_dir,
            config.chunk_size,
            config.chunk_overlap,
        )?);
    }
    eprintln!("[ingest] files={} chunks={}", files.len(), points.len());

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
        let points = iter_chunks(&path, &dir, 4, 1).unwrap();
        assert!(!points.is_empty());
        assert!(points[0]["payload"]["text"]
            .as_str()
            .unwrap()
            .starts_with("# Title"));
        let _ = fs::remove_dir_all(&dir);
    }

    /// 読み取り失敗を模した純粋なヘルパー: 固定文言かつ件数以外の情報を出さない。
    #[test]
    fn read_failure_errors_report_only_fixed_text_and_count() {
        let err = read_file_error(1);
        assert_eq!(err, "ファイルの読み取りに失敗しました（1 件）");
        // パス・内容・認証情報が混入しないこと。
        assert!(!err.contains('/'));
        assert!(!err.contains("secret"));
        assert_eq!(chunk_error(2), "チャンク化に失敗しました（2 件）");
        assert_eq!(read_dir_error(3), "ディレクトリ走査に失敗しました（3 件）");
        assert_eq!(
            file_type_error(4),
            "ファイル種別の判定に失敗しました（4 件）"
        );
    }

    /// 見出しのみのセクションは空チャンクを 1 件残す（旧 Python 契約・回帰維持）。
    #[test]
    fn iter_chunks_keeps_empty_body_as_one_empty_chunk() {
        let dir = std::env::temp_dir().join("ragcore-ingest-empty-section-test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("headings-only.md");
        fs::write(&path, "# One\n## Two\n### Three\n").unwrap();
        let points = iter_chunks(&path, &dir, 800, 100).unwrap();
        // 3 つの見出しセクションそれぞれに空チャンクが 1 件残る。
        assert_eq!(points.len(), 3);
        assert!(points
            .iter()
            .all(|p| p["payload"]["text"].as_str().unwrap().is_empty()));
        let _ = fs::remove_dir_all(&dir);
    }

    /// オーバーラップ不正による chunk_text 失敗は Err として伝播する。
    #[test]
    fn iter_chunks_propagates_chunk_text_failure() {
        let dir = std::env::temp_dir().join("ragcore-ingest-chunk-error-test");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("doc.md");
        fs::write(&path, "# Title\n本文").unwrap();
        let err = iter_chunks(&path, &dir, 4, 4).unwrap_err();
        assert_eq!(err, "チャンク化に失敗しました（1 件）");
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
