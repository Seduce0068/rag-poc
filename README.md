# rag_poc — Rust のみのローカル RAG

> - **何か**: Ollama + Qdrant を使う RAG を Rust だけで実装したもの。
> - **前提**: Ollama / Qdrant がローカル起動済み、Rust toolchain。
> - **最初のコマンド**: `cargo build --manifest-path ragcore/Cargo.toml` → `ragcore/target/debug/ragcore ingest --corpus-dir <DIR>` → `ragcore/target/debug/ragcore query '質問' --json`
> - **検証**: `cargo test --manifest-path ragcore/Cargo.toml` と `cargo clippy --manifest-path ragcore/Cargo.toml -- -D warnings -D clippy::undocumented_unsafe_blocks`

## まず動かす

リポジトリルートで次を実行する（Kubuntu 想定）。

```bash
cargo build --manifest-path ragcore/Cargo.toml
export OLLAMA_URL=http://127.0.0.1:11434
export QDRANT_URL=http://127.0.0.1:6333
export EMBED_MODEL=kun432/cl-nagoya-ruri-base
export RAG_VECTOR_SIZE=768
export CHUNK_SIZE=300
export CHUNK_OVERLAP=50
export RAG_EMBED_PREFIX_QUERY='クエリ: '
export RAG_EMBED_PREFIX_PASSAGE='文章: '

ragcore/target/debug/ragcore ingest \
  --corpus-dir "$RAG_CORPUS_DIR" \
  --collection knowledge_rust
ragcore/target/debug/ragcore query 'Pin<&mut Self> が必要な理由は？' \
  --mode hybrid --json
```

## 目的

Markdown の知識源を Ollama でベクトル化して Qdrant に保存し、質問に対して
関連する文章を返す。Python は使わない。検索モードは
`dense` / `keyword` / `hybrid` の 3 種が実装済み。

未実装（旧 Python 実装にあった機能）:

- `hybrid-rerank` は `hybrid` と同一で、LLM による再ランクは行わない
- 回答生成（`answer` / `sources`）と `--with-answer`
- 接続先のリモート制限（`RAG_ALLOW_REMOTE_ENDPOINTS` 等）と
  コーパス範囲制限（`RAG_ALLOW_EXTERNAL_CORPUS`）

## CLI

```bash
# 投入（--recreate は --yes 必須）
ragcore/target/debug/ragcore ingest --corpus-dir DIR --collection NAME [--recreate --yes]

# 検索
ragcore/target/debug/ragcore query '質問' --mode hybrid --top-k 10 --json

# 評価
ragcore/target/debug/ragcore evaluate --qrels qrels.json --output eval-results.json
```

stdin JSON の op モード（`{"op":"cosine","a":[1.0],"b":[1.0]}`）も後方互換で残している。

## 主要な環境変数

| 変数 | 既定値 |
| --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `QDRANT_URL` | `http://127.0.0.1:6333` |
| `EMBED_MODEL` | `qwen2.5:3b` |
| `CHAT_MODEL` | `qwen2.5:3b` |
| `RAG_COLLECTION` | `rag_poc_docs` |
| `RAG_VECTOR_SIZE` | `2048` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` |
| `RAG_TOP_K` | `10` |
| `RAG_RERANK_MAX_DOCS` | `5` |
| `RAG_EMBED_PREFIX_QUERY` / `PASSAGE` | 空 |

Ruri 系モデルを使う場合は `RAG_VECTOR_SIZE=768` かつ `CHUNK_SIZE<=512` が必須。

## テスト

```bash
cargo fmt --manifest-path ragcore/Cargo.toml -- --check
cargo test --manifest-path ragcore/Cargo.toml
cargo clippy --manifest-path ragcore/Cargo.toml -- -D warnings -D clippy::undocumented_unsafe_blocks
betterleaks git
```

## ドキュメント

- `docs/` 配下は旧 Python 実装の設計・実測資料。現行 CLI の仕様は本 README と
  `ragcore/src/` のコードを正とする。
- `ragcore/src/` — `config` / `ollama` / `qdrant` / `search` / `ingest` / `query` / `evaluate`
