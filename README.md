# rag_poc — Rust のみのローカル RAG

> - **何か**: llama.cpp (llama-server) または opencodeGO の埋め込み + Qdrant を使う RAG を Rust だけで実装したもの。
> - **前提**: 既定は llama-server / Qdrant がローカル起動済み、Rust toolchain。
> - **最初のコマンド**: `cargo build --manifest-path ragcore/Cargo.toml` → `ragcore/target/debug/ragcore ingest --corpus-dir <DIR>` → `ragcore/target/debug/ragcore query '質問' --json`
> - **検証**: `cargo test --manifest-path ragcore/Cargo.toml` と `cargo clippy --manifest-path ragcore/Cargo.toml -- -D warnings -D clippy::undocumented_unsafe_blocks`

## まず動かす

リポジトリルートで次を実行する（Kubuntu 想定）。
別ターミナルで llama-server を起動しておく（Ruri の GGUF を使う例）。

```bash
llama-server -m ~/models/ruri-F16.gguf --embedding --port 8090 --host 127.0.0.1 -ngl 0 -c 512
```

```bash
cargo build --manifest-path ragcore/Cargo.toml
export LLAMA_SERVER_URL=http://127.0.0.1:8090
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

Markdown の知識源を llama-server（既定）または opencodeGO（opt-in）でベクトル化して
Qdrant に保存し、質問に対して関連する文章を返す。Python は使わない。検索モードは
`dense` / `keyword` / `hybrid` の 3 種が実装済み。

埋め込みの切り替えは `RAG_CLOUD_ENABLED=1` で行う。既定 `0` では外部送信・課金は
一切発生しない。クラウドを使う場合は `RAG_CLOUD_EMBED_MODEL` と
`OPENCODE_GO_API_KEY` が必須で、埋め込み次元と `RAG_VECTOR_SIZE` が一致しないと
起動前に拒否される。API キー値はログ・エラー・成果物へ出力しない。

未実装（旧 Python 実装にあった機能）:

- `hybrid-rerank` は `hybrid` と同一で、LLM による再ランクは行わない
- 回答生成（`answer` / `sources`）と `--with-answer`
- 接続先のリモート制限（`RAG_ALLOW_REMOTE_ENDPOINTS` 等）と
  コーパス範囲制限（`RAG_ALLOW_EXTERNAL_CORPUS`）
- クラウド埋め込みのリトライ（429/5xx で即失敗）
- `RAG_CLOUD_CHAT_ENABLED` は受け付けるが no-op（chat クライアント未実装）
- ハイブリッド検索で dense / keyword が失敗した場合は部分結果を返し、
  その理由は stderr にも JSON にも残さない

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
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8090` |
| `QDRANT_URL` | `http://127.0.0.1:6333` |
| `EMBED_MODEL` | `qwen2.5:3b` |
| `CHAT_MODEL` | `qwen2.5:3b` |
| `RAG_COLLECTION` | `rag_poc_docs` |
| `RAG_VECTOR_SIZE` | `2048` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` |
| `RAG_TOP_K` | `10` |
| `RAG_RERANK_MAX_DOCS` | `5` |
| `RAG_EMBED_PREFIX_QUERY` / `PASSAGE` | 空 |
| `RAG_CLOUD_ENABLED` | `0`（`"1"` のみ有効） |
| `RAG_CLOUD_EMBED_MODEL` | 空（有効時は必須） |
| `RAG_CLOUD_EMBED_DIMENSION` | 未設定（モデル表から解決） |
| `RAG_CLOUD_BASE_URL` | `https://opencode.ai/inference/go/openai/v1` |
| `OPENCODE_GO_API_KEY` | 未設定（クラウド時に必須） |

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
- `ragcore/src/` — `config` / `llama` / `cloud` / `qdrant` / `search` / `ingest` / `query` / `evaluate`
