# rag_poc — ローカル RAG 検証

> - **何か**: ローカル Ollama + Qdrant で日本語特化埋め込み Ruri の RAG を試す PoC。
> - **前提**: Ollama / Qdrant がローカル起動済み、Python 3.10 以降（クラウドは既定無効）。
> - **最初のコマンド**: `python3 -m pip install -r rag_poc/requirements.txt` → `ollama pull kun432/cl-nagoya-ruri-base` → `python3 rag_poc/ingest.py`。
> - **検証**: `python3 rag_poc/test_rag.py`（全 270 件、外部サービス接続なし）。
> - **docs への導線**: [docs/index.md](docs/index.md) → [背景](docs/explanation.md) / [設定・実測値](docs/reference.md)。

## まず動かす（5分）

ローカル Ollama + Qdrant で日本語特化埋め込み Ruri の RAG を最短で試す。リポジトリルートで以下を順にコピペ実行する（Kubuntu 想定）。

```bash
# 0. Ruri 設定（2026-11-26 時点の再現条件。サブセットで短時間に確認）
export RAG_CORPUS_DIR=rag_poc/corpus-subset
export RAG_COLLECTION=rag_poc_ruri
export EMBED_MODEL=kun432/cl-nagoya-ruri-base
export RAG_VECTOR_SIZE=768
export CHUNK_SIZE=300
export CHUNK_OVERLAP=50
export RAG_EMBED_PREFIX_QUERY='クエリ: '
export RAG_EMBED_PREFIX_PASSAGE='文章: '

# 1. Python 依存をインストール
python3 -m pip install -r rag_poc/requirements.txt

# 2. モデル取得とサービス生存確認
ollama pull "$EMBED_MODEL"
curl -fsS http://127.0.0.1:11434/api/tags
curl -fsS http://127.0.0.1:6333/collections

# 3. コーパスを埋め込みして Qdrant へ投入
python3 rag_poc/ingest.py

# 4. 質問する（--mode 省略時は hybrid）
python3 rag_poc/query.py '質問' --mode hybrid --json
```

設定の意味・全量への切り替え・Rust バイナリや評価の詳細は[セットアップ](#セットアップ)、
[使い方](#使い方) および[Ruri での再現手順](docs/explanation.md#ruri-での再現手順)を参照。

## 目的

`passkey-gateway/docs/` 配下の Markdown を知識源に、外部 API を使わず
ローカルの Ollama と Qdrant だけで RAG（検索拡張生成）を試す PoC。
埋め込みと回答生成を Ollama、ベクトル検索を Qdrant が担い、検索
モード（dense / keyword / hybrid / hybrid-rerank）の差を評価する。

既定はローカル Ollama + Qdrant で、クラウドは `RAG_CLOUD_ENABLED=1` の
opt-in（詳細は[クラウドモデル（opencodeGO）](#クラウドモデルopencodego)を参照）。

### 推奨クイックスタート

クラウド RAG 調査（`passkey-gateway/docs/cloud-rag-research-2026-11-26.md`、
リポジトリ外のためパス表記）に従い、日本語特化埋め込み
`kun432/cl-nagoya-ruri-base` を推奨する（実測値は
[代表的な実測値](docs/reference.md#代表的な実測値)）。**config 既定は後方互換のため
`qwen2.5:3b` のまま**で、未設定の `evaluate.py` は旧ベースラインを再現する。
Ruri を使う場合は[まず動かす（5分）](#まず動かす5分) /
[Ruri での再現手順](docs/explanation.md#ruri-での再現手順)を参照して設定を揃えること。

## 必要なもの

- Ollama がローカルで起動していること（既定 `http://127.0.0.1:11434`）
- Qdrant がローカルで起動していること（既定 `http://127.0.0.1:6333`）
- Python 3.10 以降
- `requests`（`ingest.py` / `query.py` / `evaluate.py` が使用）
- 埋め込みモデルは次のいずれか一方でよい:
  - **Ruri クイックスタート（推奨）**: `ollama pull kun432/cl-nagoya-ruri-base`。
    [まず動かす（5分）](#まず動かす5分)の設定を使えば **`qwen2.5:3b` は不要**。
  - 旧ベースライン再現: `ollama pull qwen2.5:3b`（config 既定。未設定の
    `evaluate.py` はこちらを再現する）。

## セットアップ

リポジトリルートで以下を実行する。

```bash
# requests をインストール
python3 -m pip install -r rag_poc/requirements.txt

# compare_embeddings 用の Rust バイナリをビルド（任意。未ビルドなら Python へフォールバック）
cargo build --manifest-path rag_poc/ragcore/Cargo.toml

# モデル取得（未取得の場合。Ruri 推奨、旧ベースラインは qwen2.5:3b）
ollama pull kun432/cl-nagoya-ruri-base
# ollama pull qwen2.5:3b  # 旧ベースラインを再現する場合のみ

# サービスの生存確認
curl -fsS http://127.0.0.1:11434/api/tags
curl -fsS http://127.0.0.1:6333/collections
```

`requirements.txt` の内容は `requests>=2.31,<3` のみ。`evaluate.py` は
`query.py` を import するため、`ingest.py` / `query.py` / `evaluate.py`
のいずれを使う場合も上のインストールが必要。未導入の場合は
`evaluate.py` が明示エラーで終了する。

Rust バイナリは `rag_poc/ragcore/target/debug/ragcore` に生成される。
op の配線状況（本番配線は `cosine_batch` と `distribution` の 2 op）、
同等性テスト件数、CI ゲート専用 `check_eval_results`、および
`RAG_RUST_BIN` / `RAG_RUST_TIMEOUT` / `RAG_REQUIRE_RUST` の詳細は
[Rust op の配線状況](docs/reference.md#rust-op-の配線状況)と
[設定（環境変数）](docs/reference.md#設定環境変数)を参照。

## 使い方

```bash
# 1. コーパスを埋め込みして Qdrant へ投入（初回・更新時）
python3 rag_poc/ingest.py

# 2. 質問する（--mode 省略時は hybrid）
python3 rag_poc/query.py '質問' --mode hybrid --json

# 3. 4 モードを比較し eval-results.json を出力
python3 rag_poc/evaluate.py

# 4. 全量コーパスを使う場合（投入と検索で同じ値を指定）
RAG_COLLECTION=rag_poc_full python3 rag_poc/ingest.py
RAG_COLLECTION=rag_poc_full python3 rag_poc/query.py '質問' --json

# 5. 軽量サブセット（6 ファイル／81 チャンク）で試す場合
RAG_COLLECTION=rag_poc_docs \
RAG_CORPUS_DIR=rag_poc/corpus-subset python3 rag_poc/ingest.py
RAG_COLLECTION=rag_poc_docs python3 rag_poc/query.py '質問' --json

# 6. 既存成果物を横断比較（API 呼び出しなし、キー不要）
python3 rag_poc/compare_cloud_local.py --output rag_poc/cloud-local-comparison.json
```

`compare_cloud_local.py` は既存の `eval-results*.json` だけを読み取り、
コーパス／チャンク数／クエリ数／指標定義を明示した比較表を標準出力へ出す
（新規 API 呼び出しなし）。`--filter NAME` で絞り込み、`--sort KEY`
（`ndcg` / `latency` / `hit`）で同一条件グループ内のみ並び替え、
`--output PATH` で出力先を指定する。`--run-ab` は
[A/B 比較の手順](#ab-比較の手順--run-ab)を参照。

`rag_poc/corpus-subset/` は評価用の 6 ファイル／81 チャンクのサブセット
（2026-11-26 時点）。成果物ごとのファイル数・チャンク数の違いは
[代表的な実測値](docs/reference.md#代表的な実測値)と
[成果物一覧](docs/reference.md#成果物一覧)を参照。比較前に
`eval-results.json` の `corpus_file_count` / `corpus_sha256` を確認すること。

`--mode` には `dense` / `keyword` / `hybrid` / `hybrid-rerank` を指定する。
`--top-k N` で取得候補数を変更できる（既定 `config.TOP_K = 10`）。

### `query.py --json` の hits スキーマ

`--json` 出力の対象キー（`mode` / `hits` / `answer` / `sources`）、`hits` の
全 9 キーと出現条件、consumer 側の既定値付き参照ルールは
[hits スキーマ](docs/reference.md#hits-スキーマ)を参照。

### CLI オプション

`ingest.py` / `evaluate.py` / `query.py` のオプション一覧、
`--recreate` と `--yes` の破壊的操作ゲート、`--collections` と `--output` の
併用不可は[CLI オプション](docs/reference.md#cli-オプション)を参照。

## クラウドモデル（opencodeGO）

ユーザー承認により、opencodeGO の OpenAI 互換 API を生成モデルとして使える
（`cloud_client.py`、**既定は無効**）。有効化するには次を設定する。

> **警告**: 有効化すると質問文・チャンク本文・ベクトルが外部 API へ送信され、
> token 課金が発生する。キーと接続先を確認し、長時間の連続評価は避けること。
> この注意は README 内では本章に集約。docs/explanation.md では ARID に従い要約を重複。

共有 `dsh.env` 全体を `set -a; source ...; set +a` で export すると他サービスのキーまで子プロセスへ継承されるリスクがあるため、必要なキーだけを明示的に読み込む。

```bash
# dsh.env から必要なキー（OPENCODE_GO_API_KEY）だけを読み込む（値は表示しない）
export OPENCODE_GO_API_KEY="$(grep '^OPENCODE_GO_API_KEY=' ~/.config/opencode/dsh.env | cut -d= -f2-)"

export RAG_CLOUD_ENABLED=1
export RAG_CLOUD_BASE_URL=https://opencode.ai/inference/go/openai/v1
export RAG_CLOUD_MODEL=deepseek-v4.1-flash
```

認証キーは `OPENCODE_GO_API_KEY`（正規。`dsh.env.example` と一致）を使う。
`OPENCODE_API_KEY` は後方互換名で、両方設定時は正規が優先される
（値はログ・例外・成果物へ出力しない）。

埋め込みモデルも利用できる（[公式 Go ドキュメント](https://opencode.ai/docs/console/go/)、
2026-11-26 取得）。モデル ID と 1M 入力トークン単価は
[埋め込み単価](docs/reference.md#埋め込み単価)を参照。

同一サブセット（6 ファイル／81 チャンク、`qrels-subset.json` 6 クエリ、
`ingest.iter_chunks` 既定 CHUNK_SIZE、repeat=1）で日本語埋め込みモデルを比較した。
単価は `qwen3-embedding-8b` の方が安いが、この日本語コーパスでは
`text-embedding-3-small` が精度・レイテンシともに上回った。
`gemini-embedding-2`（3072 次元）も計測した。計測条件と実測値は
[クラウド/ローカル比較](#クラウドローカル比較2026-11-26)および
[代表的な実測値](docs/reference.md#代表的な実測値)を参照。
エンドポイントは `https://opencode.ai/inference/go/openai/v1/embeddings` の
OpenAI 互換 `/embeddings` API で、`dimensions` 指定が可能。`encoding_format` は
本クライアント未対応。埋め込みリクエストに `x-opencode-session` は不要である。

```bash
export RAG_CLOUD_EMBED_MODEL=text-embedding-3-small
```

重要な制約: 認証情報は環境変数名のみを参照し値を出力しない。キーの有効性と
レート制限は未検証で、長時間の連続評価は避ける。Go 上限内の要求では API の
`usage.cost` が `0` になる（仕様）。実請求との照合には Console の Usage export
API を使うが、認証情報を扱うため本 PoC では未実施である。既定は埋め込みも生成も
ローカル Ollama で、クラウドを使う場合は `RAG_CLOUD_ENABLED=1` を明示する。

### コストと token 推定キーの使い分け

`eval-results-cloud-*.json` の `usage` / `summary` に含まれるコストキーの
意味と `cost_source=api-zero` の読み方は
[コストと token 推定キー](docs/reference.md#コストと-token-推定キー)を参照。

### 簡易計測と正式パイプラインの使い分け

検索経路・コーパス・成果物・生成オプションの違いは
[簡易計測と正式パイプラインの使い分け](docs/reference.md#簡易計測と正式パイプラインの使い分け)を参照。
`--with-generation` の既定は全件（`rerank_query_count=9`、n=9）。先頭 N 件に
絞る場合は `--generation-max-queries N` を指定する。

### クラウド生成で回答を作る（--with-answer）

`RAG_CLOUD_CHAT_ENABLED=1` は `RAG_CLOUD_ENABLED=1` が前提で、単独では
fail-closed で停止する。`--with-answer` と組み合わせると
`cloud_client.chat()` が回答を生成する（タイムアウト `RAG_ANSWER_TIMEOUT`＝60 秒）。
埋め込みもクラウドへ切り替わるため、`RAG_CLOUD_EMBED_MODEL` と整合する
`RAG_VECTOR_SIZE` も設定しておくこと。

```bash
# 前提: dsh.env のキー読み込みと RAG_CLOUD_EMBED_MODEL は上記設定を再利用
RAG_CLOUD_ENABLED=1 RAG_CLOUD_CHAT_ENABLED=1 \
  python3 rag_poc/evaluate.py --with-answer \
  --output rag_poc/eval-results-cloud-answer.json
```

生成結果は `answer` / `sources` / `answer_error` / `answer_latency_sec`
として記録される。失敗は `answer_error` へ分離され検索指標を汚さない。

### クラウド/ローカル比較（2026-11-26）

計測の前提のみをここに残す。実測値は
[代表的な実測値](docs/reference.md#代表的な実測値)を単一情報源とし、指標定義・
レイテンシ・repeat・コーパス条件の注記は
[比較表の指標定義と注記](docs/reference.md#比較表の指標定義と注記)を参照。

### A/B 比較の手順（`--run-ab`）

A/B 比較は**各コマンドを手順どおり実行し、比較は
`compare_cloud_local.py --run-ab` で行う**。外部送信と token 課金を伴うため、
実計測をまとめる一括スクリプトは作らず既存コマンドの組合せで再現する。

```bash
# 1. ローカル Ruri を計測（全量。環境変数は「まず動かす」を参照）
export RAG_CORPUS_DIR=passkey-gateway/docs
export RAG_COLLECTION=rag_poc_ruri_full
export EMBED_MODEL=kun432/cl-nagoya-ruri-base
export RAG_VECTOR_SIZE=768
export CHUNK_SIZE=300
export CHUNK_OVERLAP=50
export RAG_EMBED_PREFIX_QUERY='クエリ: '
export RAG_EMBED_PREFIX_PASSAGE='文章: '
python3 rag_poc/ingest.py
python3 rag_poc/evaluate.py --qrels rag_poc/qrels.json \
  --output rag_poc/eval-results-ruri-full.json

# 2. クラウドを計測（キー読み込みと opt-in は「クラウドモデル」章を参照）
export RAG_CLOUD_ENABLED=1
export RAG_CLOUD_EMBED_MODEL=text-embedding-3-small
python3 rag_poc/cloud_bench.py --corpus ../passkey-gateway/docs \
  --qrels rag_poc/qrels.json \
  --output rag_poc/eval-results-cloud-full.json

# 3. 既存成果物だけを読んで A/B 比較表を出力（API 呼び出しなし）
python3 rag_poc/compare_cloud_local.py --run-ab --sort ndcg
```

`--run-ab` は `eval-results*.json` を横断し、`eval-results-cloud-*` をクラウド側、
それ以外（`baseline` / `ruri-*` 等）をローカル側として条件別にグループ化する。
条件が異なる行には `[条件不一致]` フラグを付け、`--sort` は同一条件グループ内
だけで並び替える。新たな API 呼び出しも実計測も行わない。

## テスト

```bash
python3 rag_poc/test_rag.py

# Rust 未ビルドの skip を失敗に変える（CI の最終ゲート）
RAG_REQUIRE_RUST=1 python3 rag_poc/test_rag.py
```

外部サービスへ接続せず、`unittest` のみで設定・RRF・JSON 解析・チャンク分割・
評価クエリ・held-out 指標・rerank のフォールバックを検証する（全 270 件）。
Rust 同等性テスト 9+8件のうち、バイナリを要する 8 件は未ビルド時に skip される。
第2のコマンドは skip を失敗へ切り替える CI ゲートである（`binary_path_default`
や `RustSkipGateTest` の 3 件など、未ビルドでも skip しないテストもある）。
ディレクトリ名にハイフンがあるため `python3 -m unittest` のディスカバリは使えず、
直接実行する。

## 詳細ドキュメント

- [docs/index.md](docs/index.md) — 詳細ドキュメントの索引（用途別の入口）
- [docs/reference.md](docs/reference.md) — 検索モード、全設定項目（35変数）、実測値の単一情報源、成果物一覧
- [docs/explanation.md](docs/explanation.md) — 構成図、設計判断（なぜハイブリッド／Ruri／クラウドか）、既知の制限、モデル比較実験、別ドメインベンチマーク、入力と設定の検証
- [ragcore/README.md](ragcore/README.md) — Rust 純粋計算クレートの CLI 仕様
- [CONTRIBUTING.md](CONTRIBUTING.md) — 検証ループとコミット規約
- [SECURITY.md](SECURITY.md) — 脆弱性の報告先と対応方針
- ドキュメント設計の判断根拠: `passkey-gateway/docs/documentation-references-extra-2026-11-26.md`

実測値は [docs/reference.md の代表的な実測値](docs/reference.md#代表的な実測値)を
単一情報源とする。日付表記は本文 `YYYY-MM-DD`、メタデータは RFC 3339、
ファイル名は `YYYYMMDD`。
