# rag_poc — ローカル RAG 検証

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

> **TL;DR / まず読む順**: [まず動かす（5分）](#まず動かす5分) → [目的](#目的) →
> [セットアップ](#セットアップ) → [使い方](#使い方) → [テスト](#テスト)。
> 背景・設計判断は [docs/explanation.md](docs/explanation.md)、全設定と実測値・成果物一覧は
> [docs/reference.md](docs/reference.md) を参照（この README は入口のみ）。実測値は
> [docs/reference.md の代表的な実測値](docs/reference.md#代表的な実測値)を単一情報源とする。
> 日付表記: 本文は `YYYY-MM-DD`、メタデータは RFC 3339、ファイル名は `YYYYMMDD`。

## 目的

`passkey-gateway/docs/` 配下の Markdown を知識源に、外部 API を使わず
ローカルの Ollama と Qdrant だけで RAG（検索拡張生成）を試す PoC。
埋め込みと回答生成を Ollama、ベクトル検索を Qdrant が担い、検索
モード（dense / keyword / hybrid / hybrid-rerank）の差を評価する。

既定はローカル Ollama + Qdrant で、クラウドは `RAG_CLOUD_ENABLED=1` の
opt-in（詳細は[クラウドモデル（opencodeGO）](#クラウドモデルopencodego)を参照）。

### 推奨クイックスタート

クラウド RAG 調査の結論（レポート `passkey-gateway/docs/cloud-rag-research-2026-11-26.md`、
リポジトリ外のため相対リンクではなくパス表記）に従い、
日本語特化埋め込み `kun432/cl-nagoya-ruri-base` の利用を推奨する。
実測値は [docs/reference.md の代表的な実測値](docs/reference.md#代表的な実測値)を参照。
ただし **config 既定は後方互換のため `qwen2.5:3b` のまま**で、何も設定せず
`evaluate.py` を走らせると旧ベースラインを再現する。Ruri を使う場合は
[まず動かす（5分）](#まず動かす5分) または
[Ruri での再現手順](docs/explanation.md#ruri-での再現手順)を参照して
`EMBED_MODEL` / `RAG_VECTOR_SIZE=768` / `CHUNK_SIZE=300` / `CHUNK_OVERLAP=50` /
prefix を揃えること。

## 必要なもの

- Ollama がローカルで起動していること（既定 `http://127.0.0.1:11434`）
- Qdrant がローカルで起動していること（既定 `http://127.0.0.1:6333`）
- Ollama に `qwen2.5:3b` が投入済みであること
  - 未取得なら `ollama pull qwen2.5:3b`
- Python 3.10 以降
- `requests`（`ingest.py` / `query.py` / `evaluate.py` が使用）

## セットアップ

リポジトリルートで以下を実行する。

```bash
# requests をインストール
python3 -m pip install -r rag_poc/requirements.txt

# compare_embeddings 用の Rust バイナリをビルド（任意。未ビルドなら Python へフォールバック）
cargo build --manifest-path rag_poc/ragcore/Cargo.toml

# モデル取得（未取得の場合）
ollama pull qwen2.5:3b

# サービスの生存確認
curl -fsS http://127.0.0.1:11434/api/tags
curl -fsS http://127.0.0.1:6333/collections
```

`requirements.txt` の内容は `requests>=2.31,<3` のみ。`evaluate.py` は
`query.py` を import するため、`ingest.py` / `query.py` / `evaluate.py`
のいずれを使う場合も上のインストールが必要。未導入の場合は
`evaluate.py` が明示エラーで終了する。

Rust バイナリは `rag_poc/ragcore/target/debug/ragcore` に生成される。
`ragcore` の op は 9 件（`chunk` / `cosine` / `cosine_batch` /
`distribution` / `recall` / `ndcg` / `latency` / `fingerprint` /
`point_id`）で、本番配線は `cosine_batch` と `distribution` の 2 op だけである。
`compare_embeddings.py` が `rust_bridge.py` 経由で呼び、`cosine_batch` は
質問×ファイルのコサイン類似度を 1 回で計算し、`distribution` はスコアの
min/max/mean/stdev を求める。Rust が無い／失敗した場合は Python 実装へ
フォールバックする（フォールバック発生は出力 JSON の `notes` に記録される）。
`cosine` は実計算には使わず、`rust_bridge.available()` の probe
（`a=[1.0]` / `b=[1.0]` の応答確認）に限られる。

残り 6 op（`chunk` / `point_id` / `fingerprint` / `recall` / `ndcg` /
`latency`）は未配線である。`ingest.py` のチャンク分割と point ID 生成、
`evaluate.py` の fingerprint / recall / nDCG / レイテンシ集計は純 Python の
ままで、Rust 実装は同等性テスト（`test_rag.py`）でのみ使われる。つまり
「Rust 再実装」は完了しておらず、本番配線は `compare_embeddings.py` の
2 op に限定される。

`check_eval_results` は上記 9 op に含めない、CI ゲート専用の別 op である。
`python3 rag_poc/rust_bridge.py check_eval_results` が
`eval-results-*.json` のパスだけを Rust へ渡し、成果物のキー検証をゲートする
（Python の検索・評価経路では呼ばない）。
Rust 同等性テストは `ragcore` 側 9 件と Python 側 8 件の計 17 件
（9+8件）で、`RAG_REQUIRE_RUST=1` を付けると skip ではなく失敗にできる。

ビルド済みバイナリを別の場所へ置く場合は `RAG_RUST_BIN` でパスを
差し替えられる（相対パスはリポジトリルート基準）。Rust CLI 呼び出しの
タイムアウトは `RAG_RUST_TIMEOUT`（既定 30 秒）で変更する。

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
コーパス／チャンク数／クエリ数／指標定義を明示した比較表を標準出力へ出す。
新たな API 呼び出しは行わない。`--filter NAME` で比較行を名前で絞り込み、
`--sort KEY`（`ndcg` / `latency` / `hit`）で同一条件グループ内のみの
並び替え、`--output PATH` で比較 JSON の出力先を指定する（省略時は stdout）。
`--run-ab` を付けるとローカル／クラウドを条件別にグループ化した A/B 比較表を
出力する（詳細は[A/B 比較の手順](#ab-比較の手順--run-ab)）。

`rag_poc/corpus-subset/` は評価用に抜き出した 6 ファイルのサブセット
（2026-11-26 時点）。全量は既定の `passkey-gateway/docs` で、
成果物ごとのファイル数は異なる。ローカル qwen / Ruri の全量成果物は
88 ファイル／2200 チャンク、クラウド全量成果物は 91 ファイル／1380 チャンク、
Qdrant 正式パイプライン成果物は 88 ファイル／214 チャンク
（旧比較表の 87 ファイル／214 チャンクは旧値）。`passkey-gateway/docs` の
実体は 91 ファイルで、成果物ごとにコーパスのスナップショットが異なる。
簡易評価の初期値はサブセットで実施した履歴があるため、
現在の `eval-results.json` の `corpus_file_count` / `corpus_sha256` を
確認してから比較すること。

`--mode` には `dense` / `keyword` / `hybrid` / `hybrid-rerank` を指定する。
`--top-k N` で取得候補数を変更できる（既定 `config.TOP_K = 10`）。

### `query.py --json` の hits スキーマ

`--json` を付けたときの出力対象は `mode` / `hits` / `answer` / `sources` で、
`hits` の各要素は次のキーを持つ。全モードで必ず現れる共通キーと、
モードや実行状態によって欠落し得る optional キーを分けて記載する。

| キー | 種別 | 内容 | 現れる条件 |
| --- | --- | --- | --- |
| `id` | 共通 | Qdrant の point ID | 全モード |
| `score` | 共通 | 検索スコア。`dense` はコサイン類似度、`keyword` は一致語数、`hybrid` 以降は RRF スコア。`hybrid-rerank` では `final_score` と同値 | 全モード |
| `source` | 共通 | 出典（ファイル名など。取得できない場合は `point:<id>`） | 全モード |
| `text` | 共通 | チャンク本文（空文字の可能性あり） | 全モード |
| `reranked_by_llm` | optional | LLM 再ランクが適用されたか（`true` / `false`） | `hybrid-rerank` のみ |
| `base_score` | optional | 再ランク前の元スコア | `hybrid-rerank` のみ |
| `llm_score` | optional | LLM が返した 0..1 のスコア（欠損時は `null`） | `hybrid-rerank` のみ |
| `final_score` | optional | 並び替え用の 0..1 正規化スコア | `hybrid-rerank` のみ |
| `truncated` | optional | scroll が `MAX_SCROLL_PAGES` に到達し結果が切れたことを示す `true` | `keyword` / `hybrid` / `hybrid-rerank` で keyword 側 scroll が上限到達時のみ |

consumer は optional キーの存在を前提にしないこと。参照時は
`hit.get("reranked_by_llm")` のように既定値を与えるか、`"key" in hit` で
有無を確認する。また共通キーも `id` / `score` はソース次第で `null`
になり得るため、型を決め打ちしないこと。

### CLI オプション

`ingest.py`:

| オプション | 動作 |
| --- | --- |
| `--help` | ヘルプを表示して終了（設定検証より前に処理される） |
| `--corpus-dir DIR` | コーパスディレクトリ（リポジトリ配下。既定 `RAG_CORPUS_DIR`） |
| `--collection NAME` | Qdrant コレクション名（既定 `RAG_COLLECTION`） |
| `--recreate` | コレクションを削除して再作成。**`--yes` を同時指定しない限り削除せず終了コード 2 で停止する**（破壊的操作のため実 DELETE は明示承認が必要） |
| `--yes` | `--recreate` と併用してコレクション削除（実 DELETE）を実行する。単独指定では削除しない |

`evaluate.py`:

| オプション | 動作 |
| --- | --- |
| `--qrels PATH` | 正解ラベルファイル（3 種類の使い分けは後述） |
| `--heldout-only` | 簡易指標を既存 JSON から再利用し held-out のみ再計測 |
| `--collections a,b,c` | 複数コレクションを一括評価し `eval-results-<collection>.json` へ個別保存。**`--output` と併用不可（終了コード 2）** |
| `--with-answer` | 回答生成を実行し `answer` / `sources` / `answer_error` / `answer_latency_sec` を記録（既定は検索指標のみ） |
| `--output PATH` | 指定パスへ書き出す。未指定時は上書き前に `eval-results-<UTC>.json` へ自動退避 |

`query.py` は `--mode` / `--top-k` / `--json` に加え、`--help` が設定不正時も
先に表示される。生成失敗は検索指標を汚さず `answer_error` へ分離される。

`RAG_COLLECTION` は参照する Qdrant コーパスを切り替える。サブセットは
`rag_poc_docs`（既定）、全量は `rag_poc_full`。投入時と検索・評価時で
同じ値を指定すること。`RAG_RERANK_MAX_DOCS` は LLM に再ランクさせる件数
（既定 5）。上位 5 件のみを再評価し、残りは元のスコアを保持するため、
返却件数は `--top-k` のまま変わらない。

## クラウドモデル（opencodeGO）

ユーザー承認により、opencodeGO の OpenAI 互換 API を生成モデルとして
使える。実装は `cloud_client.py` にあり、**既定は無効**（ローカル Ollama
を維持）。有効化するには次を設定する。

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
`OPENCODE_API_KEY` は後方互換の互換名として併記され、両方が設定されて
いる場合は `OPENCODE_GO_API_KEY` が優先される（値はログ・例外・成果物へ
出力しない）。

埋め込みモデルも利用できる（[公式 Go ドキュメント](https://opencode.ai/docs/console/go/)、
2026-11-26 取得）:

| モデル ID | 1M 入力トークン単価 |
| --- | --- |
| `text-embedding-3-small` | $0.02 |
| `qwen3-embedding-8b` | $0.01 |
| `gemini-embedding-2` | $0.20 |

同一サブセット（6 ファイル／81 チャンク、`qrels-subset.json` 6 クエリ、
`ingest.iter_chunks` 既定 CHUNK_SIZE、repeat=1）で日本語埋め込みモデルを
比較した。計測の前提は[クラウド/ローカル比較](#クラウドローカル比較2026-11-26)へ、
実測値は [docs/reference.md](docs/reference.md#代表的な実測値)へ集約した。

単価は `qwen3-embedding-8b` の方が安いが、この日本語コーパスでは
`text-embedding-3-small` が精度・レイテンシともに上回った
（`eval-results-cloud-small-subset.json` /
`eval-results-cloud-qwen3-subset.json`）。
`gemini-embedding-2`（3072 次元、1M 入力トークン $0.20）も同一サブセットで
計測した（`eval-results-cloud-gemini-subset.json`）。実測値は
[docs/reference.md](docs/reference.md#代表的な実測値)を参照。

エンドポイントは `https://opencode.ai/inference/go/openai/v1/embeddings` で、
OpenAI 互換の `/embeddings` API。`dimensions` は指定可能。
`encoding_format` は API 仕様上存在するが本クライアント未対応
（リクエストへ渡さない）。埋め込みリクエストには `x-opencode-session` が
不要で、RAG ライブラリから直接呼べる（生成/chat 呼び出しには会話単位
UUID ヘッダを付与する）。

```bash
export RAG_CLOUD_EMBED_MODEL=text-embedding-3-small
```

重要な制約:

- 認証情報は環境変数名のみを参照し、値はログ・例外・成果物へ出力しない。
- キーの有効性・レート制限・コストは未検証。長時間の連続評価は避ける
  （外部送信＋token 課金の警告は本章冒頭に集約済み）。
- 既定は埋め込みも生成もローカル Ollama。クラウドを使う場合は
  `RAG_CLOUD_ENABLED=1` を明示する。

### コストと token 推定キーの使い分け

`cloud_bench.py` が出力する `eval-results-cloud-*.json` には、API 応答の
`usage` を集約した次のキーが入る（根拠: `cloud_bench.py` の
`_usage_cost_source` / `_estimated_cost_usd` / `_token_cost_usd`）。

| キー | 値・意味 | 読み方 |
| --- | --- | --- |
| `usage.cost_source` | `api` / `api-zero` / `estimated` / `None` | コストの出所。`api` は正の API cost、`api-zero` は `cost=0.0` かつ `total_tokens>0`、`estimated` は cost キーが無い単価推定、`None` は判定不能 |
| `summary.estimated_cost_usd` | 小数または `None` | API cost を 6 桁で保持。cost が無ければ単価 × total_tokens の 6 桁推定 |
| `summary.estimated_cost_from_tokens_usd` | 小数（12 桁） | 6 桁丸めで 0.0 に潰れる場合のトークン由来見込み。`usage` 側にも同じ値を記録 |

`cost_source=api-zero` のとき、`summary.estimated_cost_usd` は API 応答の
`cost=0.0` を維持する。**API が 0.0 を返した理由は未確認**であり、
実コストがゼロだと断定しない。このときは
`estimated_cost_from_tokens_usd` を主指標として読み、成果物 `notes` に
「cost_source=api-zero のときは token 推定を参照」を明記する。
単価が未知のモデルの場合は `estimated_cost_from_tokens_usd` を作らない。

### 簡易計測と正式パイプラインの使い分け

| 観点 | 簡易: `cloud_bench.py` | 正式: `cloud_ingest.py` → `evaluate.py` |
| --- | --- | --- |
| 検索 | 埋め込みコサイン類似度のみ（Qdrant を使わない） | Qdrant 投入後に `dense` / `keyword` / `hybrid` / `hybrid-rerank` の 4 モード |
| コーパス | 既定 `corpus-subset`（`--corpus` で変更可） | `RAG_CORPUS_DIR` / `RAG_COLLECTION`（全量は `rag_poc_cloud_full`） |
| 成果物 | `eval-results-cloud-*.json`（`schema_version` なし） | `eval-results-cloud-pipeline.json`（`schema_version` あり） |
| 生成 | `--with-generation` で rerank / 回答（既定は全件、`--generation-max-queries N` で先頭 N 件に制限） | `--with-answer` で回答生成（`RAG_ANSWER_TIMEOUT` が適用） |
| 用途 | 埋め込みモデルの簡易比較。Token 消費を抑えて確認する | 本番相当の 4 モード評価と退避済み成果物の更新 |

`--with-generation` の既定は全件で、全量コーパスでは 9 クエリ
（`rerank_query_count=9`、n=9）を計測した。実測値は
[docs/reference.md](docs/reference.md#代表的な実測値)を参照。先頭 N 件だけに
絞る場合は `--generation-max-queries N` を指定する。

### クラウド生成で回答を作る（--with-answer）

`RAG_CLOUD_CHAT_ENABLED=1` は `RAG_CLOUD_ENABLED=1` が前提で、単独では
設定検証（fail-closed）で停止する。`evaluate.py` の `--with-answer` と
組み合わせると `query.generate_answer()` の LLM 呼び出しが
`cloud_client.chat()` へ差し替わり、クラウド生成モデルが回答を生成する
（タイムアウトは `RAG_ANSWER_TIMEOUT`＝60 秒）。埋め込みもクラウドへ
切り替わるため、上記の `RAG_CLOUD_EMBED_MODEL` と整合する
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

計測の前提のみをここに残す。各構成の実測値は[docs/reference.md の代表的な実測値](docs/reference.md#代表的な実測値)を単一情報源とする。

- `hit@1` は query-level、`set_recall@1` は集合 recall。ローカルの `recall@1` は `hit@1` に相当する。
- 指標定義は3種。`evaluate query-level`（`evaluate.py`: query-level hit@1 と集合 set_recall@1、nDCG@10 は上位10件・ファイル1回）、`cloud_bench top10`（`cloud_bench.py`: nDCG@10 は上位10件・ファイル1回）、`legacy 全順位`（旧 `ndcg_at_10`、全順位ベースで現行定義と直接比較不可）。`—` は未計測を表す。
- レイテンシは計測対象が異なる（クラウドは埋め込み平均、ローカルは `hybrid-rerank` の検索平均）ため単純比較しないこと。
- 計測回数はクラウド `repeat=1`、ローカル全量 `RAG_EVAL_REPEAT=10`（ウォームアップ 1 回＋10 回計測＝60 サンプル）で混在している。rerank nDCG@5 は 9 クエリ（`rerank_query_count=9`、n=9）の計測値。
- 全量ファイル数は成果物ごとに異なる。`passkey-gateway/docs` の実体は 91 ファイル、`eval-results-cloud-pipeline.json` は 88 ファイル／214 チャンクで、旧比較表の「87 ファイル／214 チャンク」は旧値。
- 全量クラウドは 91 ファイル／1380 チャンク（`ingest.iter_chunks` 委譲後）、全量 Ruri は 88 ファイル／2200 チャンクでコーパスとチャンク条件が異なるため、両者の nDCG に優劣は付けられない。

### A/B 比較の手順（`--run-ab`）

A/B 比較は**各コマンドを手順どおり実行し、比較は
`compare_cloud_local.py --run-ab` で行う**。実計測をまとめて呼び出す
一括スクリプト（ingest → evaluate → cloud_bench を順に呼ぶもの）は
作らない。外部送信と token 課金を伴ううえ、既存コマンドの組合せで
十分に再現できるためである（車輪の再発明を避ける）。

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

`--run-ab` は `eval-results*.json` を横断し、`eval-results-cloud-*` を
クラウド側、それ以外（`baseline` / `ruri-*` 等）をローカル側として
条件別にグループ化する。コーパス／チャンク数／クエリ数／repeat／
指標定義が反対側と異なる行には `[条件不一致]` フラグを付け、直接
比較できないことを明示する。並び替えは `--sort` を併用したときに
同一条件の行グループ内だけで行われ、現行 `--sort` のグループ化を
再利用する。新たな API 呼び出しも実計測も行わない。

## テスト

```bash
python3 rag_poc/test_rag.py

# Rust 未ビルドの skip を失敗に変える（CI の最終ゲート）
RAG_REQUIRE_RUST=1 python3 rag_poc/test_rag.py
```

外部サービスへ接続せず、`unittest` のみで設定・RRF・JSON 解析・
チャンク分割・評価クエリ・held-out 指標・rerank のフォールバックを
検証する（全 230 件）。Rust 同等性テスト 9+8件（`ragcore` 側 9 件と
Python 側 8 件）のうち、Rust バイナリを要する 8 件は未ビルド時に
理由付きで skip される。第2のコマンドはこの skip を失敗へ切り替え、
Rust 未ビルドを green で通さないためのゲートである。なお
`binary_path_default` や `RustSkipGateTest` の 3 件など、バイナリが
未ビルドでも skip しない Rust 関連テストもある。
ディレクトリ名にハイフンがあるため `python3 -m unittest` の
ディスカバリは使えず、直接実行する。

## 詳細ドキュメント

- [docs/reference.md](docs/reference.md) — 検索モード、全設定項目（35変数）、実測値の単一情報源、成果物一覧
- [docs/explanation.md](docs/explanation.md) — 構成図、設計判断（なぜハイブリッド／Ruri／クラウドか）、既知の制限、モデル比較実験、別ドメインベンチマーク、入力と設定の検証
- [ragcore/README.md](ragcore/README.md) — Rust 純粋計算クレートの CLI 仕様
- ドキュメント設計の判断根拠: `passkey-gateway/docs/documentation-references-extra-2026-11-26.md`
