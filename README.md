# rag_poc — ローカル RAG 検証

> この README は入口（how-to）です。背景は `docs/explanation`、仕様は `docs/reference` へ分離する。
> 日付表記: 本文は `YYYY-MM-DD`、メタデータは RFC 3339、ファイル名は `YYYYMMDD`。

## 目的 [explanation]

`passkey-gateway/docs/` 配下の Markdown を知識源に、外部 API を使わず
ローカルの Ollama と Qdrant だけで RAG（検索拡張生成）を試す PoC。
埋め込みと回答生成を Ollama、ベクトル検索を Qdrant が担い、検索
モード（dense / keyword / hybrid / hybrid-rerank）の差を評価する。

既定はローカル Ollama + Qdrant で、クラウドは `RAG_CLOUD_ENABLED=1` の
opt-in（詳細は[クラウドモデル（opencodeGO）](#クラウドモデルopencodego-how-to)を参照）。

### 推奨クイックスタート [tutorial]

クラウド RAG 調査の結論（レポート `passkey-gateway/docs/cloud-rag-research-2026-11-26.md`、
リポジトリ外のため相対リンクではなくパス表記）に従い、
日本語特化埋め込み `kun432/cl-nagoya-ruri-base` の利用を推奨する。
同一条件下で dense の `recall@10` / `nDCG@10` がともに 1.0 / 0.7992 を
達成している（qwen2.5:3b は両方 0.0）。ただし **config 既定は後方互換のため
`qwen2.5:3b` のまま**で、何も設定せず `evaluate.py` を走らせると旧ベースライン
を再現する。Ruri を使う場合は[Ruri での再現手順](#ruri-での再現手順-how-to)を参照して
`EMBED_MODEL` / `RAG_VECTOR_SIZE=768` / `CHUNK_SIZE=300` / prefix を揃えること。

## 構成図（テキスト） [reference]

```text
Markdown (passkey-gateway/docs/*.md)
        │
        ▼
  ingest.py ── チャンク分割 ──> チャンク
        │
        ├── 埋め込み要求 ──> ollama_client.py ──> Ollama /api/embed
        │                    (prefix / 次元検証 / Retry /
        │                     /api/embeddings フォールバック)
        │<───────── ベクトル ─────────────────────────────┘
        ▼
  Qdrant コレクション rag_poc_docs (Cosine / 2048 次元)
        │
        ▼
  query.py ── 検索 ──> dense / keyword / hybrid
        │                  / hybrid-rerank
        ├── 再ランク / 回答生成 ──> Ollama /api/chat
        ▼
  evaluate.py ── 比較 ──> rag_poc/eval-results.json
```

`config.py` が接続先・モデル・チャンク設定を一元管理する。
`ollama_client.py` は埋め込み呼び出しを共通化し、`ingest.py` / `query.py` /
`compare_embeddings.py` がこれを利用する（prefix・次元検証・リトライ・
旧 API フォールバックを一箇所で管理）。各値は環境変数で上書きできる。

## 必要なもの [reference]

- Ollama がローカルで起動していること（既定 `http://127.0.0.1:11434`）
- Qdrant がローカルで起動していること（既定 `http://127.0.0.1:6333`）
- Ollama に `qwen2.5:3b` が投入済みであること
  - 未取得なら `ollama pull qwen2.5:3b`
- Python 3.10 以降
- `requests`（`ingest.py` / `query.py` / `evaluate.py` が使用）

## セットアップ [tutorial]

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
ただし Rust bridge が Python 本番コードで使われるのは
`compare_embeddings.py` の一部だけである。具体的には `cosine_batch`
（質問×ファイルのコサイン類似度を 1 回で計算）と `distribution`
（スコアの min/max/mean/stdev）を `rust_bridge.py` 経由で呼び、Rust が
無い／失敗した場合は Python 実装へフォールバックする（フォールバック
発生は出力 JSON の `notes` に記録される）。

`ragcore` の他の op（`chunk` / `point_id` / `fingerprint` / `recall` /
`ndcg` / `latency`）は Python 本番コードへ配線されていない。`ingest.py` の
チャンク分割と point ID 生成、`evaluate.py` の fingerprint / recall /
nDCG / レイテンシ集計は純 Python のままで、Rust 実装は同等性テスト
（`test_rag.py`）でのみ使われる。つまり「Rust 再実装」は完了しておらず、
本番配線は `compare_embeddings.py` の一部に限定される。
Rust 同等性テストは `ragcore` 側 9 件と Python 側 8 件の計 17 件
（9+8件）で、`RAG_REQUIRE_RUST=1` を付けると skip ではなく失敗にできる。

ビルド済みバイナリを別の場所へ置く場合は `RAG_RUST_BIN` でパスを
差し替えられる（相対パスはリポジトリルート基準）。Rust CLI 呼び出しの
タイムアウトは `RAG_RUST_TIMEOUT`（既定 30 秒）で変更する。

## 使い方 [how-to]

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
`--sort KEY` で指定キー（例: `name` / `ndcg_at_10` / `latency_sec`）による
並び替え、`--output PATH` で比較 JSON の出力先を指定する（省略時は stdout）。

`rag_poc/corpus-subset/` は評価用に抜き出した 6 ファイルのサブセット
（2026-11-26 時点）。全量は既定の `passkey-gateway/docs`
（88 ファイル／2200 チャンク）。簡易評価の初期値はサブセットで実施した
履歴があるため、
現在の `eval-results.json` の `corpus_file_count` / `corpus_sha256` を
確認してから比較すること。

`--mode` には `dense` / `keyword` / `hybrid` / `hybrid-rerank` を指定する。
`--top-k N` で取得候補数を変更できる（既定 `config.TOP_K = 10`）。

### `query.py --json` の hits スキーマ [reference]

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

### CLI オプション [reference]

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

## 検索モード [reference]

| モード | 動作 |
| --- | --- |
| `dense` | 質問を埋め込み、Qdrant でコサイン類似度の意味検索 |
| `keyword` | 全ポイントを scroll し、語の中包含一致でスコア化 |
| `hybrid` | dense と keyword の順位を RRF (k=60) で統合 |
| `hybrid-rerank` | hybrid の上位 5 件を LLM が再ランクし全件を返す |

## 設定（環境変数） [reference]

`config.py` の既定値は次のとおり。必要なら環境変数で上書きする。

| 環境変数 | 既定値 |
| --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `QDRANT_URL` | `http://127.0.0.1:6333` |
| `EMBED_MODEL` / `CHAT_MODEL` | `qwen2.5:3b` |
| `RAG_COLLECTION` | `rag_poc_docs`（全量は `rag_poc_full`） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` |
| `RAG_TOP_K` | `10` |
| `RAG_RERANK_MAX_DOCS` | `5` |
| `RAG_VECTOR_SIZE` | `2048` |
| `OLLAMA_TIMEOUT` / `QDRANT_TIMEOUT` | `300 秒` / `300 秒` |
| `RAG_INGEST_RESUME` | `0`（1 でチェックポイント再開） |
| `RAG_EVAL_REPEAT` | `1`（ウォームアップなし 1 回。2 以上でウォームアップ 1 回＋N 回） |
| `RAG_CORPUS_DIR` | `passkey-gateway/docs` |
| `RAG_ALLOW_REMOTE_ENDPOINTS` | `0`（既定でリモート拒否。`1` でオプトイン） |
| `RAG_ALLOWED_ENDPOINT_HOSTS` | 空（カンマ区切りの許可ホスト。`host:port` 形式も可） |
| `RAG_QUESTION_MAX_CHARS` | `2000` |
| `RAG_EMBED_PREFIX_QUERY` | `""`（空。ruri は config テーブルで「クエリ: 」） |
| `RAG_EMBED_PREFIX_PASSAGE` | `""`（空。ruri は config テーブルで「文章: 」） |
| `RAG_RUST_BIN` | `rag_poc/ragcore/target/debug/ragcore` |
| `RAG_RUST_TIMEOUT` | `30 秒` |
| `RAG_ANSWER_TIMEOUT` | `60 秒`（回答生成 1 回のタイムアウト） |
| `RAG_CLOUD_TIMEOUT` | `60 秒`（クラウド HTTP タイムアウト） |
| `RAG_CLOUD_ENABLED` | `0`（1 でクラウド有効。既定はローカル Ollama） |
| `RAG_CLOUD_CHAT_ENABLED` | `0`（1 で生成もクラウド。`RAG_CLOUD_ENABLED=1` が前提） |
| `RAG_CLOUD_EMBED_MODEL` | 空（有効時は明示指定が必要。既定モデルは `text-embedding-3-small`） |
| `RAG_CLOUD_EMBED_DIMENSION` | 空（正規。未知モデルの期待次元を明示。未指定なら fail-closed。`RAG_CLOUD_DIMENSIONS` より優先） |
| `RAG_CLOUD_DIMENSIONS` | 空（移行期互換の複数形。正規が未設定のときだけ読む） |
| `RAG_CLOUD_BASE_URL` | `https://opencode.ai/inference/go/openai/v1` |
| `RAG_CLOUD_MODEL` | `deepseek-v4.1-flash` |

リモート endpoint は **オプトイン＋許可ホスト＋https の 3 条件をすべて
満たす場合だけ**致命問題が警告へ格下げされる。平文 HTTP の外部ホスト、
許可リストに無いホスト、不正 scheme / port はオプトインしても拒否される
（`config.validate_endpoint()` が権威）。設定例:
`RAG_ALLOW_REMOTE_ENDPOINTS=1 RAG_ALLOWED_ENDPOINT_HOSTS=ollama.example.test` ＋
`OLLAMA_URL=https://ollama.example.test:11434`。
`RAG_QUESTION_MAX_CHARS` は CLI と `search()` の両方で適用され、超過すると
0 ヒット＋エラー記録になる。`RAG_EMBED_PREFIX_QUERY` /
`RAG_EMBED_PREFIX_PASSAGE` が非空なら config のモデル別 prefix より
優先される。`RAG_RUST_BIN` の相対パスはリポジトリルート基準で解決し、
`RAG_RUST_TIMEOUT` は Rust CLI 専用（`OLLAMA_TIMEOUT` は流用しない）。
クラウド側の補助変数として、未知の埋め込みモデルには
`RAG_CLOUD_EMBED_DIMENSION`（期待次元を明示。未指定なら fail-closed）、
HTTP タイムアウトに `RAG_CLOUD_TIMEOUT`（既定 60 秒）、
`cloud_bench.py` のコーパスハッシュ上書きに `RAG_CLOUD_CORPUS_SHA256` がある。
埋め込み次元は単数形 `RAG_CLOUD_EMBED_DIMENSION` が正規で、複数形
`RAG_CLOUD_DIMENSIONS` は移行期互換。両方が設定されていれば単数形を
優先し、値は出さずに環境変数名だけを警告する。

`RAG_RERANK_TOP_N` は廃止した。実効値は `RAG_RERANK_MAX_DOCS`。
`eval-results.json` には `embed_model` / `chat_model` / `vector_size` /
`corpus_file_count` / `corpus_sha256` / `repeat` を記録する。

`RAG_VECTOR_SIZE` の実効値には優先順位がある。`EMBED_MODEL` が
`config._EXPECTED_VECTOR_SIZE_TABLE` に登録済みの既知モデル
（例: `kun432/cl-nagoya-ruri-base` は 768、`qwen2.5:3b` は 2048）では
テーブル値が優先され、環境変数の不一致は `ingest.py` の警告どまり
（期待次元を採用）となる。未知モデルの場合だけ `RAG_VECTOR_SIZE` が
実効値になる。ただし `query.py` の設定検証は ruri に 768 を要求するため、
ruri を使う場合は環境変数も 768 に揃えること（後述 [Ruri での再現手順](#ruri-での再現手順-how-to)）。

## 評価 [how-to]

`python3 rag_poc/evaluate.py` は簡易指標と held-out 評価を実行し、
`rag_poc/eval-results.json` に出力する。`--heldout-only` を付けると
held-out のみを再計測し、簡易指標は既存 JSON の `results` を再利用する。
既存 JSON が無い場合は簡易評価も実行する。

held-out は 9 クエリに正解ファイル名ラベルを付け、次をモード別に集計する。

- `recall@1/3/5/10`: 上位 k 件の `source` に正解が 1 件でもあれば 1、
  なければ 0 をクエリ平均（query-level hit@k）。正解ファイル全体の
  回収割合を測る集合 recall ではない。
- `nDCG@3/10`: gain=1 / discount=log2(rank+1) の DCG を IDCG で割り、
  クエリ平均する。同一ファイルの複数チャンクは最上位の 1 回だけ計上。

正解は `passkey-gateway/docs` 内の実在ファイル名。完全一致・相対パス
suffix・basename で照合し、検索失敗ケースは 0 として分母に残す。
recall@10 / nDCG@10 のため取得件数は最大 10 まで引き上げる。

### qrels ファイルの使い分け [reference]

評価ラベルは 3 種類の qrels ファイルで切り替える。`evaluate.py` の
`--qrels` 省略時は `rag_poc/qrels.json`。投入時と評価時で
`RAG_CORPUS_DIR` / `RAG_COLLECTION` を揃えること。

| qrels ファイル | コーパス（`RAG_CORPUS_DIR`） | コレクション（`RAG_COLLECTION`） | クエリ数 |
| --- | --- | --- | --- |
| `rag_poc/qrels.json` | `passkey-gateway/docs` | `rag_poc_full` | 簡易 6 + held-out 9 |
| `rag_poc/qrels-subset.json` | `rag_poc/corpus-subset` | `rag_poc_docs` | 簡易 6 + held-out 6 |
| `rag_poc/qrels-alt.json` | `rag_poc/corpus-alt` | `rag_poc_other_domain` | 簡易 6 + held-out 6 |

```bash
# qrels.json（既定。全量 passkey docs / rag_poc_full）
export RAG_CORPUS_DIR=passkey-gateway/docs
export RAG_COLLECTION=rag_poc_full
python3 rag_poc/evaluate.py --qrels rag_poc/qrels.json

# qrels-subset.json（6 ファイルのサブセット / rag_poc_docs）
export RAG_CORPUS_DIR=rag_poc/corpus-subset
export RAG_COLLECTION=rag_poc_docs
python3 rag_poc/evaluate.py --qrels rag_poc/qrels-subset.json

# qrels-alt.json（ADR 6 ファイル / rag_poc_other_domain）
export RAG_CORPUS_DIR=rag_poc/corpus-alt
export RAG_COLLECTION=rag_poc_other_domain
python3 rag_poc/evaluate.py --qrels rag_poc/qrels-alt.json
```

出力 JSON の `corpus_file_count` / `corpus_sha256` / `qrels_label` を確認し、
ラベルとコーパスが対応していることを確かめる。既に投入済みなら
`python3 rag_poc/ingest.py` は不要だが、コーパス切替時は先に投入する。

### 全量コーパスでの実測（2026-11-26） [explanation]

88 ファイル／2200 チャンク（`rag_poc_full`）・held-out 9 クエリの結果。
レイテンシは `RAG_EVAL_REPEAT=10`（ウォームアップ 1 回＋10 回計測＝60 サンプル）
の平均値（`eval-results-baseline.json`）:

| モード | recall@10 | nDCG@10 | レイテンシ平均 | 中央値 | 標準偏差 |
| --- | --- | --- | --- | --- | --- |
| `dense` | 0.0 | 0.0 | 0.183 秒 | 0.182 | 0.017 |
| `keyword` | 1.0 | 0.7681 | 0.045 秒 | 0.044 | 0.006 |
| `hybrid` | 1.0 | 0.6292 | 0.22 秒 | 0.218 | 0.021 |
| `hybrid-rerank` | 1.0 | 0.6243 | 6.22 秒 | 6.25 | 0.66 |

> `dense` 行の但し書き: これは qwen2.5:3b の query-level hit@10 であり集合 recall ではない。Ruri では recall@10=1.0（[全量コーパスでの実測](#全量コーパスでの実測2026-11-26-explanation)の Ruri 表を参照）。

他構成の `hybrid-rerank` 平均は ruri 全量 3.32 秒（中央値 3.33 / 標準偏差 0.18,
`eval-results-ruri-full.json`）、別ドメイン 7.17 秒（中央値 7.05 / 標準偏差 0.45,
`eval-results-alt.json`）。Ruri は埋め込みが高速な分だけ rerank も速い。

`dense` は正解ドキュメントを上位 10 件に一度も取得できなかった。
追跡実験 `python3 rag_poc/compare_embeddings.py`（2026-11-26、
サブセット6ファイル・固定クエリ5件、結果は
`rag_poc/embedding-comparison.json`）では `qwen2.5:3b` の
`avg_top_score` が 0.748604、`qwen2.5:7b` が 0.749004 とほぼ同値で、
両モデルとも「パスキー sudo の使い方」「DSH 起動トークン」の
top_source が `jarvis-muse-dsh-roles.md` だった（正解は
`passkey-sudo-design.md` / `dsh-token-rd-fix-2026-11-25.md`）。
つまり原因はモデルサイズではなく、この汎用埋め込み空間では
日本語クエリと文書の意味が十分に分離されていないこと。
日本語特化モデル（sarashina / ruri 等）またはキーワード側の補完が
必要だという裏付けになる。

実際に `kun432/cl-nagoya-ruri-base`（768 次元、Apache-2.0、
クエリ側「クエリ: 」／文章側「文章: 」prefix）を組み込み、
同一サブセット 6 ファイル・同一 `qrels-subset.json` 6 クエリで
比較した結果（2026-11-26）は「クラウド/ローカル比較」の表へ集約した。

Ruri は正解を rank1 に置けたのに対し、qwen は 6 件中 1 件しか
rank1 に置けなかった。`recall@10` は両者 1.0 だが、ランキング
品質は埋め込みモデルで大きく変わることが実証された。
クラウド埋め込みも同一サブセット・同一 qrels で recall@1 / nDCG@10
ともに 1.0 だった（`eval-results-cloud-subset.json`）。ローカル Ruri と
同等の品質だが、注意は[クラウド章冒頭の警告](#クラウドモデルopencodego-how-to)を参照。
Ruri は 512 トークンcontextのため `CHUNK_SIZE=300` 前後で投入する
（800 のままでは HTTP 400 になる）。サブセット結果は
`rag_poc/eval-results-ruri-subset.json` に退避済み。

全量（88 ファイル／2200 チャンク、`rag_poc_ruri_full`、`qrels.json`
9 クエリ、2026-11-26）でも改善は維持された。結果は
`rag_poc/eval-results-ruri-full.json` に退避済み。

| モード | recall@10 | nDCG@10 |
| --- | --- | --- |
| `dense` | 1.0 | 0.7992 |
| `keyword` | 1.0 | 0.8179 |
| `hybrid` | 1.0 | 0.8624 |
| `hybrid-rerank` | 1.0 | 0.7542 |

opencodeGO の `text-embedding-3-small` を全量コーパス
（91 ファイル／1380 チャンク、`ingest.iter_chunks` 委譲後、
`qrels.json` 9 クエリ、repeat=1）で計測すると、
`hit@1=0.7778`（query-level）/ `set_recall@1=0.7222`（集合 recall）/
`hit@10=1.0` / `nDCG@10=0.8597`（現行定義＝上位10件・ファイル1回）だった
（`eval-results-cloud-full.json`）。ローカル qwen の dense（0.0 / 0.0）
より大幅に良い。ただし全量 Ruri は 88 ファイル／2200 チャンクで
コーパスとチャンク条件が異なるため、Ruri（dense 0.7992）との
nDCG の優劣は直接比較できない。埋め込みレイテンシ平均 42.3 秒
（外部送信＋token 課金の注意は
[クラウド章冒頭の警告](#クラウドモデルopencodego-how-to)を参照）。
生成モデル `deepseek-v4.1-flash` による rerank / 回答生成は
9 クエリで動作を確認した（`rerank_query_count=9`、
`rerank_ndcg_at_5=0.8819`）。rerank の `nDCG@5` は旧 0.3608 が
ideal のファイル重複カウントバグによる過小評価だったため、
修正後の値で置き換えた。
Qdrant 投入後の正式パイプラインでは hybrid が `nDCG@10=0.8681`
で最良だった（`eval-results-cloud-pipeline.json`）。
`cloud_bench.py` の nDCG は上位10件・ファイル単位1回の定義へ
揃えたため、旧 `ndcg_at_10=0.9034` は全順位ベースの値であり
`evaluate.py` の同名指標と直接比較できない点に注意。

qwen の全量ベースライン（`rag_poc_full`）では dense が recall@10 /
nDCG@10 ともに 0.0 だったため、日本語特化埋め込みにより本番相当の
コーパスでも正解を取得できるようになった。`hybrid` が最良で、
rerank は nDCG を下げた（0.7542）。数値は各 JSON の `heldout.summary`
に合わせた。
`keyword` が最良だが、これはファイル名・見出しの語が一致しやすかったためで、
一般の RAG 方式の優劣を示すものではない。
`hybrid-rerank` は `RAG_RERANK_MAX_DOCS=5` により
再ランク前の約 11 秒から約 5.4 秒へ短縮した。

## クラウドモデル（opencodeGO） [how-to]

ユーザー承認により、opencodeGO の OpenAI 互換 API を生成モデルとして
使える。実装は `cloud_client.py` にあり、**既定は無効**（ローカル Ollama
を維持）。有効化するには次を設定する。

> **警告**: 有効化すると質問文・チャンク本文・ベクトルが外部 API へ送信され、
> token 課金が発生する。キーと接続先を確認し、長時間の連続評価は避けること。
> この注意は本章に集約し、他箇所では繰り返さない。

```bash
# dsh.env の認証キーを環境変数へ読み込む（値は表示しない）
set -a; source ~/.config/opencode/dsh.env; set +a

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
比較した結果は「クラウド/ローカル比較」の表へ集約した。

単価は `qwen3-embedding-8b` の方が安いが、この日本語コーパスでは
`text-embedding-3-small` が精度・レイテンシともに上回った
（`eval-results-cloud-small-subset.json` /
`eval-results-cloud-qwen3-subset.json`）。

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

### 簡易計測と正式パイプラインの使い分け [how-to]

| 観点 | 簡易: `cloud_bench.py` | 正式: `cloud_ingest.py` → `evaluate.py` |
| --- | --- | --- |
| 検索 | 埋め込みコサイン類似度のみ（Qdrant を使わない） | Qdrant 投入後に `dense` / `keyword` / `hybrid` / `hybrid-rerank` の 4 モード |
| コーパス | 既定 `corpus-subset`（`--corpus` で変更可） | `RAG_CORPUS_DIR` / `RAG_COLLECTION`（全量は `rag_poc_cloud_full`） |
| 成果物 | `eval-results-cloud-*.json`（`schema_version` なし） | `eval-results-cloud-pipeline.json`（`schema_version` あり） |
| 生成 | `--with-generation` で rerank / 回答（既定は全件、`--generation-max-queries N` で先頭 N 件に制限） | `--with-answer` で回答生成（`RAG_ANSWER_TIMEOUT` が適用） |
| 用途 | 埋め込みモデルの簡易比較。Token 消費を抑えて確認する | 本番相当の 4 モード評価と退避済み成果物の更新 |

`--with-generation` の既定は全件で、全量コーパスでの実測は 9 クエリ
（`rerank_query_count=9`、`rerank_ndcg_at_5=0.8819`、
`nDCG@10=0.8597`）だった。先頭 N 件だけに絞る場合は
`--generation-max-queries N` を指定する。

### クラウド生成で回答を作る（--with-answer） [how-to]

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

### クラウド/ローカル比較（2026-11-26） [explanation]

評価章のサブセット・全量の結果を 1 表へ集約した。`hit@1` は query-level、
`set_recall@1` は集合 recall。ローカルの `recall@1` は `hit@1` に相当する。

| モデル | コーパス | チャンク数 | クエリ数 | hit@1 | set_recall@1 | nDCG@10 | rerank nDCG@5(n=9) | レイテンシ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5:3b`（ローカル） | `corpus-subset`（6 ファイル） | 81 | 6 | 0.1667 | — | 0.5384（dense） | — | — |
| `kun432/cl-nagoya-ruri-base`（ローカル） | `corpus-subset`（6 ファイル） | 81 | 6 | 1.0 | — | 1.0（dense） | — | — |
| `text-embedding-3-small`（クラウド） | `corpus-subset`（6 ファイル） | 81 | 6 | 1.0 | — | 1.0 | — | 8.03 秒（埋め込み平均） |
| `qwen3-embedding-8b`（クラウド） | `corpus-subset`（6 ファイル） | 81 | 6 | 0.8333 | — | 0.8978 | — | 10.04 秒（埋め込み平均） |
| `qwen2.5:3b`（ローカル） | `passkey-gateway/docs`（88 ファイル） | 2200 | 9 | — | — | 0.0（dense）/ 0.6243（hybrid-rerank） | — | 6.22 秒（hybrid-rerank 平均） |
| `kun432/cl-nagoya-ruri-base`（ローカル） | `passkey-gateway/docs`（88 ファイル） | 2200 | 9 | — | — | 0.7992（dense）/ 0.7542（hybrid-rerank） | — | 3.32 秒（hybrid-rerank 平均） |
| `text-embedding-3-small`（クラウド） | `passkey-gateway/docs`（91 ファイル） | 1380 | 9 | 0.7778 | 0.7222 | 0.8597 | 0.8819 | 42.3 秒（埋め込み平均） |
| `text-embedding-3-small`（クラウド・Qdrant 正式） | `passkey-gateway/docs`（87 ファイル） | 214 | 9 | — | — | 0.8681（hybrid） | — | — |

`—` は当該構成で未計測の指標。レイテンシは計測対象が異なる（クラウドは埋め込み平均、
ローカルは hybrid-rerank の検索平均）ため単純比較しないこと。さらに計測回数も
クラウドは `repeat=1`、ローカル全量は `RAG_EVAL_REPEAT=10`
（ウォームアップ 1 回＋10 回計測＝60 サンプル）で混在している。
rerank nDCG@5 は 9 クエリの計測値（`rerank_query_count=9`）。
旧 `nDCG@10=0.9034` は全順位ベースで現行定義（上位10件・ファイル1回）と
直接比較できない。全量クラウドは 91 ファイル／1380 チャンク
（`ingest.iter_chunks` 委譲後）、全量 Ruri は 88 ファイル／2200 チャンクで
コーパスとチャンク条件が異なるため、両者の nDCG に優劣は付けられない。

## 既知の制限 [explanation]

- `keyword` は日本語を空白・記号で分割するため、「起動トークン」の
  ような複合語の部分一致が弱い。形態素解析はしていない。
- 一致判定は小文字化のみ。同義語・表記ゆれ・送り仮名の差は吸収しない。
- `hybrid` は片方の検索が失敗してももう片方で継続するが、結果が
  片面だけになることがある。
- `dense` / 再ランク / 回答生成は Ollama が応答しないと失敗する。
- `hybrid-rerank` が LLM で評価するのは上位 `RAG_RERANK_MAX_DOCS`
  件（既定 5）だけ。スコアは `llm_score` / `base_score` /
  `final_score`（0..1 に正規化した並び替え用、既定の `score` と同値）に
  分離した。各 hit の `reranked_by_llm` が `true` なら LLM 再ランク済み、
  `false` なら元スコア。`base_score` に再ランク前の値を残す。
- `ingest.py` は既定で毎回すべてを再埋め込み・再投入する。
  `RAG_INGEST_RESUME=1` でチェックポイントから再開できる。
  現状は 88 ファイル／2200 チャンク（全量）で数分かかる。
- Qdrant のベクトル次元は `RAG_VECTOR_SIZE`（既定 2048、`qwen2.5:3b` の
  実測値）。モデルを変更する場合は同値を変更しコレクションを作り直す。
  point ID は blake2b 64bit のため、旧 ID のポイントが残る場合は
  コレクションを削除してから再投入する。
- 再ランク／回答生成のプロンプトは質問とチャンクをタグで囲み、
  データとして扱うよう system ロールで指示する。警告ログは LLM 生本文
  ではなく長さ・ハッシュのみを残す。
- `evaluate.py` の指標は真の `recall@k` ではない。
  `source_keyword_hit@3` は上位3件の `source` に expected 語が1つでも
  含まれるかの any-match、`content_keyword_hit_all_hits` は全ヒットの
  `text` に対する any-match で、分母となる取得深度が異なる。
  `avg_distinct_source_ratio_at3` は出典の多様性であり検索品質ではない。
  レイテンシは `RAG_EVAL_REPEAT`（既定 1）回の計測で、1 回のときは
  ウォームアップなし（実運用に近い）。2 以上ならウォームアップ 1 回＋N 回の
  `avg` / `median` / `min` / `max` / `stdev` / `samples` /
  `cold_latency_sec` を記録する。`hybrid-rerank` は CPU の `qwen2.5:3b` が
  上位 5 件分の JSON を生成するため他モードより遅い。
  性能表の値は `RAG_EVAL_REPEAT=10`（60サンプル）で再計測済み。
- 簡易評価は 6 ファイル／81 チャンクのサブセット・6 クエリで実施。
  本番の 88 ファイル／2200 チャンクや一般 RAG 方式の比較ではない。
  held-out 評価は 9 クエリの正解ラベル付きだがドメイン内のみで、
  一般化には別コーパスでの検証が必要。
- チャンク本文は `CHUNK_SIZE` 以内だが、見出し行を先頭に付けるため
  `payload.text` 全体は見出し長だけ長くなる。

## テスト [how-to]

```bash
python3 rag_poc/test_rag.py

# Rust 未ビルドの skip を失敗に変える（CI の最終ゲート）
RAG_REQUIRE_RUST=1 python3 rag_poc/test_rag.py
```

外部サービスへ接続せず、`unittest` のみで設定・RRF・JSON 解析・
チャンク分割・評価クエリ・held-out 指標・rerank のフォールバックを
検証する（全 154 件）。Rust 同等性テスト 9+8件（`ragcore` 側 9 件と
Python 側 8 件）のうち、Rust バイナリを要する 8 件は未ビルド時に
理由付きで skip される。第2のコマンドはこの skip を失敗へ切り替え、
Rust 未ビルドを green で通さないためのゲートである。なお
`binary_path_default` や `RustSkipGateTest` の 3 件など、バイナリが
未ビルドでも skip しない Rust 関連テストもある。
ディレクトリ名にハイフンがあるため `python3 -m unittest` の
ディスカバリは使えず、直接実行する。

## モデル比較実験（2026-11-26 時点） [how-to]

2026-11-26 時点のベースラインでは `dense` の recall@10 が 0.0 で、
日本語技術文書に対する汎用埋め込みの限界が見えた。候補モデルに
入れ替えて同じ held-out 9 クエリで比較する手順を記す。候補は
`qwen2.5:7b` と日本語特化モデルを想定し、取得可能なモデルを選ぶ。

```bash
# リポジトリルートで実行。比較条件を先に固定する（2026-11-26 時点）。
export RAG_CORPUS_DIR=passkey-gateway/docs
export CHUNK_SIZE=800 CHUNK_OVERLAP=100
export RAG_TOP_K=10 RAG_RERANK_MAX_DOCS=5
export RAG_EVAL_REPEAT=1
export RAG_COLLECTION=rag_poc_qwen25_7b

# 候補モデルの埋め込み次元を実測し、変数に設定してから実行する。
: "${NEW_VECTOR_SIZE:?候補モデルの埋め込み次元を設定}"
export EMBED_MODEL=qwen2.5:7b RAG_VECTOR_SIZE="$NEW_VECTOR_SIZE"

ollama pull "$EMBED_MODEL"
python3 rag_poc/ingest.py

# 簡易 6 クエリと held-out 9 クエリを同一条件で実行する。
RAG_COLLECTION=rag_poc_qwen25_7b python3 rag_poc/evaluate.py

# eval-results.json は上書きされるためモデル別に退避する。
cp rag_poc/eval-results.json \
  rag_poc/eval-results-qwen25-7b-20261126.json
```

`RAG_VECTOR_SIZE` は候補モデルの埋め込み次元（実測値）に合わせる。
既定の 2048 は `qwen2.5:3b` の値なので、そのまま使うと `ingest.py` は
次元不一致で停止する。`ensure_collection()` は既存コレクションを
作り直さないので、比較用は専用名（`rag_poc_qwen25_7b`）へ切り替え、
`rag_poc_full`（88 ファイル／2200 チャンク）のベースラインは残す。
回答生成と再ランクの `CHAT_MODEL` は既定のまま変えず、
埋め込みモデルだけを比較対象にする。

結果の記録先は `rag_poc/eval-results.json`。`collection` /
`embed_model` / `chat_model` / `vector_size` / `corpus_sha256` /
`repeat` / `schema_version` が記録されるため、評価時も投入時と
同じ `EMBED_MODEL` を指定すること。既定のままだと実際のベクトルと
記録が食い違う。比較条件はコーパス（`RAG_CORPUS_DIR` と
`corpus_sha256`）、チャンク（`CHUNK_SIZE` / `CHUNK_OVERLAP`）、
取得件数（`RAG_TOP_K`、held-out は `max(10, RAG_TOP_K)`）、
再ランク件数（`RAG_RERANK_MAX_DOCS`）、プロンプト（`query.py` の
system / user プロンプト）、計測回数（`RAG_EVAL_REPEAT`）を揃える。
1 つでも変わると既存の 5.38 秒などの数値と直接比較できない。

### Ruri での再現手順 [how-to]

日本語特化モデル `kun432/cl-nagoya-ruri-base` でサブセットの結果を
再現する手順。Ruri は 512 トークン context のため
`CHUNK_SIZE=300` / `CHUNK_OVERLAP=50` を使う（800 のままでは HTTP 400
になる）。

```bash
# リポジトリルートで実行（2026-11-26 時点の再現条件）。
export RAG_CORPUS_DIR=rag_poc/corpus-subset
export RAG_COLLECTION=rag_poc_ruri
export EMBED_MODEL=kun432/cl-nagoya-ruri-base
export RAG_VECTOR_SIZE=768
export CHUNK_SIZE=300
export CHUNK_OVERLAP=50
export RAG_TOP_K=10
export RAG_RERANK_MAX_DOCS=5
export RAG_EVAL_REPEAT=1

# クエリ側 / 文章側の prefix（config のテーブル値と同じ値を明示）。
export RAG_EMBED_PREFIX_QUERY='クエリ: '
export RAG_EMBED_PREFIX_PASSAGE='文章: '

ollama pull "$EMBED_MODEL"
python3 rag_poc/ingest.py
python3 rag_poc/evaluate.py --qrels rag_poc/qrels-subset.json

# eval-results.json は上書きされるため退避する。
cp rag_poc/eval-results.json rag_poc/eval-results-ruri-subset.json
```

`kun432/cl-nagoya-ruri-base` は config の `_EMBED_PREFIX_TABLE` /
`_EXPECTED_VECTOR_SIZE_TABLE` に登録済みの既知モデル。そのため prefix
（`クエリ:` / `文章:` の後ろに半角スペース）と期待次元 768 は config の
テーブル値が優先され、
`RAG_VECTOR_SIZE` の不一致は `ingest.py` の警告どまり（期待次元 768 を
採用）となる。ただし `query.py` の設定検証は ruri に 768 を要求するため、
上の例のように明示的に 768 を指定する。全量コーパスで再現する場合は
`RAG_CORPUS_DIR=passkey-gateway/docs` /
`RAG_COLLECTION=rag_poc_ruri_full` /
`--qrels rag_poc/qrels.json` に切り替える（結果は
`rag_poc/eval-results-ruri-full.json` に退避済み）。

## 別ドメインベンチマーク（2026-11-26 時点） [how-to]

2026-11-26 時点のベースラインは簡易 6 クエリ＋held-out 9 クエリの
計 15 クエリで、いずれも `passkey-gateway/docs` のドメイン内コーパスと
その中の実在ファイル名を正解ラベルに使う。ドメイン内 15 クエリだけでは
分野内の傾向しか見えず、他ドメインや一般の RAG 方式への一般化は
示せない。別ドメイン（openrustbooks の ADR）で `dense` / `keyword` /
`hybrid` / `hybrid-rerank` の優劣が変わるかを検証するため、
コーパスと正解ラベルを切り替えて同一モードで再計測する。

2026-11-26 時点の別ドメイン用アセットは
`rag_poc/corpus-alt/`（openrustbooks の ADR 6 ファイル）と
`rag_poc/qrels-alt.json`（`simple_queries` 6 件 / `heldout_queries` 6 件）
である。既存の `rag_poc/qrels.json`（ドメイン内 15 クエリ）は変更しない。
`corpus-alt/` の ADR は openrustbooks 由来で AGPL-3.0 の下にある。
本リポジトリも AGPL-3.0-or-later とし、出典をここに明記する。

```bash
# リポジトリルートで実行（2026-11-26 時点）。
# コーパスとコレクションを別ドメイン用へ切り替える。
export RAG_CORPUS_DIR=rag_poc/corpus-alt
export RAG_COLLECTION=rag_poc_other_domain

# 別コーパスを埋め込みして専用コレクションへ投入する。
python3 rag_poc/ingest.py

# ラベルは --qrels オプションで切り替える（2026-11-26 時点で実装済み）。
python3 rag_poc/evaluate.py --qrels rag_poc/qrels-alt.json

# eval-results.json は上書きされるため別ファイルへ退避する。
cp rag_poc/eval-results.json rag_poc/eval-results-other-domain.json
```

`RAG_CORPUS_DIR` がコーパスの切り替え口で、`ingest.py` と
`evaluate.py` は同じ `config.CORPUS_DIR` を参照する。相対パスは
リポジトリルート基準で解決されるため cwd に依存しない。
`RAG_COLLECTION` は専用名（`rag_poc_other_domain`）にして、
`rag_poc_full`（88 ファイル／2200 チャンク）のベースラインと
旧ポイントが混在しないようにする。`RAG_TOP_K` と
`RAG_RERANK_MAX_DOCS` はベースラインと揃え、4 モードを比較する。
出力 JSON の `corpus_file_count` / `corpus_sha256` が旧コーパスと
異なることを確認する。

`--qrels` は qrels ファイルを指定するオプション（2026-11-26 時点で
実装済み、省略時は `rag_poc/qrels.json`）。出力 JSON の `qrels_label` に
指定したファイル名が記録される。
ラベルは別コーパス側のファイル名で用意する。旧ラベルのままだと
`notes` に「held-out 正解ファイルがコーパスに不在」と記録される。
`evaluate.py` の preflight は `RAG_CORPUS_DIR` 配下に各
`relevant_sources` が実在するかを確認する。

### 実測結果（2026-11-26、ADR 6 ファイル／50 チャンク） [explanation]

| モード | recall@10 | nDCG@10 |
| --- | --- | --- |
| `dense` | 0.8333 | 0.5774 |
| `keyword` | 1.0 | 1.0 |
| `hybrid` | 1.0 | 0.8552 |
| `hybrid-rerank` | 1.0 | 0.7936 |

ドメイン内（passkey docs）でも keyword が最良だったが、ADR では
`keyword` が `recall@1` から 1.0 / `nDCG` も 1.0 と明確に最良。
`dense` は 0.83 まで改善したが、ファイル名・見出しの語が一致しやすい
構造のため keyword が有利な点は共通している。結果は
`rag_poc/eval-results-other-domain.json` に退避済み。

結果の解釈には注意が必要である。正解ラベルも人手で作成している
ため完全な客観性は無く、作成者の選択バイアスが残る。数値の差は
「ドメイン内 15 クエリでは見えなかった順位の入れ替わり」を示す
仮説であり、一般の RAG 方式の優劣を確定させるものではない。

## 入力と設定の検証（2026-11-26 時点） [reference]

以下は 2026-11-26 時点の実装を前提にした確認項目。

### 入力長とプロンプト [reference]

- `RAG_QUESTION_MAX_CHARS`（既定 2000）は `config.validate_question_length()`
  で共通検証され、`query.py` の `main()` と `search()` の両方で適用される。
  超過時は CLI では終了コード 2、`search()` では 0 ヒット＋エラー記録になる。
- `rerank()` / `generate_answer()` は `<question>` / `<chunk>` タグ
  内の値を `_escape_text()` でエスケープする（`&` `<` `>` `"`）。
  指示は system ロールに固定し、ユーザーロールにはタグ付きデータ
  だけを置く。タグ内の文章を指示として扱わない設計。

### 警告のマスキング [reference]

- 警告・エラーは `config.mask_secrets()` で URL / 認証情報 / IP /
  パスを `<masked>` に置換して stderr へ出力する。stdout の JSON は
  汚さない。rerank の JSON 抽出失敗では本文を出さず、
  `reason` / `len` / `sha256_12` だけを残す。

### 接続先 [reference]

- `OLLAMA_URL`（既定 `http://127.0.0.1:11434`）と `QDRANT_URL`
  （既定 `http://127.0.0.1:6333`）は既定でループバック
  （`127.0.0.1` / `::1` / `localhost`）のみ許可される。
- リモート URL を使う場合は `RAG_ALLOW_REMOTE_ENDPOINTS=1` の明示的な
  オプトインが必要。オプトインせずリモートを指定すると接続先検証で
  停止する（平文 HTTP の外部ホストは追加で警告される）。
- オプトインしてリモートへ接続すると、質問文・チャンク本文・ベクトルが
  そのホストへ送信される。外部送信にあたるため、信頼できるホスト
  以外への変更は避ける。

### 再開とコレクション [reference]

- `RAG_INGEST_RESUME=1` のチェックポイント
  `.ingest-checkpoint.json` は `collection` / `vector_size` /
  `embed_model` / `signature`（ID と本文の fingerprint）を記録する。
  モデル変更時は `embed_model` が不一致で完了バッチが無効化され、
  最初から投入される。コレクションの新規作成時も無効になる。
- point ID の blake2b 64bit 化で旧 ID が残る場合は `ingest.py` の
  preflight が検出して停止する。旧コレクションを削除してから
  再投入する（[既知の制限](#既知の制限-explanation)も参照）。

## 成果物一覧 [reference]

評価で生成・利用する JSON の一覧。いずれも `rag_poc/` 配下に置く。

| ファイル名 | 生成コマンド | 上書き / 退避 |
| --- | --- | --- |
| `eval-results.json` | `python3 rag_poc/evaluate.py`（既定出力） | 実行ごとに上書きされるが、上書き前に `eval-results-<UTC>.json` へ自動退避（`--output` 指定時は退避なし） |
| `eval-results-<UTC>.json` | 既定実行時の自動退避 | タイムスタンプ付きで蓄積する。保持方針は利用者が決める |
| `eval-results-<collection>.json` | `python3 rag_poc/evaluate.py --collections a,b,c` | コレクション別に個別保存（サニタイズ＋安定ハッシュ suffix で衝突回避） |
| `eval-results-baseline.json` | `RAG_COLLECTION=rag_poc_full python3 rag_poc/evaluate.py --qrels rag_poc/qrels.json`（全量 qwen ベースライン、`RAG_EVAL_REPEAT=10`） | 退避済み。再計測時は `eval-results.json` から `cp` で更新する |
| `eval-results-alt.json` | `RAG_COLLECTION=rag_poc_other_domain python3 rag_poc/evaluate.py`（別ドメインコーパスで計測後に退避） | 退避済み。`eval-results.json` から `cp` で更新する |
| `eval-results-other-domain.json` | `python3 rag_poc/evaluate.py --qrels rag_poc/qrels-alt.json`（`RAG_CORPUS_DIR=rag_poc/corpus-alt` / `RAG_COLLECTION=rag_poc_other_domain`） | 退避済み。`evaluate.py` の直接出力先ではないため上書きされない |
| `eval-results-ruri-subset.json` | `python3 rag_poc/evaluate.py --qrels rag_poc/qrels-subset.json`（`RAG_COLLECTION=rag_poc_ruri`） | 退避済み。再計測時は `eval-results.json` から `cp` で更新する |
| `eval-results-ruri-full.json` | `python3 rag_poc/evaluate.py --qrels rag_poc/qrels.json`（`RAG_CORPUS_DIR=passkey-gateway/docs` / `RAG_COLLECTION=rag_poc_ruri_full`） | 退避済み。再計測時は `eval-results.json` から `cp` で更新する |
| `eval-results-cloud-full.json` | `RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py --corpus ../passkey-gateway/docs --qrels rag_poc/qrels.json --output rag_poc/eval-results-cloud-full.json --with-generation` | 退避済み（注意は[クラウド章冒頭の警告](#クラウドモデルopencodego-how-to)を参照） |
| `eval-results-cloud-pipeline.json` | `RAG_CLOUD_ENABLED=1 RAG_VECTOR_SIZE=1536 RAG_COLLECTION=rag_poc_cloud_full python3 rag_poc/evaluate.py`（投入は `cloud_ingest.py`） | 退避済み。4モードの正式パイプライン評価 |
| `eval-results-cloud-subset.json` | `RAG_CLOUD_ENABLED=1 RAG_CLOUD_EMBED_MODEL=text-embedding-3-small python3 rag_poc/cloud_bench.py --corpus rag_poc/corpus-subset --qrels rag_poc/qrels-subset.json --output rag_poc/eval-results-cloud-subset.json`（当時は `CHUNK_SIZE=300`） | 退避済み（[クラウド章冒頭の警告](#クラウドモデルopencodego-how-to)を参照） |
| `eval-results-cloud-small-subset.json` | 同上を `ingest.iter_chunks` 既定の `CHUNK_SIZE` で計測し `--output rag_poc/eval-results-cloud-small-subset.json` | 退避済み。`text-embedding-3-small` のサブセット結果 |
| `eval-results-cloud-qwen3-subset.json` | 同上を `RAG_CLOUD_EMBED_MODEL=qwen3-embedding-8b` で計測し `--output rag_poc/eval-results-cloud-qwen3-subset.json` | 退避済み。`qwen3-embedding-8b` のサブセット結果 |
| `embedding-comparison.json` | `python3 rag_poc/compare_embeddings.py` | 実行ごとに上書き。必要なら `cp` で退避する |
| `cloud-local-comparison.json` | `python3 rag_poc/compare_cloud_local.py --output rag_poc/cloud-local-comparison.json` | 実行ごとに上書き。新規 API 呼び出しなしで既存 `eval-results*.json` から生成 |
| `qrels.json` / `qrels-subset.json` / `qrels-alt.json` | 人手で作成・管理（生成コマンドなし） | `evaluate.py` は読み取り専用で上書きしない。コーパス更新時に内容を見直す |

`eval-results.json` の `qrels_label` に指定した qrels ファイル名が記録される。
退避ファイルを比較するときは `collection` / `embed_model` / `vector_size` /
`corpus_sha256` / `qrels_label` も揃っているか確認する。

公開用に成果物をコミットする際は `generated_at` を日付精度
（例: `2026-11-26T00:00:00+00:00`）へ丸める。コード変更は不要で、
`.gitignore` の `eval-results-cloud-*` 名示例外は維持したまま運用で対応する。
丸めは `generated_at` だけに適用し、比較条件のキーは変更しない。
