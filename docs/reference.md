---
type: Reference
title: "rag_poc 設定・検索モード・成果物リファレンス"
description: "rag_poc の検索モード、全環境変数（既定値・単位）、評価成果物一覧と代表的な実測値を集約した参照ドキュメント"
tags:
  - rag
  - configuration
  - search-modes
  - artifacts
  - reference
generated:
  by: "dsh-subagent/eg-reference"
  at: "2026-11-26T21:41:25+09:00"
verified:
  - by: "dsh-subagent/eg-reference"
    at: "2026-11-26T21:41:25+09:00"
status: "stable"
stale_after: "2026-12-26"
---

> 出典は分割前 README スナップショット（`/tmp/README-full-backup.md`）の
> 「検索モード」「設定（環境変数）」「評価」「成果物一覧」の各章である。
> 現行 README では章の再編があるため行番号は示さない。数値・相対パス・
> コマンドは元 README の記述をそのまま転記している。表内の `#...` アンカーは
> 元 README 内の見出しを指す。

## 概要

`config.py` が接続先・モデル・チャンク設定を一元管理し、各値は環境変数で
上書きできる。既定はローカル Ollama + Qdrant で、クラウドは
`RAG_CLOUD_ENABLED=1` の opt-in。認証情報の値はログ・例外・成果物へ出力しない。

## 検索モード

| モード | 動作 |
| --- | --- |
| `dense` | 質問を埋め込み、Qdrant でコサイン類似度の意味検索 |
| `keyword` | 全ポイントを scroll し、語の中包含一致でスコア化 |
| `hybrid` | dense と keyword の順位を RRF (k=60) で統合 |
| `hybrid-rerank` | hybrid の上位 5 件を LLM が再ランクし全件を返す |

`RAG_COLLECTION` は参照する Qdrant コーパスを切り替える。サブセットは
`rag_poc_docs`（既定）、全量は `rag_poc_full`。投入時と検索・評価時で同じ値を
指定すること。`RAG_RERANK_MAX_DOCS` は LLM に再ランクさせる件数（既定 5）。
上位 5 件のみを再評価し、残りは元のスコアを保持するため、返却件数は
`--top-k` のまま変わらない。

## 設定（環境変数）

`config.py` の既定値は次のとおり。必要なら環境変数で上書きする。全変数を
5 グループに整理した。

### 接続先

| 環境変数 | 既定値 | 単位・備考 |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | URL |
| `QDRANT_URL` | `http://127.0.0.1:6333` | URL |
| `RAG_COLLECTION` | `rag_poc_docs`（全量は `rag_poc_full`） | コレクション名 |
| `RAG_CORPUS_DIR` | `passkey-gateway/docs` | ディレクトリパス（リポジトリ配下） |
| `RAG_ALLOW_REMOTE_ENDPOINTS` | `0`（`1` でオプトイン） | フラグ |
| `RAG_ALLOWED_ENDPOINT_HOSTS` | 空（カンマ区切りの許可ホスト。`host:port` 形式も可） | ホスト一覧 |
| `RAG_CLOUD_BASE_URL` | `https://opencode.ai/inference/go/openai/v1` | URL |

### モデル

| 環境変数 | 既定値 | 単位・備考 |
| --- | --- | --- |
| `EMBED_MODEL` | `qwen2.5:3b` | 埋め込みモデル ID |
| `CHAT_MODEL` | `qwen2.5:3b` | 生成モデル ID |
| `RAG_VECTOR_SIZE` | `2048` | 次元（ruri は 768） |
| `RAG_CLOUD_EMBED_MODEL` | 空（有効時は明示指定が必要。既定モデルは `text-embedding-3-small`） | 埋め込みモデル ID |
| `RAG_CLOUD_MODEL` | `deepseek-v4.1-flash` | 生成モデル ID |
| `RAG_CLOUD_EMBED_DIMENSION` | 空（正規。未知モデルの期待次元を明示。未指定なら fail-closed。`RAG_CLOUD_DIMENSIONS` より優先） | 次元（正の整数） |
| `RAG_CLOUD_DIMENSIONS` | 空（移行期互換の複数形。正規が未設定のときだけ読む） | 次元（正の整数） |

### チャンク・評価

| 環境変数 | 既定値 | 単位・備考 |
| --- | --- | --- |
| `CHUNK_SIZE` | `800` | 文字（ruri は 512 以下） |
| `CHUNK_OVERLAP` | `100` | 文字 |
| `RAG_TOP_K` | `10` | 件 |
| `RAG_RERANK_MAX_DOCS` | `5` | 件 |
| `OLLAMA_TIMEOUT` | `300 秒` | 秒 |
| `QDRANT_TIMEOUT` | `300 秒` | 秒 |
| `RAG_ANSWER_TIMEOUT` | `60 秒`（回答生成 1 回のタイムアウト） | 秒 |
| `RAG_INGEST_RESUME` | `0`（1 でチェックポイント再開） | フラグ |
| `RAG_EVAL_REPEAT` | `1`（ウォームアップなし 1 回。2 以上でウォームアップ 1 回＋N 回） | 回 |
| `RAG_QUESTION_MAX_CHARS` | `2000` | 文字 |
| `RAG_EMBED_PREFIX_QUERY` | `""`（空。ruri は config テーブルで「クエリ: 」） | 文字列 |
| `RAG_EMBED_PREFIX_PASSAGE` | `""`（空。ruri は config テーブルで「文章: 」） | 文字列 |

### Rust

| 環境変数 | 既定値 | 単位・備考 |
| --- | --- | --- |
| `RAG_RUST_BIN` | `rag_poc/ragcore/target/debug/ragcore` | バイナリパス（リポジトリルート基準） |
| `RAG_RUST_TIMEOUT` | `30 秒` | 秒 |
| `RAG_REQUIRE_RUST` | 未設定（実質無効。`1` で Rust 必須ゲート） | フラグ（テスト時） |

### クラウド・セキュリティ

| 環境変数 | 既定値 | 単位・備考 |
| --- | --- | --- |
| `RAG_CLOUD_ENABLED` | `0`（1 でクラウド有効。既定はローカル Ollama） | フラグ |
| `RAG_CLOUD_CHAT_ENABLED` | `0`（1 で生成もクラウド。`RAG_CLOUD_ENABLED=1` が前提） | フラグ |
| `RAG_CLOUD_TIMEOUT` | `60 秒`（クラウド HTTP タイムアウト） | 秒 |
| `RAG_CLOUD_CORPUS_SHA256` | 空（未設定なら実コーパスから計算） | SHA-256 |
| `OPENCODE_GO_API_KEY` | 未設定（正規のキー名。値は記載しない） | 認証情報 |
| `OPENCODE_API_KEY` | 未設定（後方互換のキー名。値は記載しない） | 認証情報 |

### 優先順位と検証ルール

- リモート endpoint は **オプトイン＋許可ホスト＋https の 3 条件をすべて満たす場合だけ**致命問題が警告へ格下げされる。平文 HTTP の外部ホスト、許可リストに無いホスト、不正 scheme / port はオプトインしても拒否される（`config.validate_endpoint()` が権威）。
- 設定例: `RAG_ALLOW_REMOTE_ENDPOINTS=1 RAG_ALLOWED_ENDPOINT_HOSTS=ollama.example.test` ＋ `OLLAMA_URL=https://ollama.example.test:11434`。
- `RAG_QUESTION_MAX_CHARS` は CLI と `search()` の両方で適用され、超過すると 0 ヒット＋エラー記録になる。
- `RAG_EMBED_PREFIX_QUERY` / `RAG_EMBED_PREFIX_PASSAGE` が非空なら config のモデル別 prefix より優先される。
- `RAG_RUST_BIN` の相対パスはリポジトリルート基準で解決し、`RAG_RUST_TIMEOUT` は Rust CLI 専用（`OLLAMA_TIMEOUT` は流用しない）。
- 埋め込み次元は単数形 `RAG_CLOUD_EMBED_DIMENSION` が正規で、複数形 `RAG_CLOUD_DIMENSIONS` は移行期互換。両方が設定されていれば単数形を優先し、値は出さずに環境変数名だけを警告する。
- `RAG_RERANK_TOP_N` は廃止した。実効値は `RAG_RERANK_MAX_DOCS`。
- `eval-results.json` には `embed_model` / `chat_model` / `vector_size` / `corpus_file_count` / `corpus_sha256` / `repeat` を記録する。
- `RAG_VECTOR_SIZE` の実効値には優先順位がある。`EMBED_MODEL` が `config._EXPECTED_VECTOR_SIZE_TABLE` に登録済みの既知モデル（例: `kun432/cl-nagoya-ruri-base` は 768、`qwen2.5:3b` は 2048）ではテーブル値が優先され、環境変数の不一致は `ingest.py` の警告どまり（期待次元を採用）となる。未知モデルの場合だけ `RAG_VECTOR_SIZE` が実効値になる。ただし `query.py` の設定検証は ruri に 768 を要求するため、ruri を使う場合は環境変数も 768 に揃えること。
- 認証キーは `OPENCODE_GO_API_KEY`（正規）を使い、`OPENCODE_API_KEY` は後方互換。両方が設定されている場合は正規が優先され、値はログ・例外・成果物へ出力しない。

## qrels ファイルの使い分け

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

## 評価指標の計算定義

held-out は 9 クエリに正解ファイル名ラベルを付け、次をモード別に集計する。

- `recall@1/3/5/10`: 上位 k 件の `source` に正解が 1 件でもあれば 1、
  なければ 0 をクエリ平均（query-level hit@k）。正解ファイル全体の
  回収割合を測る集合 recall ではない。
- `nDCG@3/10`: gain=1 / discount=log2(rank+1) の DCG を IDCG で割り、
  クエリ平均する。同一ファイルの複数チャンクは最上位の 1 回だけ計上。

正解は `passkey-gateway/docs` 内の実在ファイル名。完全一致・相対パス
suffix・basename で照合し、検索失敗ケースは 0 として分母に残す。
recall@10 / nDCG@10 のため取得件数は最大 10 まで引き上げる。

## 代表的な実測値

元 README の評価章に記載された数値を変更せず転記する。比較条件が表だけ
で完結するよう definition / コーパス / チャンク / クエリ / repeat の列を
追加した。出典に値が無い場合は `—` とする。レイテンシは計測対象が
異なる（クラウドは埋め込み平均、ローカルは `hybrid-rerank` 検索平均）ため、
definition 列と併せて読むこと。

| 指標 | 値 | definition | コーパス | チャンク | クエリ | repeat | 出典 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 全量 qwen `hybrid-rerank` の nDCG@10 | 0.6243 | evaluate query-level hit@k | `rag_poc_full`（88 ファイル） | 2200 | 9 | 10 | `eval-results-baseline.json` |
| 全量 Ruri `hybrid-rerank` の nDCG@10 | 0.7542 | evaluate query-level hit@k | `rag_poc_ruri_full`（88 ファイル） | 2200 | 9 | `—` | `eval-results-ruri-full.json` |
| クラウド Qdrant 正式パイプライン hybrid の nDCG@10 | 0.8681 | evaluate query-level hit@k | `rag_poc_cloud_full`（88 ファイル） | 214 | 9 | 1 | `eval-results-cloud-pipeline.json` |
| クラウド全量 `text-embedding-3-small` の nDCG@10 | 0.8597 | cloud_bench top10 + hit@k | 91 ファイル | 1380 | 9 | 1 | `eval-results-cloud-full.json` |
| rerank nDCG@5 | 0.8819 | cloud_bench top10 + hit@k | 91 ファイル | 1380 | 9（`rerank_query_count=9`） | 1 | `eval-results-cloud-full.json` |
| `qwen3-embedding-8b` の nDCG@10 | 0.8978 | cloud_bench top10 + hit@k | 6 ファイルのサブセット | 81 | 6 | 1 | `eval-results-cloud-qwen3-subset.json` |
| 全量 qwen `hybrid-rerank` のレイテンシ平均 | 6.22 秒 | evaluate query-level hit@k | `rag_poc_full`（88 ファイル） | 2200 | 9 | 10 | `eval-results-baseline.json`（ウォームアップ 1 回＋10 回計測＝60 サンプル） |

### レイテンシ実測値

`todo.md` のレイテンシ欠落項目（8.03 / 10.04 / 42.3 / 3.32 秒）を条件列
付きで追記する。値は既存記録の転記であり、新たな実計測は行っていない。

| レイテンシ | definition | コーパス | チャンク | クエリ | repeat | 出典・照合状況 |
| --- | --- | --- | --- | --- | --- | --- |
| 8.03 秒 | cloud_bench top10 + hit@k | 6 ファイルのサブセット | 81 | 6 | 1 | 親提示値（`todo.md` GC-DOCS）。ローカル JSON の `avg` とは計測範囲が異なる可能性があり要照合 |
| 10.04 秒 | evaluate query-level hit@k | `rag_poc_cloud_full`（88 ファイル） | 214 | 9 | 1 | 親提示値（`todo.md` GC-DOCS）。ローカル JSON の `avg` とは計測範囲が異なる可能性があり要照合 |
| 42.3 秒 | cloud_bench top10 + hit@k | 91 ファイル | 1380 | 9 | 1 | `eval-results-cloud-full.json` の埋め込み平均 42.259882 秒を丸めた値（一致確認済み） |
| 3.32 秒 | evaluate query-level hit@k | `rag_poc_ruri_full`（88 ファイル） | 2200 | 9 | `—` | `eval-results-ruri-full.json` の `hybrid-rerank` 検索平均 3.322006 秒を丸めた値（一致確認済み） |

8.03 / 10.04 秒の成果物単位の照合は `compare_cloud_local.py --run-ab` で
条件を揃えてから行うこと（[A/B 比較の手順](../README.md#ab-比較の手順--run-ab)）。

## 成果物一覧

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
| `eval-results-cloud-full.json` | `RAG_CLOUD_ENABLED=1 python3 rag_poc/cloud_bench.py --corpus ../passkey-gateway/docs --qrels rag_poc/qrels.json --output rag_poc/eval-results-cloud-full.json --with-generation` | 退避済み。91 ファイル／1380 チャンク（注意は[クラウド章冒頭の警告](../README.md#クラウドモデルopencodego)を参照） |
| `eval-results-cloud-pipeline.json` | `RAG_CLOUD_ENABLED=1 RAG_VECTOR_SIZE=1536 RAG_COLLECTION=rag_poc_cloud_full python3 rag_poc/evaluate.py`（投入は `cloud_ingest.py`） | 退避済み。88 ファイル／214 チャンクの4モード正式パイプライン評価（旧比較表の 87 ファイルは旧値） |
| `eval-results-cloud-subset.json` | `RAG_CLOUD_ENABLED=1 RAG_CLOUD_EMBED_MODEL=text-embedding-3-small python3 rag_poc/cloud_bench.py --corpus rag_poc/corpus-subset --qrels rag_poc/qrels-subset.json --output rag_poc/eval-results-cloud-subset.json`（当時は `CHUNK_SIZE=300`） | 退避済み（[クラウド章冒頭の警告](../README.md#クラウドモデルopencodego)を参照） |
| `eval-results-cloud-small-subset.json` | 同上を `ingest.iter_chunks` 既定の `CHUNK_SIZE` で計測し `--output rag_poc/eval-results-cloud-small-subset.json` | 退避済み。`text-embedding-3-small` のサブセット結果 |
| `eval-results-cloud-qwen3-subset.json` | 同上を `RAG_CLOUD_EMBED_MODEL=qwen3-embedding-8b` で計測し `--output rag_poc/eval-results-cloud-qwen3-subset.json` | 退避済み。`qwen3-embedding-8b` のサブセット結果 |
| `eval-results-cloud-gemini-subset.json` | 同上を `RAG_CLOUD_EMBED_MODEL=gemini-embedding-2` で計測し `--output rag_poc/eval-results-cloud-gemini-subset.json`（3072 次元、`repeat=1`） | 退避済み。`gemini-embedding-2` のサブセット結果（ファイルは実在） |
| `embedding-comparison.json` | `python3 rag_poc/compare_embeddings.py` | 実行ごとに上書き。必要なら `cp` で退避する |
| `cloud-local-comparison.json` | `python3 rag_poc/compare_cloud_local.py --output rag_poc/cloud-local-comparison.json` | 実行ごとに上書き。新規 API 呼び出しなしで既存 `eval-results*.json` から生成 |
| `qrels.json` / `qrels-subset.json` / `qrels-alt.json` | 人手で作成・管理（生成コマンドなし） | `evaluate.py` は読み取り専用で上書きしない。コーパス更新時に内容を見直す |

### 退避ファイルの比較条件

`eval-results.json` の `qrels_label` に指定した qrels ファイル名が記録される。
退避ファイルを比較するときは `collection` / `embed_model` / `vector_size` /
`corpus_sha256` / `qrels_label` も揃っているか確認する。

公開用に成果物をコミットする際は `generated_at` を日付精度
（例: `2026-11-26T00:00:00+00:00`）へ丸める。コード変更は不要で、
`.gitignore` の `eval-results-cloud-*` 名示例外は維持したまま運用で対応する。
丸めは `generated_at` だけに適用し、比較条件のキーは変更しない。
**未丸めの cloud 成果物が残っている場合は公開前に必ずこの丸めを適用する**
（JSON の直接編集は本担当範囲外のため、公開前の運用で対応する）。

## Rust op の配線状況

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
`latency`）は未配線で、`ingest.py` / `evaluate.py` の該当処理は純 Python の
まま、Rust 実装は同等性テスト（`test_rag.py`）でのみ使われる。
`check_eval_results` は上記 9 op に含めない、CI ゲート専用の別 op である。
`python3 rag_poc/rust_bridge.py check_eval_results` が `eval-results-*.json`
のパスだけを Rust へ渡し、成果物のキー検証をゲートする（Python の検索・評価
経路では呼ばない）。

## コストと token 推定キー

`cloud_bench.py` が出力する `eval-results-cloud-*.json` の `usage` / `summary`
には次のキーが入る（根拠: `cloud_bench.py` の `_usage_cost_source` /
`_estimated_cost_usd` / `_token_cost_usd`）。

| キー | 値 | 意味 |
| --- | --- | --- |
| `usage.cost_source` | `api` / `api-zero` / `estimated` / `None` | API 実値／cost 0.0／単価推定／判定不能 |
| `summary.estimated_cost_usd` | 小数または `None` | API cost を 6 桁で保持。cost が無ければ単価 × total_tokens の 6 桁推定 |
| `summary.estimated_cost_from_tokens_usd` | 小数（12 桁） | 6 桁丸めで 0.0 に潰れる場合のトークン由来見込み |
| `usage.estimated_cost_from_tokens_usd` | 小数（12 桁） | 同上（`usage` 側にも記録） |

`cost_source=api-zero` は `cost=0.0` かつ `total_tokens>0` のときに設定する。
**API が 0.0 を返した理由は未確認**で、実コストがゼロだと断定しない。
`summary.estimated_cost_usd` は API cost（0.0）を維持し、主指標は
`estimated_cost_from_tokens_usd` とする。成果物 `notes` に
「cost_source=api-zero のときは token 推定を参照」を明記する。
`cost_source=estimated` は cost キーが無い場合、`api` は正の cost、
`None` は cost が無く単価推定もできない場合を表す。

## 検証

親エージェントが lint と OKF 検証を実行済み（lint 0 / OKF valid）。
子エージェントは shell 実行権限を持たないため、ここでは再確認用の
コマンドのみを示す。

```bash
markdownlint-cli2 rag_poc/docs/*.md
rust-script /home/koppe/.agents/skills/okf/scripts/okf_validate.rs rag_poc/docs/reference.md
```

転記漏れの確認には次を用いる。

```bash
grep -nE "0\.6243|0\.7542|0\.8681|0\.8597|0\.8819|0\.8978|6\.22" rag_poc/docs/reference.md
```

## 参照元

- 分割前 README スナップショット（`/tmp/README-full-backup.md`）: 「検索モード」「設定（環境変数）」「評価」「成果物一覧」の各章
- [rag_poc/docs/index.md](index.md): 詳細ドキュメントの索引
- `rag_poc/config.py`: 設定の既定値、優先順位、endpoint・クラウド設定の検証
- `rag_poc/cloud_client.py` / `rag_poc/cloud_bench.py` / `rag_poc/cloud_ingest.py`: クラウド設定とキー名の参照
- `rag_poc/rust_bridge.py`: Rust バイナリとタイムアウトの参照
- `rag_poc/query.py` / `rag_poc/evaluate.py` / `rag_poc/ingest.py` / `rag_poc/test_rag.py`: 各環境変数の参照箇所
