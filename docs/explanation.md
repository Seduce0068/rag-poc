---
type: Explanation
title: "rag_poc 設計判断の説明"
description: "ローカル RAG 検証 PoC の構成・ハイブリッド検索・モデル比較の設計理由と既知の制限を、README の各章を根拠に説明する。"
tags:
  - rag
  - explanation
  - hybrid-search
  - embedding
  - ollama
  - qdrant
  - japanese
generated:
  by: "team:eh-explanation"
  at: "2026-11-26T21:40:00+09:00"
verified:
  - by: "human:lead"
    at: "2026-11-26T21:41:00+09:00"
status: "stable"
---

## この文書の目的と位置づけ

`rag_poc/README.md` は入口に限定し、背景は `docs/explanation.md`、
詳細仕様と実測データは `docs/reference.md` 等へ順次分離する予定である。
本文書はそのうち「なぜこの構成か」を説明する背景ドキュメントである。

根拠は分割前 README スナップショット（`/tmp/README-full-backup.md`）の次の 6 章。
現行 README では章の再編があるため、行番号ではなく見出し名で示す。

- `## 構成図（テキスト）`
- `## 検索モード`（設計背景のみ）
- `## 既知の制限`
- `## モデル比較実験`
- `## 別ドメインベンチマーク`
- `## 入力と設定の検証`

数値・相対パス・コマンドは README の記載を変更せず転記する。

## 構成の設計判断

### ローカル Ollama と Qdrant を既定にした理由

PoC の目的は、外部 API を使わずローカルの Ollama と Qdrant だけで
RAG を試すことである。埋め込みと回答生成を Ollama、ベクトル検索を
Qdrant が担う。クラウドは `RAG_CLOUD_ENABLED=1` の opt-in とし、
既定はローカルを維持する。これにより既定状態では外部送信も token 課金も
発生しない。

データの流れは `passkey-gateway/docs/*.md` → `ingest.py`（チャンク分割）
→ `ollama_client.py`（`/api/embed`）→ Qdrant コレクション
`rag_poc_docs`（Cosine / 2048 次元）→ `query.py`（4 モード）
→ 再ランク / 回答生成（`/api/chat`）→ `evaluate.py`
→ `rag_poc/eval-results.json` である。

### 共通クライアントと設定の一元化

`config.py` が接続先・モデル・チャンク設定を一元管理し、各値は環境変数で
上書きできる。`ollama_client.py` は埋め込み呼び出しを共通化し、
`ingest.py` / `query.py` / `compare_embeddings.py` がこれを利用する。
prefix・次元検証・リトライ・旧 API（`/api/embeddings`）フォールバックを
一箇所で管理するのは、モデル差と API 差を呼び出し側へ漏らさないためである。

`RAG_VECTOR_SIZE` の実効値には優先順位がある。
`kun432/cl-nagoya-ruri-base` は 768、`qwen2.5:3b` は 2048 が
`config._EXPECTED_VECTOR_SIZE_TABLE` のテーブル値として優先され、
未知モデルの場合だけ環境変数が実効値になる。

### Rust の段階的導入

Rust バイナリは `rag_poc/ragcore/target/debug/ragcore` に生成されるが、
本番配線は `compare_embeddings.py` の `cosine_batch` と `distribution`
の 2 op に限定される。`rust_bridge.py` 経由で呼び、Rust が無い／
失敗した場合は Python 実装へフォールバックする。
`chunk` / `point_id` / `fingerprint` / `recall` / `ndcg` / `latency` は
`test_rag.py` の同等性テストでのみ使われる（Rust 側 9 件と Python 側 8 件の
計 17 件（9+8件））。完全な Rust 再実装ではなく、効果の測れる箇所から
段階的に置き換える判断である。

### 入力と設定の検証が設計に与える影響

`RAG_QUESTION_MAX_CHARS`（既定 2000）は `config.validate_question_length()`
で共通検証され、`query.py` の `main()` と `search()` の両方で適用される。
超過時は CLI では終了コード 2、`search()` では 0 ヒット＋エラー記録になる。

`rerank()` / `generate_answer()` は `<question>` / `<chunk>` タグ内の値を
`_escape_text()` でエスケープする（`&` `<` `>` `"`）。指示は system ロールに
固定し、ユーザーロールにはタグ付きデータだけを置く。タグ内の文章を
指示として扱わない設計である。警告・エラーは `config.mask_secrets()` で
URL / 認証情報 / IP / パスを `<masked>` に置換して stderr へ出力し、
stdout の JSON は汚さない。

接続先は既定でループバック（`127.0.0.1` / `::1` / `localhost`）のみ許可し、
リモート URL は `RAG_ALLOW_REMOTE_ENDPOINTS=1` の明示的なオプトインが必要。
オプトインすると質問文・チャンク本文・ベクトルがそのホストへ送信されるため、
信頼できるホスト以外への変更は避ける。`RAG_INGEST_RESUME=1` の
チェックポイント `.ingest-checkpoint.json` は `collection` / `vector_size` /
`embed_model` / `signature` を記録し、モデル変更時は完了バッチが無効化されて
最初から投入される。

## 検索モードの設計判断

### 4 モードの役割分担

| モード | 動作 |
| --- | --- |
| `dense` | 質問を埋め込み、Qdrant でコサイン類似度の意味検索 |
| `keyword` | 全ポイントを scroll し、語の中包含一致でスコア化 |
| `hybrid` | dense と keyword の順位を RRF (k=60) で統合 |
| `hybrid-rerank` | hybrid の上位 5 件を LLM が再ランクし全件を返す |

### なぜハイブリッドが必要だったか

88 ファイル／2200 チャンク（`rag_poc_full`）・held-out 9 クエリの
`RAG_EVAL_REPEAT=10`（ウォームアップ 1 回＋10 回計測＝60 サンプル）の
ベースラインでは `dense` が recall@10 / nDCG@10 ともに 0.0 だった。

| モード | recall@10 | nDCG@10 | レイテンシ平均 | 中央値 | 標準偏差 |
| --- | --- | --- | --- | --- | --- |
| `dense` | 0.0 | 0.0 | 0.183 秒 | 0.182 | 0.017 |
| `keyword` | 1.0 | 0.7681 | 0.045 秒 | 0.044 | 0.006 |
| `hybrid` | 1.0 | 0.6292 | 0.22 秒 | 0.218 | 0.021 |
| `hybrid-rerank` | 1.0 | 0.6243 | 6.22 秒 | 6.25 | 0.66 |

追跡実験 `python3 rag_poc/compare_embeddings.py` では `qwen2.5:3b` の
`avg_top_score` が 0.748604、`qwen2.5:7b` が 0.749004 とほぼ同値で、
両モデルとも「パスキー sudo の使い方」「DSH 起動トークン」の
top_source が `jarvis-muse-dsh-roles.md` だった（正解は
`passkey-sudo-design.md` / `dsh-token-rd-fix-2026-11-25.md`）。
つまり原因はモデルサイズではなく、この汎用埋め込み空間では
日本語クエリと文書の意味が十分に分離されていないことである。
一方 `keyword` は日本語を空白・記号で分割し形態素解析をしないため、
「起動トークン」のような複合語の部分一致が弱い。
両者の弱点が補完関係にあるため、順位を RRF (k=60) で統合する
`hybrid` を既定（`--mode` 省略時）にした。`hybrid` は片方の検索が
失敗してももう片方で継続するが、結果が片面だけになることがある。

### 再ランクのトレードオフ

`hybrid-rerank` が LLM で評価するのは上位 `RAG_RERANK_MAX_DOCS`
件（既定 5）だけである。返却件数は `--top-k` のまま変わらない。
スコアは `llm_score` / `base_score` / `final_score`
（0..1 に正規化した並び替え用、既定の `score` と同値）に分離し、
各 hit の `reranked_by_llm` が `true` なら LLM 再ランク済み、
`false` なら元スコア、`base_score` に再ランク前の値を残す。
全量 Ruri では `hybrid` の nDCG@10=0.8624 に対し
`hybrid-rerank` は 0.7542 と下がった。README では
`hybrid-rerank` は `RAG_RERANK_MAX_DOCS=5` により再ランク前の
約 11 秒から約 5.4 秒へ短縮したと記録する。
レイテンシと精度の調整ノブとして残す判断である。

## モデル比較の設計判断

### なぜ Ruri を比較したか

2026-11-26 時点のベースラインでは `dense` の recall@10 が 0.0 で、
日本語技術文書に対する汎用埋め込みの限界が見えた。候補は
`qwen2.5:7b` と日本語特化モデルを想定し、取得可能なモデルを選ぶ。
日本語特化 `kun432/cl-nagoya-ruri-base`（768 次元、Apache-2.0、
クエリ側「クエリ: 」／文章側「文章: 」prefix）を組み込んだ結果、
サブセット 6 ファイル・6 クエリで Ruri は正解を rank1 に置けたのに対し、
qwen は 6 件中 1 件しか rank1 に置けなかった。`recall@10` は両者 1.0 だが、
ランキング品質は埋め込みモデルで大きく変わる。
Ruri は 512 トークン context のため `CHUNK_SIZE=300` /
`CHUNK_OVERLAP=50` を使う（800 のままでは HTTP 400 になる）。

全量（88 ファイル／2200 チャンク、`rag_poc_ruri_full`、`qrels.json`
9 クエリ、2026-11-26）でも改善は維持された。

| モード | recall@10 | nDCG@10 |
| --- | --- | --- |
| `dense` | 1.0 | 0.7992 |
| `keyword` | 1.0 | 0.8179 |
| `hybrid` | 1.0 | 0.8624 |
| `hybrid-rerank` | 1.0 | 0.7542 |

`hybrid` が最良で、rerank は nDCG を下げた（0.7542）。
他構成の `hybrid-rerank` 平均は ruri 全量 3.32 秒
（中央値 3.33 / 標準偏差 0.18, `eval-results-ruri-full.json`）、
別ドメイン 7.17 秒（中央値 7.05 / 標準偏差 0.45,
`eval-results-alt.json`）。Ruri は埋め込みが高速な分だけ rerank も速い。

### モデル比較実験（2026-11-26 時点）

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

### Ruri での再現手順

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

### なぜクラウドを比較したか

opencodeGO の OpenAI 互換 API はユーザー承認で使えるが、
**既定は無効**で `RAG_CLOUD_ENABLED=1` を明示する。
有効化すると質問文・チャンク本文・ベクトルが外部 API へ送信され、
token 課金が発生するため、ローカル Ruri と同等の品質を別経路でも
確認し、opt-in の判断材料にする目的で比較した。

埋め込みモデルの単価は `text-embedding-3-small` が $0.02、
`qwen3-embedding-8b` が $0.01、`gemini-embedding-2` が $0.20 である。
単価は `qwen3-embedding-8b` の方が安いが、この日本語コーパスでは
`text-embedding-3-small` が精度・レイテンシともに上回った。
`gemini-embedding-2`（3072 次元、1M 入力トークン $0.20）も
同一サブセットで計測し、`hit@1=1.0` / `nDCG@10=1.0` だった。

### 最新の実測値と旧値の扱い

全量クラウドは 91 ファイル／1380 チャンク（`ingest.iter_chunks` 委譲後）、
`qrels.json` 9 クエリ、repeat=1 で計測した。
`hit@1=0.7778`（query-level）/ `set_recall@1=0.7222`（集合 recall）/
`hit@10=1.0` / `nDCG@10=0.8597`（現行定義＝上位10件・ファイル1回）
だった（`eval-results-cloud-full.json`）。
生成モデル `deepseek-v4.1-flash` による rerank / 回答生成は
9 クエリで動作を確認した（`rerank_query_count=9`、
`rerank_ndcg_at_5=0.8819`）。

| モデル | definition | nDCG@10 | rerank nDCG@5 (n=9) |
| --- | --- | --- | --- |
| `qwen3-embedding-8b`（クラウド） | cloud_bench top10 | 0.8978 | — |
| `gemini-embedding-2`（クラウド） | cloud_bench top10 | 1.0 | — |
| `text-embedding-3-small`（クラウド・全量） | cloud_bench top10 | 0.8597 | 0.8819 |
| （旧値）`text-embedding-3-small`（クラウド・全量） | legacy 全順位 | 0.9034（旧 ndcg_at_10、全順位ベース） | — |

Qdrant 投入後の正式パイプラインでは hybrid が `nDCG@10=0.8681`
で最良だった（`eval-results-cloud-pipeline.json`、
88 ファイル／214 チャンク）。`cloud_bench.py` の nDCG は
上位10件・ファイル単位1回の定義へ揃えたため、旧 `ndcg_at_10=0.9034` は
全順位ベースの値であり `evaluate.py` の同名指標と直接比較できない。
rerank の `nDCG@5` は旧 0.3608 が ideal のファイル重複カウントバグに
よる過小評価だったため、修正後の値で置き換えた。

全量クラウドと全量 Ruri はコーパスとチャンク条件が異なる
（91 ファイル／1380 チャンクと 88 ファイル／2200 チャンク）ため、
両者の nDCG に優劣は付けられない。レイテンシも計測対象が異なる
（クラウドは埋め込み平均 42.3 秒、ローカルは hybrid-rerank の検索平均）
ため、単純比較しない。

## 別ドメインベンチマーク（2026-11-26 時点）

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

### 実測結果（2026-11-26、ADR 6 ファイル／50 チャンク）

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

## 制限と解釈の注意

### 検索・一致の限界

- `keyword` は日本語を空白・記号で分割するため、「起動トークン」の
  ような複合語の部分一致が弱い。形態素解析はしていない。
- 一致判定は小文字化のみ。同義語・表記ゆれ・送り仮名の差は吸収しない。
- `hybrid` は片方の検索が失敗してももう片方で継続するが、結果が
  片面だけになることがある。
- `dense` / 再ランク / 回答生成は Ollama が応答しないと失敗する。
- `ingest.py` は既定で毎回すべてを再埋め込み・再投入する。
  `RAG_INGEST_RESUME=1` でチェックポイントから再開できる。
  現状は 88 ファイル／2200 チャンク（全量）で数分かかる。
- Qdrant のベクトル次元は `RAG_VECTOR_SIZE`（既定 2048、
  `qwen2.5:3b` の実測値）。モデルを変更する場合は同値を変更し
  コレクションを作り直す。point ID は blake2b 64bit のため、
  旧 ID のポイントが残る場合はコレクションを削除してから再投入する。
- チャンク本文は `CHUNK_SIZE` 以内だが、見出し行を先頭に付けるため
  `payload.text` 全体は見出し長だけ長くなる。

### 評価指標の限界

- `evaluate.py` の指標は真の `recall@k` ではない。
  `source_keyword_hit@3` は上位3件の `source` に expected 語が1つでも
  含まれるかの any-match、`content_keyword_hit_all_hits` は全ヒットの
  `text` に対する any-match で、分母となる取得深度が異なる。
  `avg_distinct_source_ratio_at3` は出典の多様性であり検索品質ではない。
- レイテンシは `RAG_EVAL_REPEAT`（既定 1）回の計測で、1 回のときは
  ウォームアップなし（実運用に近い）。2 以上ならウォームアップ 1 回＋N 回の
  `avg` / `median` / `min` / `max` / `stdev` / `samples` /
  `cold_latency_sec` を記録する。性能表の値は `RAG_EVAL_REPEAT=10`
  （60サンプル）で再計測済み。
- held-out は 9 クエリに正解ファイル名ラベルを付け、query-level hit@k と
  nDCG@3/10 を集計する。正解ファイル全体の回収割合を測る集合 recall ではない。

### 一般化の限界

- 簡易評価は 6 ファイル／81 チャンクのサブセット・6 クエリで実施。
  本番の 88 ファイル／2200 チャンクや一般 RAG 方式の比較ではない。
  held-out 評価は 9 クエリの正解ラベル付きだがドメイン内のみで、
  一般化には別コーパスでの検証が必要。
- ドメイン内ベースラインは簡易 6 クエリ＋held-out 9 クエリの
  計 15 クエリで、いずれも `passkey-gateway/docs` のドメイン内コーパスと
  その中の実在ファイル名を正解ラベルに使う。
- 別ドメイン（openrustbooks の ADR 6 ファイル／50 チャンク）でも
  同一モードで再計測した。`keyword` が `recall@1` から 1.0 /
  `nDCG` も 1.0 と明確に最良で、`dense` は 0.8333 / 0.5774、
  `hybrid` は 1.0 / 0.8552、`hybrid-rerank` は 1.0 / 0.7936 だった。
  ファイル名・見出しの語が一致しやすい構造のため keyword が有利な点は
  ドメイン内でも共通している。
- 正解ラベルも人手で作成しているため完全な客観性は無く、
  作成者の選択バイアスが残る。数値の差は「ドメイン内 15 クエリでは
  見えなかった順位の入れ替わり」を示す仮説であり、
  一般の RAG 方式の優劣を確定させるものではない。

### 外部送信とコストの限界

- クラウド有効化時は質問文・チャンク本文・ベクトルが外部 API へ送信され、
  token 課金が発生する。キーの有効性・レート制限・コストは未検証で、
  長時間の連続評価は避ける。
- 共有 `dsh.env` 全体を `set -a; source ...; set +a` で export すると
  他サービスのキーまで子プロセスへ継承されるリスクがあるため、
  必要なキーだけを明示的に読み込む。
- `RAG_CLOUD_CHAT_ENABLED=1` は `RAG_CLOUD_ENABLED=1` が前提で、
  単独では設定検証（fail-closed）で停止する。
- 未知の埋め込みモデルは `RAG_CLOUD_EMBED_DIMENSION`
  （期待次元を明示。未指定なら fail-closed）が必要で、複数形
  `RAG_CLOUD_DIMENSIONS` は移行期互換。`encoding_format` は
  API 仕様上存在するが本クライアント未対応（リクエストへ渡さない）。

## 検証のヒント

親エージェントが lint と OKF 検証を実行済み（lint 0 / OKF valid）。
子エージェントは shell 実行を禁止されているため、ここでは再確認用の
コマンドのみを示す。

```bash
markdownlint-cli2 rag_poc/docs/*.md
rust-script /home/koppe/.agents/skills/okf/scripts/okf_validate.rs rag_poc/docs/explanation.md
```

あわせてフロントマターの `verified[0].at >= generated.at` と、
本文の最初の見出しが `##` であることを確認する。

## 参照

- [rag_poc/README.md](../README.md) — 本文書の根拠元（現行 README）
- [rag_poc/docs/index.md](index.md) — 詳細ドキュメントの索引
- 分割前 README スナップショット（`/tmp/README-full-backup.md`）:
  `## 構成図（テキスト）` / `## 検索モード`（設計背景のみ） /
  `## 既知の制限` / `## モデル比較実験` / `## 別ドメインベンチマーク` /
  `## 入力と設定の検証`
