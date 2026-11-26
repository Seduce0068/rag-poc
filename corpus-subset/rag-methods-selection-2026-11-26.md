---
type: Report
title: "RAG方式の用語整理と選定指針"
description: "OKF・ベクトル検索・SQLite/Turso/libSQL・セマンティックランク・ハイブリッド検索の正体と、この環境での選び方"
tags:
  - rag
  - vector-search
  - hybrid-search
  - sqlite
  - turso
  - okf
generated:
  by: "lead/swarm-unlimited"
  at: "2026-11-26T00:06:48+09:00"
verified:
  - by: "lead/r4-adversarial"
    at: "2026-11-26T00:14:02+09:00"
  - by: "lead/context7"
    at: "2026-11-26T00:22:08+09:00"
  - by: "lead/swarm-research-r2-r4"
    at: "2026-11-26T01:16:41+09:00"
  - by: "lead/context7-sentence-transformers"
    at: "2026-11-26T01:08:50+09:00"
  - by: "lead/runtime-proof"
    at: "2026-11-26T01:34:13+09:00"
  - by: "lead/runtime-proof-turso"
    at: "2026-11-26T01:38:21+09:00"
status: "stable"
---

## 結論（先に答え）

挙げられた用語は「同じ軸の競合方式」ではなく、**RAG パイプラインの別の層にある部品**です。
混乱の原因はここにあります。

この環境（jarvis が PostgreSQL と Qdrant を常駐）での推奨は次のとおりです。

1. まず Qdrant に dense ベクトル 1 本でコレクションを作る。
2. 検索品質が足りなければ BM25 sparse を追加してハイブリッド化する（RRF で統合）。
3. それでも足りなければ、検索後に Cross-Encoder 系 reranker を適用する。
4. OKF は「検索方式」ではなくドキュメントのメタデータ形式として扱う。
5. Turso / libSQL は同期・エッジ用途の選択肢であり、常駐単一マシンでは不要。

## 用語の整理（どの層の話か）

| 層 | 用語 | 役割 | 検索方式か |
| --- | --- | --- | --- |
| ドキュメント表現 | OKF | YAML frontmatter + Markdown のナレッジ形式 | いいえ |
| 取得（候補生成） | ベクトル検索 | 埋め込みの類似度で意味検索 | はい |
| 取得（候補生成） | SQLite FTS5 / BM25 | トークン一致のキーワード検索 | はい |
| 保存・配信 | SQLite / Turso / libSQL | データの保存と配信基盤 | いいえ（検索機能は持つ） |
| 絞り込み | セマンティックランク | 検索結果の再ランキング（rerank） | いいえ（後段処理） |
| 統合 | ハイブリッド検索 | ベクトル + BM25 を RRF 等で統合 | はい（組み合わせ） |

「どれがいいか」は、この表の**行ごとに答えが違う**というのが正確な答えです。

## 各方式の正体

### OKF は検索方式ではない

Open Knowledge Format は「ナレッジのメタデータ・コンテキスト・キュレーション済み知見」のための
オープン形式で、仕様は「ストレージ・配信・クエリ基盤を規定しない」ことを non-goal に挙げています。
検索インデックスは consumer（読み手）の 1 例として列挙されるだけで、検索アルゴリズム自体は
規定しません（仕様全文に vector / rerank / semantic search は出現しません）。

RAG では `type` をフィルタ、`description` をスニペット、`tags` を候補絞り込み、
`sources[].id` を出典表示に使えます。ただし索引やメタデータは候補選択の材料であり根拠では
ないため、回答の根拠は必ず本文チャンクを開いて確認します。

### ベクトル検索は「意味」を拾う

埋め込み（dense vector）は語ではなく意味を表現します。Qdrant の公式表現では
「Dense vectors can capture the meaning, not the words used in a text」です。
全件の距離計算を避けるため ANN（HNSW など）で候補のみを探索します。
距離指標は Cosine / Dot / Euclid / Manhattan があり、埋め込みモデルの学習指標に
合わせて選びます。

弱点は完全一致です。商品コード・エラー文字列・日付・固有名詞では BM25 側が有利というのが
Qdrant / Pinecone / Azure の共通した記載です。

### SQLite FTS5 はキーワード検索

FTS5 はトークン列の部分列一致で文書を探し、BM25（フレーズ頻度 × IDF）で並べます。
公式定義では全文検索を「検索語のインスタンスを含む文書の抽出」としており、
語彙的検索です。
公式ページに semantic search / embedding の記述はなく、「意味検索不可」は明記ではなく
トークン一致機構からの帰結です。

日本語では unicode61 が連続文字を 1 トークンにするため分かち書き課題があります。
組み込みトークナイザーは unicode61 / ascii / porter / trigram の4種のみで、
日本語特化はありません。trigram（SQLite 3.34.0 以降）は部分文字列一致で
回避できますが、3 文字未満はマッチしません。
外部トークナイザー（fts5_api の xCreateTokenizer で登録）なら辞書ベースの
分かち書きが可能です。2026-11-26 時点で活発なのは lindera-sqlite v7.0.0
（2026-11-23）と sqlite-vaporetto v0.4.0（2026-04）です。
古い fts5_mecab は最終更新が 2019 年で保守状況は未確認。

### Turso / libSQL は保存・同期の層

- **libSQL**: SQLite のオープンソースフォーク。ファイル形式・API 互換。シングルライターを継承。
- **Turso Database**: SQLite 互換だが Rust で書かれた別実装。並行書き込みと双方向同期に対応。
- **Turso Cloud**: 両エンジンをホストするマネージド基盤。

現行の turso-docs では、Turso と libSQL の両方で拡張なしにベクトル検索が
使えると記載されます（`Turso and libSQL enable vector search capability
without an extension`）。ただしエンジンで API が異なります。

- **libSQL**: 型は `FLOAT32` / `F32_BLOB`（旧表記 `FLOAT32_BLOB` は現行
  ドキュメントで未確認）。`vector32()` 変換、`vector_distance_cos`、
  DiskANN による ANN インデックス（`libsql_vector_idx` + `vector_top_k`）。
- **Turso Database（Rust 実装）**: BLOB 列 + `vector32()` / `vector64()`。
  `Vector indexes are not yet supported` と明記され、探索は brute-force。
  `libsql_vector_idx` / `vector_top_k` の同等サポートは未確認。

「Turso 側だけ ✓ / libSQL 側 ✗」という比較表は現在の一次情報で確認できず、
両者とも native だが実装が異なる、として扱います。
価値が出るのはローカルレプリカで読み取りをローカルに置き、書き込みを
クラウドのプライマリへ送るエッジ／モバイル用途です。新規の同期プロジェクトには
Embedded Replicas より、ローカル読み書き + `push()` / `pull()` の
`turso::sync` が推奨されています。常駐単一マシンでは不要です。

### セマンティックランクは rerank であって検索ではない

Azure の定義では semantic ranker は「BM25 または RRF でスコアリングされた初期結果セットに
二次ランキングを加える」機能で、「コーパス全体を再検索することはできない」と明記されています。
Sentence Transformers も Cross Encoder を reranker と呼び、top-k 結果の再ランクに使うと
説明します。

Bi-Encoder（埋め込み）は候補生成、Cross-Encoder はペア評価、と役割が異なります。
Cross-Encoder は精度が高い一方、クエリと文書のペアごとに計算するため遅く、
埋め込みを事前計算してインデックスできません。必ず「検索で top-100 に絞ってから
rerank」の二段構えにします。

### ハイブリッド検索は補完の統合

ベクトルは意味、BM25 は語の完全一致に強く、互いの穴を埋めます。
統合には RRF（Reciprocal Rank Fusion）が一般的です。
RRF はスコアの絶対値を捨てて順位だけを使うため、cosine 0.7 と BM25 12.4 のような
スケール差を比較せずに統合できます。原典では `k = 60` が pilot で固定され、
単一システムより平均 4〜5% 良好だったと報告されています。

ただし常に勝つわけではありません。Qdrant の公開ベンチでは 5 データセット中 4 つで
単独 retriever を上回りましたが、DBPedia-entity では dense 単独が上回りました。
「どちらが勝つかは自データで測る」が原則です。

## この環境での選定（判断木）

1. **知識ベースが 20 万トークン（約 500 ページ）未満**なら RAG を作らずプロンプトに
   丸ごと入れる（Anthropic 公式推奨）。
2. 収まらないなら、常駐 Qdrant に dense コレクションを作る。新規ベクトル DB は増やさない。
3. キーワード完全一致（コード・ID・固有名詞）が必要なら BM25 sparse を追加して
   RRF ハイブリッドにする。
4. キーワード側だけ先に試す選択もあり。Qdrant は「BM25 から始め、学習モデルを
   計測してから採用せよ」と案内しています。
5. 品質が足りなければ top-100 → rerank で top-3 に絞る。効果は held-out 検証で確認する。
6. 単一マシンで Qdrant が使えない場合の代替は SQLite + FTS5 + sqlite-vec（1M ベクトル
   付近で遅延限界）。Turso / libSQL は端末・エッジ同期が必要になった時点で検討する。

段階ごとに `recall@k` / `nDCG@10` を測り、改善が確認できたら次の段へ進みます。
Qdrant の言葉を借りれば「Measure whether the gain is worth the cost instead of guessing」。

## 埋め込みモデルの候補（日本語）

| モデル | 特徴 | 注意 |
| --- | --- | --- |
| sarashina-v2 / ruri-v3 | JMTEB 76.38 / 75.85（2025-09-30） | 日本語特化 |
| BGE-M3 | 100+ 言語、8192 トークン、1024 次元、MIT | 3 方式対応、hybrid 推奨 |
| multilingual-e5-large | JMTEB 70.67 | e5-large-v2 は英語専用なので日本語では使わない |
| gte-multilingual-base | 70+ 言語、305M、768 次元、Apache-2.0 | 高速・軽量 |
| text-embedding-3-large | API、3072 次元（切詰可）、8192 トークン | JMTEB 73.86 |
| jina-embeddings-v3 | 日本語を含む 30 言語、Matryoshka | CC BY-NC 4.0（商用は確認必須） |

JMTEB の数値はデータセット数（16 vs 28）と平均の取り方でカードごとに異なるため、
比較には出典と評価日を添えます。

## 検証結果

- `markdownlint-cli2`: 0 issues（MD013 の行長違反 3 件を修正済み）。
- 見出し構造: `##` 始まり、 Frontmatter の `title` を H1 として扱う構造で妥当。
- context7 MCP: `/websites/qdrant_tech`、`/websites/sqlite_docs`、
  `/tursodatabase/turso-docs`、`/huggingface/sentence-transformers` を再確認。
  Qdrant の BM25 優先と RRF/DBSF、SQLite の `bm25()` / trigram、
  Turso のネイティブベクトル検索、Sentence Transformers の
  CrossEncoder による top-k rerank とペアごとの計算（bi-encoder より遅い）は
  公式 quickstart と一致し、確認済み。
- R4 で context7 スニペットの鮮度問題を発見。`Vector search is a Turso
  extension` という一節は現行 raw ファイルに存在せず、旧スナップショットの
  可能性があるため、レポートの根拠は web_fetch で取得した現行原文に切り替えた。
- Runtime Proof（2026-11-26T01:27:44+09:00、Python `sqlite3` 3.46.1）:
  `CREATE VIRTUAL TABLE ... USING fts5(x)` と `tokenize='trigram'` は成功、
  `tokenize='icu'` は `no such tokenizer: icu` で失敗。`PRAGMA compile_options`
  に ICU はなし。この環境では ICU を直接使えないことを実証。
  `sqlite3` CLI は apt で、`turso` / `sqld` は公式スクリプトでインストール済み。
  バージョンは 3.46.1 / v1.0.34 / 0.24.32。
- Runtime Proof（2026-11-26T01:37:31+09:00、`turso dev` + sqld 0.24.32）:
  `vector32()` と `vector_distance_cos()` は動作。`F32_BLOB(2)` 型の列に
  `libsql_vector_idx` で CREATE INDEX し、`vector_top_k` で探索成功。
  素の `BLOB` 型では `unexpected vector column type: BLOB` で失敗。
- OKF バリデータ（ローカルスクリプト）: `status: stable` を不正と判定した。
  ただしグローバル `~/.agents/AGENTS.md` §8 と okf スキル本文は有効値を
  `draft` / `stable` / `deprecated` と規定しており、バリデータ側が旧仕様。
  正本に従い `stable` を維持した（バリデータのその他のチェックは通過）。

## 未確認事項

- r2-turso が参照した「Turso 側 ✓ / libSQL 側 ✗」の skill.md の表は特定できず、
  未確認。現行ドキュメントは両エンジンとも native vector search と明記し、
  実装（型名・インデックス方式）が異なるとしている。
- `libsql_vector_idx` / `vector_top_k` はこの環境の sqld 0.24.32 で動作確認済み
  （`F32_BLOB` 型必須）。ただし現行 CLI manual の「vector index 未サポート」
  は別エンジン（Rust 実装の Turso Database）を指す可能性があり、両者の
  対応関係は未確認。
- 「FTS5 に意味検索機能がない」は公式明記ではなく、トークン一致ベースの記述からの帰結。
- FTS5 で ICU トークナイザーを直接使えるかは、この環境（SQLite 3.46.1）では
  `no such tokenizer: icu` で不可と実証。ICU 有効ビルドでの利用は未確認。
- lindera-sqlite / sqlite-vaporetto の実運用での分かち書き精度は未測定。
- 各モデルの実測レイテンシと自データでの nDCG / recall（未測定）。

## 参照

- [OKF SPEC v0.3](https://github.com/open-knowledge-format/open-knowledge-format/blob/527e81aff5/SPEC.md)
- [Qdrant ドキュメント](https://qdrant.tech/documentation/)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [Turso ドキュメント](https://docs.turso.tech/)
- [Turso AI & Embeddings](https://raw.githubusercontent.com/tursodatabase/turso-docs/main/features/ai-and-embeddings.mdx)
- [Turso Database vector manual](https://raw.githubusercontent.com/tursodatabase/turso/main/cli/manuals/vector.md)
- [lindera-sqlite](https://github.com/lindera/lindera-sqlite)
- [sqlite-vaporetto](https://github.com/hotchpotch/sqlite-vaporetto)
- [Sentence Transformers Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- [Azure AI Search ハイブリッド検索](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval)
- [Cormack et al. Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
