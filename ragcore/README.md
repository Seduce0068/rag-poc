---
type: readme
title: ragcore — RAG PoC 指標計算ライブラリ / CLI
description: Python 版 RAG PoC の指標計算を unsafe なしで再実装した Rust crate。
tags: [rust, rag, cli]
---

## 概要

`ragcore` は `rag_poc/ingest.py` / `rag_poc/evaluate.py` / `rag_poc/compare_embeddings.py` の
指標計算を Rust で再実装した crate です。`unsafe` は使用していません。
依存は `serde` (derive) / `serde_json` / `sha2` / `blake2` のみです。

## CLI の使い方

リポジトリルートで次を実行します。入力は 1 行の JSON オブジェクト、出力は結果の JSON 値です。

```bash
echo '{"op":"chunk","text":"あいうえおかきく","chunk_size":3,"overlap":1}' \
  | cargo run --quiet --manifest-path rag_poc/ragcore/Cargo.toml
# => ["あいう","うえお","おかき","きく"]
```

失敗時は `{"error":"..."}` を stdout に出力し、終了コード 1 で終了します。
ライブラリとしては `use ragcore::...;` で各関数を直接呼び出せます。

## op 一覧

| op | 必須入力フィールド | 出力 |
| --- | --- | --- |
| `chunk` | `text`, `chunk_size`, `overlap` | チャンク文字列の配列 |
| `cosine` | `a`, `b` (数値配列) | コサイン類似度 (数値) |
| `cosine_batch` | `pairs` (ベクトルペア配列) | コサイン類似度の数値配列（入力順・同件数、丸めなし） |
| `distribution` | `scores` (数値配列) | `{"min","max","mean","stdev"}` |
| `recall` | `relevant`, `retrieved` (文字列配列), `k` | hit@k (0.0 / 1.0) |
| `ndcg` | `relevant`, `retrieved` (文字列配列), `k` | nDCG@k (6 桁丸め) |
| `latency` | `samples` (数値配列) | `{"count","avg","median","min","max","stdev"}` |
| `fingerprint` | `paths` (後述) | `{"count","sha256"}` |
| `point_id` | `source` (文字列), `chunk_index` (整数) | u64 ID |

`latency` は `samples` 配列が必須です。Python の
`evaluate._aggregate_latency()` が対応する旧形式 record（`samples` 無しの
`count` / `avg` / `median` / `min` / `max` / `stdev`）のフォールバックは
Rust CLI 側には無く、`samples` 欠落時は `{"error":"..."}` になります。

`fingerprint` の `paths` は `{"path":"a.md","content":"本文"}` の配列です。
`content` の代わりに `text` (文字列) または `bytes` (0..255 の整数配列) も使えます。
パス文字列でソートしてから本文バイトを連結し、SHA-256 hex を計算します。

`cosine_batch` の `pairs` は `[[a, b], ...]` 形式の数値ベクトルペア配列です。
各ペアのコサイン類似度を入力順に数値配列で返します（件数は `pairs` と同じ）。
`cosine` とは異なり**丸めは行いません**。丸めが必要な場合は呼び出し側（`compare_embeddings.py`
では `round(..., 6)`）で行います。1 件でも次元不一致・空ベクトルがある場合は
結果を 1 件も返さず `{"error":"..."}` を出力します。

入力例:

```bash
echo '{"op":"point_id","source":"doc.md","chunk_index":0}' \
  | cargo run --quiet --manifest-path rag_poc/ragcore/Cargo.toml
echo '{"op":"distribution","scores":[1,2,3,4]}' \
  | cargo run --quiet --manifest-path rag_poc/ragcore/Cargo.toml
echo '{"op":"cosine_batch","pairs":[[[1,0],[1,0]],[[1,0],[0,1]]]}' \
  | cargo run --quiet --manifest-path rag_poc/ragcore/Cargo.toml
# => [1.0,0.0]
echo '{"op":"recall","relevant":["doc.md"],"retrieved":["dir/doc.md"],"k":1}' \
  | cargo run --quiet --manifest-path rag_poc/ragcore/Cargo.toml
```

## 検証コマンド

以下のコマンドを親エージェントが実行する。

```bash
cargo fmt --manifest-path rag_poc/ragcore/Cargo.toml -- --check
cargo test --manifest-path rag_poc/ragcore/Cargo.toml
cargo clippy --manifest-path rag_poc/ragcore/Cargo.toml -- -D warnings
```
