---
type: Reference
title: ragcore — Rust のみの RAG ライブラリ / CLI
description: RAG の投入・検索・評価を unsafe なしで実装した Rust crate。
tags: [rust, rag, cli]
generated:
  by: "team:rag-poc"
  at: "2026-11-28T01:00:00+09:00"
status: "stable"
stale_after: "2027-02-28T00:00:00+09:00"
---

## 概要

`ragcore` は llama.cpp (llama-server) + Qdrant を使う RAG を Rust だけで実装した crate です。
`unsafe` は使用していません。依存は `serde` / `serde_json` / `sha2` / `blake2` /
`reqwest` / `clap` / `url` / `chrono` です。

## CLI サブコマンド

```bash
# 投入
ragcore/target/debug/ragcore ingest --corpus-dir DIR --collection NAME [--recreate --yes]

# 検索
ragcore/target/debug/ragcore query '質問' --mode hybrid --top-k 10 --json

# 評価
ragcore/target/debug/ragcore evaluate --qrels qrels.json --output eval-results.json
```

stdin JSON の op モードも後方互換で残しています。

`hybrid-rerank` は `hybrid` と同一動作（LLM 再ランク未実装）です。
回答生成・リモート接続制限・コーパス範囲制限も未実装です。
クラウド埋め込みは opt-in で利用可能ですが、リトライ、回答生成（chat）、
ハイブリッド検索の部分失敗の構造化記録は未実装です。
`RAG_CLOUD_CHAT_ENABLED` は設定を受け付けますが no-op です。

```bash
echo '{"op":"chunk","text":"あいうえおかきく","chunk_size":3,"overlap":1}' \
  | ragcore/target/debug/ragcore
# => ["あいう","うえお","おかき","きく"]
```

失敗時は `{"error":"..."}` を stdout に出力し、終了コード 1 で終了します。

## op 一覧

| op | 必須入力フィールド | 出力 |
| --- | --- | --- |
| `chunk` | `text`, `chunk_size`, `overlap` | チャンク文字列の配列 |
| `cosine` | `a`, `b` (数値配列) | コサイン類似度 (数値) |
| `cosine_batch` | `pairs` (ベクトルペア配列) | コサイン類似度の数値配列 |
| `distribution` | `scores` (数値配列) | `{"min","max","mean","stdev"}` |
| `recall` | `relevant`, `retrieved`, `k` | hit@k (0.0 / 1.0) |
| `ndcg` | `relevant`, `retrieved`, `k` | nDCG@k (6 桁丸め) |
| `latency` | `samples` (数値配列) | `{"count","avg","median","min","max","stdev"}` |
| `fingerprint` | `paths` | `{"count","sha256"}` |
| `point_id` | `source`, `chunk_index` | u64 ID |
| `check_eval_results` | `paths` | `{"ok","checked","failed","diagnostics"}` |
| `rrf_fuse` | `rankings`, `top_k` | RRF 統合後の配列 |

`check_eval_results` は成果物の必須キー・禁止キー・`generated_at` の UTC を検査し、
diagnostics にはファイル名とキー名のみ含めます（値は含めない）。

## モジュール

| ファイル | 役割 |
| --- | --- |
| `config.rs` | 環境変数と検証 |
| `llama.rs` | llama-server の OpenAI 互換 `/v1/embeddings` |
| `cloud.rs` | opencodeGO の `/embeddings`（`RAG_CLOUD_ENABLED=1` のときだけ生成） |
| `qdrant.rs` | コレクション／points API |
| `search.rs` | keyword / RRF / rerank の純粋計算 |
| `ingest.rs` | コーパス走査・投入 |
| `query.rs` | 検索モードの統合 |
| `evaluate.rs` | qrels 評価と成果物生成 |
| `lib.rs` | 指標計算 op |

## 検証コマンド

```bash
cargo fmt --manifest-path ragcore/Cargo.toml -- --check
cargo test --manifest-path ragcore/Cargo.toml
cargo clippy --manifest-path ragcore/Cargo.toml -- -D warnings -D clippy::undocumented_unsafe_blocks
```
