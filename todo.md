---
type: Todo
title: "rag_poc 残タスク（バックログ補充 2026-11-28）"
description: "swarm-todo-discovery / swarm-project-review の R6 集約による残タスク一覧"
tags:
  - todo
  - rag
  - rust
generated:
  by: "lead/cross-skill-r6"
  at: "2026-11-28T17:20:00+09:00"
verified: []
status: draft
---

## 進行中

（なし）

## 未着手

### 優先度: 高

- [ ] **`--recreate` の破壊的順序を見直し（staging 切り替え）** — `ragcore/src/ingest.rs:206-210` / R2-ER / 削除後に失敗すると旧数据在失う
- [ ] **検証済み Endpoint 型を全 HTTP アダプタへ適用** — `ragcore/src/config.rs:298-363`, `cloud.rs:515-535` / R1-AR, R3-IV / ライブラリ経路でリモート opt-in を迂回可能

### 優先度: 中

- [ ] **埋め込みレスポンスの `data[].index` 検証（cloud / llama）** — `cloud.rs:103-139`, `llama.rs:157-179` / R2-ER / out-of-order で文書とベクトルが入れ替わる
- [ ] **LLM スコア不全時の黙った 0 点化を廃止** — `search.rs:341-346,493-496` / R3-IV, R4-OB / 一部スコアで順序が変わる
- [ ] **時限・件数の入力検証を `Config::validate` と `search()` へ集約** — `config.rs:254-256`, `query.rs:262-300` / R3-IV, R2-FG / 負 timeout・空質問・rerank_max_docs=0 が通る
- [ ] **citations_in_range を fail-close 化** — `query.rs:197-220` / R3-IV / `[ 4 ]` や巨大整数が検証を迂回
- [ ] **docs と Rust CLI の差分を解消** — `docs/reference.md:104-107,122,150-165`, `main.rs:44-50` / R2-FG / 未実装オプションと旧 env が残る
- [ ] **Usage を SearchResult / CLI JSON へ公開** — `cloud.rs:496-552`, `query.rs:82-172` / R4-OB / コスト・トークンを観測できない
- [ ] **型付き RagError（error_code / retryable / attempts）を導入** — `lib.rs`, `query.rs:124-131`, `qdrant.rs:81-86` / R1-AR, R4-OB / 障害分類が文字列依存
- [ ] **cloud retry の構造化ログを追加** — `cloud.rs:567-610,639-688` / R4-OB / 再試行が完全にサイレント
- [ ] **成功回答へのエラー用マスカー適用をやめ、ログ境界を共通化** — `query.rs:244-247`, `main.rs:63-65` / R4-OB / 正当な URL が壊れ、パスが漏れる
- [ ] **evaluate::run の統合テストと CI op カバレッジを追加** — `evaluate.rs:43-155`, `ci.yml:25-26` / R3-TQ / 評価本体が未テスト
- [ ] **lib.rs をファサードへ縮小しドメインモジュールへ分離** — `lib.rs:10-591` / R1-AR / 責務と変更波及が不明確
- [ ] **`fingerprint` にパス・本文境界を含める** — `lib.rs:248-257` / R3-IV / 本文再割り当てを検出できない

- [ ] **クイックスタートのコーパス指定を明示** — `README.md:17-33`, `config.rs:230`, `ingest.rs:187-196` / R5-DC / 既定パスが実在せず初回 ingest が失敗する
- [ ] **rerank 失敗と回答生成失敗の説明を分離** — `README.md:46-51`, `query.rs:292-315` / R5-DC / errors と answer_error の別経路が誤解される
- [ ] **OKF メタデータ（status / stale_after / verified.at）を現行スキーマへ修正** — `ragcore/README.md:9`, `PENDING.md:11-16`, `SECURITY.md:12-13` / R5-DC
- [ ] **PENDING.md の現行 CI 説明を更新** — `PENDING.md:34-38,60-76`, `ci.yml:16-29` / R5-DC / Python ゲートが現行と誤記
- [ ] **`docs/reference.md` の usage 記録に関する矛盾を解消** — `docs/reference.md:382-386,408-431` / R5-DC
- [ ] **`unreachable!()` を明示エラーへ置換** — `query.rs:320` / R1-ST / 列挙追加時に panic
- [ ] **`scripts/__pycache__/*.pyc` の孤立バイナリを削除** — `scripts/__pycache__/check_eval_results.cpython-314.pyc` / R1-ST / ソースが公開リポジトリに無い
- [ ] **CI に `cargo fmt --check` と `cargo clippy -D warnings` を追加** — `ci.yml:19-27`, `CONTRIBUTING.md:29-36` / R4-DP
- [ ] **CI に cargo audit / cargo deny を追加** — `ci.yml:16-29` / R4-DP / 未保守依存を検出できない
- [ ] **reqwest を `rustls-tls` へ変更し OpenSSL 依存を排除** — `Cargo.toml:12` / R4-DP
- [ ] **sha2 / blake2 の 0.11 移行を計画** — `Cargo.toml:10-11`, `lib.rs:269-278` / R4-DP / API 変更を伴うため分離 PR
- [ ] **CLI と op モードのエラー出力先・形式を統一** — `main.rs:63-65,174-176` / R5-VU

### 優先度: 低

- [ ] **RFC3339 UTC 検証で実在日時を確認し `-00:00` を拒否** — `lib.rs:442-497` / R3-IV
- [ ] **cosine / distribution / latency_stats の非有限値を拒否** — `lib.rs:97-115,213-236` / R3-IV
- [ ] **`docs/index.md` を予約ファイル形式へ修正（frontmatter 除去・H1 単一）** — `docs/index.md:1-22` / R5-DC
- [ ] **README / docs の存在しないアンカーを修正** — `docs/index.md:39-44`, `docs/explanation.md:482-485` / R5-DC
- [ ] **`check_eval_results` の読み取り失敗にファイル名と原因分類を含める** — `main.rs:330-334` / R5-VU / 値は出さないまま診断性を上げる
- [ ] **未使用 pub 関数（`count_exact` / `usage` / Usage ヘルパー）を整理** — `qdrant.rs:173`, `cloud.rs:693,340` / R1-ST / 削除または `pub(crate)` 化
- [ ] **`Cargo.toml` に `rust-version` と `license` を追加** — `Cargo.toml:1-15` / R4-DP
- [ ] **`check_eval_results` のフィクスチャを明示的に固定** — `ci.yml:26`, `.gitignore` / R4-DP
- [ ] **pre-commit に fmt / clippy を追加** — `.pre-commit-config.yaml:1-6` / R4-DP

## 完了

- [x] ingest の読み取り・チャンク化失敗を fail-close 化（112 tests）
- [x] `chrono` を `time` へ置換し RUSTSEC-2026-0543 を解消（cargo audit 通過 / 109 tests）
- [x] Qdrant レスポンスの構造検証と truncated 伝播（109 tests）
- [x] 範囲外 citation を `answer_error` に記録（`c7bb68f` / 105 tests / CI success）
- [x] ローカル llama.cpp chat と回答検証を追加（`50c2646`）
- [x] コーパス範囲制限と公開ゲート fail-close を追加（`a65f2f6`）
- [x] hybrid-rerank・回答生成・cloud chat を追加（`e27f6be`）

## メモ

- 観点: ST / FG / TQ / DP / VU / AR / ER / IV / OB / DC（10 観点並列）
- モデル: discovery = `beam-free`、review = `hungerg-free`（いずれも 0 USD）
- 同一根拠は統合済み（例: Qdrant 部分結果は ER/AR/OB を 1 件に集約）
- 推測_only の指摘は採用せず、根拠 `path:line` を親が確認できた項目のみ記載
