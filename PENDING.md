---
type: Index
title: "rag_poc 承認待ち3件の完了記録インデックス"
description: "ユーザー承認が必要だった rag_poc タスク3件の完了記録と、公開済みリポジトリの継続運用を一覧化"
tags:
  - rag
  - pending
  - approval
generated:
  by: "lead/swarm-coding"
  at: "2026-11-26T20:41:08+09:00"
verified:
  - by: "lead/r6-aggregation"
    at: "2026-11-26T20:41:08+09:00"
status: "stable"
---

## 目的

`todo.md` の RAG バックログのうち、**ユーザー承認が必要だった**項目を
1 箇所に集約する。3件はすべて完了しており、今後は公開済みリポジトリの
継続運用を記録するインデックスである。承認を求める依頼ではない。

> 2026-11-26 20:35 更新: ユーザー承認により下記3件はすべて完了。
> 詳細は `todo.md` の完了項目と git 履歴（4 コミット）を参照。
> リポジトリは public 化済み。公開前スキャン（Betterleaks 全履歴 hit zero /
> 画像0件 / 実識別子なし）は完了している。
> 次アクションは公開後の継続運用（変更時の再スキャンと CI ゲート維持）。

## 完了した3件

| # | 項目 | 必要な承認 | 実施したコマンド / 手順 | 現状 |
| --- | --- | --- | --- | --- |
| 1 | `rag_poc` の git 管理と `.gitignore` 追加 | 版管理方針（成果物をコミットするか、最新コピーのみとするか）の決定 | `git init` → `.gitignore`（`__pycache__/` / `*.pyc` / `ragcore/target/` / `.recreate-audit.json` / `.ingest-checkpoint.json`）→ 成果物の選別 → 初回コミット | 完了。git 管理と `.gitignore` 追加済み（public リポジトリで運用中） |
| 2 | クラウド埋め込みの実計測経路 | API キー（シークレット管理）と外部送信の明示承認 | `rust_bridge` とは別の専用計測経路を追加し、API キーを `dsh.env` から読み込む。端点オプトインは既定拒否を維持（`RAG_ALLOW_REMOTE_ENDPOINTS=1` + `RAG_ALLOWED_ENDPOINT_HOSTS` + https） | 完了。`cloud_bench.py` / `cloud_ingest.py` で実計測経路を追加済み |
| 3 | CI ゲートの追加（#1 に依存） | #1 の git 化方針が前提 | `.github/workflows/ci.yml` を追加し、`Seduce0068/workflows` の `rust-gates.yml@v1` を呼び出す。Python は `python3 test_rag.py`、JSON 必須キー検査は成果物別テーブルで実施 | 完了。CI 追加済み（JSON ゲートを成果物別必須キー表へ更新） |

## 他ドキュメントとの対応

件数や状態が文書ごとに異なるため、本インデックスを完了記録の入口とする。
差分は参照先の更新待ちであり、ここでは事実のみを記録する。

- [pending-root-actions.md](../passkey-gateway/docs/pending-root-actions.md) — API キーローテーション残り1件（全体スコープ）。#2 のシークレット管理と間接的に関連。
- [STATUS.md](../passkey-gateway/docs/STATUS.md) — 状況ダッシュボード。rag_poc 固有の承認待ちは載らない。
- [FINAL-SUMMARY.md](../passkey-gateway/docs/FINAL-SUMMARY.md) — 環境全体の権限拡大保留の記録。
- `todo.md` — RAG セクションの完了項目が正本。本インデックスはその抄録。

## 次アクション（公開済みリポジトリの継続運用）

リポジトリは public のため、変更のたびに以下を順に実行する。Betterleaks は
HEAD だけでなく全コミットを対象にし、画像を含む場合は OCR も併用する。
履歴改変後は `.betterleaksignore` の fingerprint を再生成してから再走査する。

```bash
echo '--- 1. Betterleaks 全履歴スキャン（ローカル / 公開リポジトリ） ---'
betterleaks git

echo '--- 2. ローカルゲート（Rust gates 相当） ---'
cargo fmt --manifest-path rag_poc/ragcore/Cargo.toml -- --check
cargo check --manifest-path rag_poc/ragcore/Cargo.toml
cargo test --manifest-path rag_poc/ragcore/Cargo.toml
cargo clippy --manifest-path rag_poc/ragcore/Cargo.toml -- -D warnings -D clippy::undocumented_unsafe_blocks

echo '--- 3. Python テスト（Rust 必須ゲート含む） ---'
python3 rag_poc/test_rag.py
RAG_REQUIRE_RUST=1 python3 rag_poc/test_rag.py

echo '--- 4. 比較成果物の再生成と JSON ゲート ---'
python3 rag_poc/compare_cloud_local.py --output rag_poc/cloud-local-comparison.json

echo '--- 5. push 後の CI ゲート通過確認（GitHub CLI） ---'
git diff --check
git push
gh run list --limit 5
gh run watch --exit-status
```

`gh run watch` が成功するまでマージしない。CI は `Seduce0068/workflows` の
`rust-gates.yml@v1` へ委譲しているため、ローカルの 2 / 3 と同じゲートが走る。

## 完了記録

1. 3件はいずれも完了（`todo.md` の完了項目と git 履歴を参照）。
2. 破壊的操作（git 履歴改変 / force push）は今後も事前確認が必要。
3. 次アクションは公開済みリポジトリの継続運用。変更時に Betterleaks 全履歴
   スキャンと CI ゲート（Rust gates + JSON 必須キー）の通過を確認する。
   具体コマンドは「次アクション（公開済みリポジトリの継続運用）」を参照。
4. 進捗は `todo.md` と本インデックスの「現状」列を合わせて更新する。
