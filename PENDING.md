---
type: Index
title: "rag_poc 承認待ち項目の再開インデックス"
description: "ユーザー承認が必要な rag_poc タスク3件と、他ドキュメントとの対応を一覧化"
tags:
  - rag
  - pending
  - approval
generated:
  by: "lead/swarm-coding"
  at: "2026-11-26T17:51:51+09:00"
verified:
  - by: "lead/r6-aggregation"
    at: "2026-11-26T17:51:51+09:00"
status: "stable"
---

## 目的

`todo.md` の RAG バックログのうち、**ユーザー承認がないと実行できない**
項目を 1 箇所に集約する。承認待ちを理由に開発ループを止めないための
再開手順インデックスであり、承認を求める依頼ではない。

> 2026-11-26 18:29 更新: ユーザー承認により下記3件はすべて対応済み。
> 詳細は `todo.md` の完了項目と git 履歴（4 コミット）を参照。
> public 化そのものは未判断で、公開前スキャン（Betterleaks 全履歴 hit zero /
> 画像0件 / 実識別子なし）は完了している。

## 承認待ち3件

| # | 項目 | 必要な承認 | 再開コマンド / 手順 | 現状 |
| --- | --- | --- | --- | --- |
| 1 | `rag_poc` の git 管理と `.gitignore` 追加 | 版管理方針（成果物をコミットするか、最新コピーのみとするか）の決定 | 方針決定後に `git init` → `.gitignore`（`__pycache__/` / `*.pyc` / `ragcore/target/` / `.recreate-audit.json` / `.ingest-checkpoint.json`）→ 成果物の選別 → 初回コミット | `todo.md` に未着手として残存。workspace ルートは non-git |
| 2 | クラウド埋め込みの実計測経路 | API キー（シークレット管理）と外部送信の明示承認 | 承認後に `rust_bridge` とは別の専用計測経路を追加し、API キーを安全なストアから読み込む。端点オプトインは既定拒否を維持（`RAG_ALLOW_REMOTE_ENDPOINTS=1` + `RAG_ALLOWED_ENDPOINT_HOSTS` + https） | `todo.md` に未着手として残存。ローカル比較のみ実施済み |
| 3 | CI ゲートの追加（#1 に依存） | #1 の git 化方針が前提 | git 化後に `.github/workflows/ci.yml` を追加し、`Seduce0068/workflows` の `rust-gates.yml@v1` を呼び出す。Python は `python3 rag_poc/test_rag.py`、JSON 整合チェック（schema_version / qrels_label）を併設 | `todo.md` に未着手として残存。#1 が未解決のため着手不可 |

## 他ドキュメントとの対応

件数や状態が文書ごとに異なるため、本インデックスを再開時の入口とする。
差分は参照先の更新待ちであり、ここでは事実のみを記録する。

- [pending-root-actions.md](../passkey-gateway/docs/pending-root-actions.md) — API キーローテーション残り1件（全体スコープ）。#2 のシークレット管理と間接的に関連。
- [STATUS.md](../passkey-gateway/docs/STATUS.md) — 状況ダッシュボード。rag_poc 固有の承認待ちは載らない。
- [FINAL-SUMMARY.md](../passkey-gateway/docs/FINAL-SUMMARY.md) — 環境全体の権限拡大保留の記録。
- `todo.md` — RAG セクションの未着手項目が正本。本インデックスはその抄録。

## 再開ルール

1. 承認が得られた項目だけを `todo.md` の未着手から着手へ移す。
2. 破壊的操作（git 履歴改変 / force push）は別途の事前確認が必要。
3. #3 は #1 の完了後にのみ着手する（依存関係を維持）。
4. 進捗は `todo.md` を更新し、本インデックスの「現状」列も合わせる。
