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
  at: "2026-11-26T21:25:00+09:00"
verified:
  - by: "lead/r6-aggregation"
    at: "2026-11-26T21:25:00+09:00"
status: "stable"
stale_after: "2026-12-26"
---

## 目的

`todo.md` の RAG バックログのうち、**ユーザー承認が必要だった**項目を
1 箇所に集約する。3件はすべて完了しており、今後は公開済みリポジトリの
継続運用を記録するインデックスである。承認を求める依頼ではない。

> 2026-11-26 21:25 更新: メタデータ時刻（`generated.at` / `verified.at`）を
> 本文更新時刻へ同期した。ユーザー承認により下記3件はすべて完了。
> 詳細は `todo.md` の完了項目と git 履歴（4 コミット）を参照。
> リポジトリは public 化済み。公開前スキャン（Betterleaks 全履歴 hit zero /
> 画像0件 / 追跡対象・スキャン対象は実識別子なし）は完了している。
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

## 次アクション（クラウド cost の実測記録）

2026-11-27 の調査で、Go 上限内の要求では API の `usage.cost` が `0` になる
仕様（[公式 Cost 定義](https://opencode.ai/docs/console/api/inference/#cost)）が
確定した。実消費は `usage.opencode.allowance`（月額上限に対するドル）と
`opencode.billing_source` / `opencode.list_cost` で別途報告されるが、現行の
`cloud_client.py` / `cloud_bench.py` はこれらを破棄している。対応は次のとおり。

1. `_accumulate_usage` / `_normalize_usage` に `opencode.allowance` /
   `billing_source` / `list_cost` を通し、成果物へ記録する（要コード変更）。
2. `cost_source` に Go 上限内を表す値（例 `go-allowance-zero`）を追加し、
   notes の「理由は未確認」を公式仕様ベースの記述へ置き換える。
3. 生レスポンスの usage をマスク済みで 1 件保存し、`opencode` フィールドの
   実在と allowance 値を Runtime Proof として確認する
   （`OPENCODE_GO_API_KEY` は一時環境変数のみへ読み込む）。
4. 実請求との突き合わせは Console の Usage export API
   （`/console/api/v2/usage/export`）を使う。権限あるキーが必要なため
   ユーザー確認後に実施する。

## 完了記録

1. 3件はいずれも完了（`todo.md` の完了項目と git 履歴を参照）。
2. 破壊的操作（git 履歴改変 / force push）は今後も事前確認が必要。
3. 次アクションは公開済みリポジトリの継続運用。変更時に Betterleaks 全履歴
   スキャンと CI ゲート（Rust gates + JSON 必須キー）の通過を確認する。
   ローカル検証用の eval-results-cloud-answer.json は追跡対象外のため、
   公開スキャンの対象外とする（生成 answer にローカル識別子が残り得る）。
   具体コマンドは「次アクション（公開済みリポジトリの継続運用）」を参照。
4. 進捗は `todo.md` と本インデックスの「現状」列を合わせて更新する。
