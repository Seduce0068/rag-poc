---
type: Policy
title: "rag_poc セキュリティポリシー"
description: "rag_poc の脆弱性報告先、対応方針、公開リポジトリとしての扱いを定める。"
tags:
  - security
  - reporting
  - rag
generated:
  by: "team:rag-poc"
  at: "2026-11-26T23:00:00+09:00"
status: "stable"
stale_after: "2026-12-26"
---

## 公開リポジトリについて

本リポジトリは公開リポジトリである。秘密情報（API キー・トークン・パスワード）、内部パス、ホスト名、個人情報を Issue・PR・コミット・成果物へ記載しない。クラウド機能は既定で無効（`RAG_CLOUD_ENABLED=0`）だが、有効化すると質問文・チャンク本文・ベクトルが外部 API へ送信される点に注意する（詳細は[README のクラウド章](README.md#クラウドモデルopencodego)）。

## 報告先

対象は `https://github.com/Seduce0068/rag-poc` である。脆弱性または機微情報の漏えいに気付いた場合は、公開の Issue を作成せず、同リポジトリの **Security** タブにある「Report a vulnerability」から非公開で報告する。報告には再現手順、影響範囲、想定される修正方針を添える。秘密情報そのものを報告文へ貼り付けないこと。

「Report a vulnerability」が表示されない場合は、まず同タブで private vulnerability reporting が有効かを確認する。未認証のブラウザでは報告画面へアクセスできないため、リポジトリ管理者が Console で設定を確認する。公開の Issue や Discord 等へ機微情報を書かないこと。

## 対応方針

- 報告を受けたら内容を確認し、受領した旨を非公開で返す。
- 影響範囲を切り分け、公開リポジトリでは履歴への残存を避けるため、必要ならキーのローテーションと履歴からの消去を検討する。
- 修正は専用ブランチで行い、[CONTRIBUTING.md](CONTRIBUTING.md) の検証ループを通してから反映する。
- 公開可能な情報のみをまとめ、対応結果を報告者へ非公開で共有する。必要に応じてセキュリティアドバイザリを発行する。

## 参考

- [CONTRIBUTING.md](CONTRIBUTING.md) — 検証ループとコミット規約
- [README.md](README.md) — セットアップ・使い方
