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

本リポジトリは公開リポジトリである。秘密情報（API キー・トークン・パスワード）、内部パス、ホスト名、個人情報を Issue・PR・コミット・成果物へ記載しない。現行 Rust 実装は既定でローカルの llama-server / Qdrant のみに接続し、外部送信も課金も発生しない。`RAG_CLOUD_ENABLED=1` を明示した場合のみ opencodeGO の `/embeddings` へ質問文・チャンク本文・ベクトルが送信され、利用料金が発生し得る。接続先は https が必須で、API キーはログ・エラー・成果物へ出力しない。
さらに `RAG_CLOUD_CHAT_ENABLED=1`（`RAG_CLOUD_ENABLED=1` が必須）では、全 query で
質問文とチャンク本文が `/chat/completions` へ送信され、`x-opencode-session` ヘッダが
付与される。これらのクラウド経路は既定では無効。
投入コーパスは既定でリポジトリ配下のみで、`RAG_ALLOW_EXTERNAL_CORPUS=1` を
明示したときだけ外部ディレクトリを読む。`RAG_REPO_ROOT` はこの信頼境界の基準を
変更するため、信頼できる運用者が明示したときだけ設定する。
公開成果物のゲートは走査上限到達時に fail-close する。

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
