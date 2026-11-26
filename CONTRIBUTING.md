---
type: Playbook
title: "rag_poc コントリビューションガイド"
description: "rag_poc への変更を提案するときの検証ループ、コミット規約、公開リポジトリの author 匿名化手順。"
tags:
  - contributing
  - verification
  - rag
generated:
  by: "team:rag-poc"
  at: "2026-11-26T23:00:00+09:00"
status: "stable"
stale_after: "2026-12-26"
---

## はじめに

[README.md](README.md) は入口、[docs/index.md](docs/index.md) は詳細ドキュメントの索引である。変更の前に双方を確認すること。本リポジトリは公開リポジトリのため、秘密情報・内部パス・ホスト名を Issue・PR・コミット・成果物へ記載しない。クラウド機能の注意は[README のクラウド章](README.md#クラウドモデルopencodego)に集約されている。

## 検証ループ

変更後は次の順で実行する。コマンド名はリポジトリで定義済みのものを使い、推測しない。

```bash
# 1. フォーマット（Rust クレート）
cargo fmt --manifest-path rag_poc/ragcore/Cargo.toml -- --check

# 2. Python 構文チェック
python3 -m py_compile rag_poc/*.py

# 3. テスト（外部サービスへ接続しない）
python3 rag_poc/test_rag.py
RAG_REQUIRE_RUST=1 python3 rag_poc/test_rag.py
cargo test --manifest-path rag_poc/ragcore/Cargo.toml

# 4. 静的解析
cargo clippy --manifest-path rag_poc/ragcore/Cargo.toml -- -D warnings

# 5. ドキュメント lint
markdownlint-cli2 rag_poc/README.md rag_poc/docs/*.md rag_poc/CONTRIBUTING.md rag_poc/SECURITY.md

# 6. シークレットスキャン（コミット履歴全体）
betterleaks git
```

設定やドキュメントのみの変更では Python 構文チェックとテストを省略してよいが、Markdown lint とシークレットスキャンは必ず実行する。失敗時は既存 baseline と新規失敗を区別し、原因とともに報告する。自動修正と再レビューは最大 2 回を目安とし、曖昧な指摘や設計判断が必要な事項、破壊的操作は無理に反復しない。

## コミット規約

- 1 コミット 1 論理変更とし、無関係な差分を混ぜない。
- 変更内容と意図が分かる簡潔なメッセージを付ける。
- 秘密情報を含む差分はコミットしない。`git add` の前に差分を確認する。
- push 先は追跡ブランチとし、upstream 未設定なら設定する。

## 公開リポジトリの author 匿名化

グローバル設定の実名・実メールが author に残らないよう、コミット時に上書きする。

```bash
git -c user.name=dsh-agent \
  -c user.email=dsh-agent@users.noreply.github.com \
  commit -m "<変更内容>"
```

恒久的なグローバル設定の変更はユーザー承認後に実施する。

## 参考

- [SECURITY.md](SECURITY.md) — 脆弱性の報告先と対応方針
- [README.md](README.md) — セットアップ・使い方
