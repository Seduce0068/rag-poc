---
type: Playbook
title: "rag_poc コントリビューションガイド"
description: "Rust のみの rag_poc への変更を提案するときの検証ループ、コミット規約、公開リポジトリの author 匿名化手順。"
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

[README.md](README.md) は入口、`ragcore/src/` が現行仕様の正本である。
`docs/` 配下は旧 Python 実装のアーカイブ資料であり、現行 CLI の手順には使わない。
本リポジトリは公開リポジトリのため、秘密情報・内部パス・ホスト名を
Issue・PR・コミット・成果物へ記載しない。

## 検証ループ

変更後は次の順で実行する。コマンド名はプロジェクトで定義済みのものを使い、推測しない。
リポジトリ直下を cwd とする。

```bash
# 1. フォーマット
cargo fmt --manifest-path ragcore/Cargo.toml -- --check

# 2. テスト
cargo test --manifest-path ragcore/Cargo.toml

# 3. 静的解析（unsafe の新規使用を禁止）
cargo clippy --manifest-path ragcore/Cargo.toml -- -D warnings -D clippy::undocumented_unsafe_blocks

# 4. ビルドと CLI スモーク
cargo build --manifest-path ragcore/Cargo.toml
ragcore/target/debug/ragcore --help
echo '{"op":"cosine","a":[1.0],"b":[1.0]}' | ragcore/target/debug/ragcore
echo '{"op":"check_eval_results","paths":["eval-results.json"]}' | ragcore/target/debug/ragcore

# 5. ドキュメント lint
markdownlint-cli2 README.md CONTRIBUTING.md SECURITY.md docs/*.md

# 6. シークレットスキャン（コミット履歴全体）
betterleaks git
```

文書のみの変更では 2〜4 を省略してよいが、markdownlint とシークレットスキャンは必ず実行する。
失敗時は既存 baseline と新規失敗を区別し、原因とともに報告する。
自動修正と再レビューは最大 2 回を目安とする。

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
- `docs/` — 旧 Python 実装の設計・実測資料（履歴アーカイブ）
