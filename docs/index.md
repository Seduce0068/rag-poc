---
type: Index
title: "rag_poc 詳細ドキュメント索引"
description: "rag_poc の背景・設計判断（explanation.md）と設定・実測値・成果物（reference.md）の description と用途を列挙した索引。"
tags:
  - rag
  - index
  - documentation
generated:
  by: "dsh-subagent/fi-restore"
  at: "2026-11-26T21:55:00+09:00"
verified:
  - by: "human:lead"
    at: "2026-11-26T21:56:00+09:00"
status: "stable"
---

## この索引の目的

`rag_poc/` の入口は [README.md](../README.md) である。本文書は詳細ドキュメントへの
入口をまとめ、各文書の `description` と用途を列挙する。

## 文書一覧

### [docs/explanation.md](explanation.md)

- `description`: ローカル RAG 検証 PoC の構成・ハイブリッド検索・モデル比較の設計理由と既知の制限を、README の各章を根拠に説明する。
- 用途: なぜこの構成かを知る背景ドキュメント。Ruri での再現手順、qwen2.5:7b のモデル比較実験（5.38 秒の比較注意含む）、別ドメインベンチマーク（AGPL-3.0-or-later の出典明記、rerank 7.17 秒含む）を収録する。

### [docs/reference.md](reference.md)

- `description`: rag_poc の検索モード、全環境変数（既定値・単位）、評価成果物一覧と代表的な実測値を集約した参照ドキュメント
- 用途: 設定値、qrels 3 種類の使い分け、評価指標の計算定義（gain / discount / IDCG / 照合規則）、成果物一覧、代表的な実測値を引くためのリファレンス。

## 入口へのリンク

- [rag_poc/README.md](../README.md) — セットアップ・使い方・検索モードの入口。README の「詳細ドキュメント」節から本文書へのリンク追加は担当外のため未実施。
- README のリンク先 [Ruri での再現手順](explanation.md#ruri-での再現手順) — explanation.md の該当見出しへ接続している。
