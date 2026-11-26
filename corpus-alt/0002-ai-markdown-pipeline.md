---
type: Decision
title: "ADR-0002: AI支援Markdown生成・修正パイプラインを導入する"
description: "AI API + markdownlint (rumdl) の組み合わせで Markdown 品質を保証するパイプラインを共通コアに実装する決定。"
tags:
  - ADR
  - AI
  - Markdown
  - rumdl
  - pipeline
generated:
  by: opencode
  at: "2026-11-17T13:19:26+09:00"
status: accepted
---

## 背景

Markdown は構造化テキスト形式として AI 生成・修正と lint 検証の相性が最良である。利用者から「AI に Markdown を生成・修正させ、markdownlint で整形する機能があってもいい」との提案があり、これをコア機能として採用する。

## 決定

以下の構成で **AI Markdown パイプライン** を共通コア (`openrustbooks-core`) に実装する。

| 構成要素 | 採用技術 | 根拠 |
|----------|----------|------|
| Markdown lint/整形 | **`rumdl`** (Rust製 markdownlint 互換) | Wasm対応・高速・Node不要・ルール番号互換 (MDxxx) |
| AI クライアント | `async-openai` / `rig-core` (抽象化層経由) | 非同期・ストリーミング・マルチプロバイダ対応 |
| プロンプト管理 | 自前レジストリ (ファイルベース・バージョニング) | タスク別テンプレート・プロンプトエンジニアリング資産化 |
| 差分計算・表示 | `similar` / `diff` | インライン差分でユーザー確認容易 |
| トークン計算 | `tiktoken-rs` | コスト見積もり・コンテキスト窓管理 |

パイプラインの基本フロー：

1. ユーザー入力 (自然言語 / 既存MD + 指示)
2. AI API で Markdown 生成・修正 (ストリーミング対応)
3. `rumdl` で検証・自動修正 (設定ファイル `.markdownlint.yaml` 準拠)
4. 合格まで最大 N 回再生成 (デフォルト 3 回)
5. 差分表示 → ユーザー確定 → 保存・プレビュー反映

## 選択肢と根拠

### lint エンジン: `rumdl` vs Node版 `markdownlint` vs 自前実装

| 観点 | `rumdl` | Node版 | 自前 |
|------|---------|--------|------|
| Wasm 化 | ✅ 容易 | ❌ 困難 (Nodeランタイム必要) | ⚠️ 巨大な実装コスト |
| 速度 | ✅ ネイティブ高速 | ⚠️ プロセス起動オーバーヘッド | ✅ 高速だが再発明 |
| ルール互換 | ✅ MDxxx 番号同一 | ✅ 本家 | ❌ 互換維持困難 |
| メンテナンス | ✅ 活発 | ✅ 本家 | ❌ 自前負担大 |

**結論:** `rumdl` 一択。Wasm 対応によりデスクトップ/ブラウザ双方で同一品質の lint を提供可能。

> **実装状況（2026-11-17 時点）:** ブラウザ版の `WasmMarkdownLinter` は
> まだ `rumdl` に配線されておらず、空の issue 配列を返すプレースホルダである。
> ADR-0003 の方針どおり AI クライアントとパイプラインはネイティブ専用。
> rumdl のブラウザ版への配線は将来タスク（本 ADR の選定自体は変更しない）。

### AI プロバイダ抽象化: 独自トレイト vs `rig` 等フレームワーク

初期は **独自最小トレイト** (`AiClient`) で抽象化し、将来 `rig` 等へ移行可能にする。プロンプトレジストリも自前管理し、ベンダーロックイン回避。

## 影響

- **共通コアに新規モジュール `ai_markdown` 追加** (型定義は両対応、クライアントとパイプラインはネイティブ専用。ADR-0003 参照)
- **依存関係追加**: `rumdl`, `async-openai`, `tiktoken-rs`, `similar`, `serde` 等
- **設定ファイル**: `.markdownlint.yaml` (プロジェクトルート) / ユーザー上書き可能
- **環境変数/設定**: `OPENAI_API_KEY` 等、ローカルLLM (Ollama等) エンドポイント対応
- **UI 統合**: CodeMirror 6 拡張として「AI 生成」「AI 改善」アクション追加
- **コスト/プライバシー**: ローカルLLM優先オプション・トークン上限設定・送信内容最小化を必須実装

## 再検討の条件

- `rumdl` の Wasm ビルドが実用的サイズ/速度でない場合
- AI API コスト/レイテンシが実用閾値を超え、ローカルLLMでも満たせない場合
- markdownlint ルールセットと AI 指示の競合が頻発し、ユーザー体験を損なう場合

再検討時は本 ADR を置き換える新 ADR を作成し、移行経路を明示する。

## 関連

- [requirements.md](../requirements.md) : REQ-010 追加予定
- [architecture.md](../architecture.md) : 「AI Markdown パイプライン」節追加予定
- [ADR-0001](../decisions/0001-use-system-webview.md) : 共通コア Wasm 化方針と整合
