---
type: Decision
title: "ADR-0004: ZIPアーカイブ閲覧をcoreの読み取り専用モジュールとして実装する"
description: "アーカイブ閲覧の境界、zipクレートの選定理由、パス逸脱と過大展開への対策を記録する。"
tags:
  - ADR
  - zip
  - core
  - security
generated:
  by: opencode
  at: "2026-10-17T15:20:00+09:00"
status: accepted
---

## 背景

`docs/requirements.md` の REQ-002 は ZIP「アーカイブ内のコンテンツ閲覧」を求める。
追加検討する品質要件は「アーカイブ処理でパスの逸脱と過大な展開を扱う」を挙げる。
ADR-0003 により、共通コアは wasm32 でもビルドできる必要がある。

実装前の調査で次を確認した。

| 調査 | 結果 |
| --- | --- |
| `zip` 4.6.x | 4.6.0 / 4.6.1 とも yanked。新規の直接依存には使えない |
| `Cargo.lock` の `zip` | 4.6.1 が推移依存として存在（既存分として残る） |
| `deflate-flate2` 単独 | flate2 のバックエンドが未指定のままビルドされ E0432 ×3 で失敗 |
| `deflate-flate2-zlib-rs` | pure Rust の zlib-rs を引き込み、native と wasm32 の両方で成功 |
| 9.x の `name()` | `Result<Cow<str>, ZipError>` を返す（4.x は `&str`） |

## 決定

`openrustbooks-core` に読み取り専用の `archive` モジュールを置き、
`zip` を pure Rust のフィーチャーだけで追加する。

```toml
[dependencies.zip]
version = "9.0"
default-features = false
features = ["deflate-flate2-zlib-rs"]
```

- 入力は `&[u8]` のみ。`std::fs` や `std::env` に依存しない（wasm 対応）。
- `ArchiveReader` はバイト列・エントリ一覧・索引を所有し、
  読み出しのたびに `ZipArchive` を作り直す。`&self` のまま読めるため
  `Send + Sync` を満たす。
- アーカイブの作成・書き込み・展開は提供しない。

### 安全性の扱い

| 攻撃 | 対策 |
| --- | --- |
| パス逸脱 | `\` を `/` に正規化し、先頭 `/`・`..`・ドライブ文字・NUL・空名を拒否 |
| 逸脱エントリを含むアーカイブ | そのエントリだけ飛ばさず、`new` が失敗する（fail-closed） |
| 過大な展開 | `ArchiveLimits` でエントリ数・単体サイズ・合計サイズを検査 |
| 宣言サイズの偽装 | 実読み出しを `max_entry_size + 1` バイトで打ち切り、超過を検出 |
| 非対応・暗号化エントリ | `Stored` と `Deflated` 以外、および暗号化は `CoreError::Archive` |

## 選択肢と根拠

| 観点 | `zip` クレート | 自作パーサ | 展開をUIに任せる |
| --- | --- | --- | --- |
| 実装コスト | ✅ 低い | ❌ deflate まで自作 | ✅ 低い |
| wasm対応 | ✅ pure Rust を選択可能 | ✅ | ⚠️ 環境ごとに別実装 |
| 安全性の一元化 | ✅ コアで検査できる | ✅ | ❌ 実装ごとに散る |
| 依存の増加 | ⚠️ 1 クレート | ✅ なし | ⚠️ 間接的に増える |

**結論:** `zip` を採用し、境界と安全性の検査をコアに集約する。
逸脱エントリは「飛ばす」案もあったが、閲覧後に欠落へ気付きにくいため
アーカイブ全体を拒否する。

## 影響

- コアの依存に pure Rust のクレートが 1 つ増える
  （`flate2/zlib-rs` 経由。C コンパイラを必要とするバックエンドは選ばない）。
- 推移依存の `zip` 4.6.1（yanked）は `Cargo.lock` に残る。
  直接依存は 9.0.4 であり、既存分は上流の更新で解消する。
- EPUB は `mimetype` による判定だけを提供する。
  `container.xml` と OPF（spine・目次）の解析は別作業。
- アーカイブ内の編集・再保存（REQ-007）は範囲外のまま。

## 検証

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo check -p openrustbooks-core --target wasm32-unknown-unknown
```

`core/src/archive.rs` の 12 テストが、順序・両圧縮方式・BOM 除去・
エラー分類・パス逸脱・リミット・破損入力・EPUB 判定・空アーカイブ・
`Send + Sync` を検証する。公開 API を外部クレートとして使う受け入れ確認も
別途行い、成功を確認した。

## 未決定

- 目次と本文抽出の範囲（EPUB のどこまでを初版に含めるか）。
- 大きなアーカイブの入力サイズ上限を誰が決めるか（呼び出し側の責務のまま）。
- アーカイブ内の文書を編集する場合の保存方式（REQ-007 の未決定事項）。
