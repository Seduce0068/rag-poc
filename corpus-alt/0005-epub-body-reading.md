---
type: Decision
title: "ADR-0005: EPUB本文の読み出しをcoreの読み取り専用モジュールとして実装する"
description: "container.xmlとOPFの解析範囲、quick-xml 0.42の選定理由、本文抽出とパス逸脱への対策を記録する。"
tags:
  - ADR
  - epub
  - core
  - parser
generated:
  by: opencode
  at: "2026-10-17T16:50:00+09:00"
status: accepted
---

## 背景

`docs/requirements.md` の REQ-002 は EPUB「電子書籍の閲覧」を求める。同書の
検証条件はリーダーが「目次・ページ・章を移動できる」ことを挙げるが、本作業では
本文と章の読み出しまでを実装し、目次 UI との接続は次段に回す。ADR-0003 により
共通コアは wasm32 でもビルドできる必要があり、ADR-0004 で読み取り専用の
`ArchiveReader`（ZIP 閲覧、`is_epub()` 付き）が入っている。

実装前の調査で次を確認した。

| 調査 | 結果 |
| --- | --- |
| `quick-xml` 0.42 | `Cargo.lock` に推移依存として既存。追加ダウンロードなしで使える |
| `roxmltree` | 未取得。DOM API は平易だが新規バージョンを lock に足す必要がある |
| `xml-rs` | キャッシュのみ。採用実績がなく API の実測もしていない |
| 0.42 の要素名 | `LocalName` が `&str` を返す。`b"rootfile"` との比較はコンパイルできない |
| 0.42 のテキスト | `Event::Text` は復号済み。`&amp;` などは `Event::GeneralRef` に分離される |
| 0.42 の `trim_text(true)` | `a &amp; b` の前後の空白まで落ちて `a&b` になる（実測） |

調査は使い捨てのスクラッチクレートで行い、native のテストと
`wasm32-unknown-unknown` のビルドで確認した。

## 決定

`openrustbooks-core` に読み取り専用の `epub` モジュールを置き、
`ArchiveReader` の上に `META-INF/container.xml` → OPF → 章本文を載せる。

```toml
[dependencies.quick-xml]
version = "0.42"
default-features = false
```

公開 API は `EpubBook::{open, with_limits, metadata, chapters, chapter_bytes,
chapter_text}`、`EpubMetadata`、`EpubChapter` とする。`EpubBook` は入力バイト列と
解析結果を所有し、内部可変性を使わないため `Send + Sync` を満たす。

### 安全性の扱い

- `mimetype` が `application/epub+zip` でないアーカイブは
  `CoreError::UnsupportedFormat` として開かない。
- コンテナ文書・OPF・章の解決は fail-closed とし、先頭 `/` とスキーム付き URL を
  拒否する。ルートを越える `../` の解決は `CoreError::Validation`、不正な OPF は
  `CoreError::Archive` として報告し、黙って読み飛ばさない。
- manifest の `href` にスキーム付き URL がある EPUB は開かない。EPUB の manifest は
  コンテナ内の資源を相対参照する前提であり、リモート資源を宣言する不正な OPF を
  読み進めないほうが安全と判断した。
- リミットは `ArchiveLimits` をそのまま使い、エントリ数と展開サイズは
  `ArchiveReader` が検証する。

### 本文抽出の規則

- `head`・`script`・`style` の内側は本文に含めない。
- ブロック要素（`p` `div` `h1`〜`h6` `li` `blockquote` `section` `article`
  `pre` `tr` `td` `th`）の終了と `br` で改行する。
- 実体参照は定義済みの 5 つと数値参照だけを文字に戻し、未知の名前付き参照は
  無視する。0.42 は実体参照を `Event::GeneralRef` として渡すため自前で解く。
- 行ごとに空白の連続を 1 つにまとめ、空行を落とす。`trim_text(false)` で読み、
  正規化は自前で行う。

## 選択肢と根拠

| 選択肢 | 判断 |
| --- | --- |
| `quick-xml` 0.42（採用） | lock 済みで追加ダウンロードなし。wasm32 でもビルドでき API を実測済み |
| `roxmltree` | DOM は扱いやすいが新規バージョンを lock に足す必要があり、0.42 で足りる |
| 手書きパーサ・正規表現 | 名前空間と実体参照の誤りが入りやすく、保守コストに見合わない |
| spine を辿らず HTML を直接抽出 | 読み上げ順と章の境界を失い、要件を満たせない |

## 影響

- `core` の公開 API が増える（`EpubBook`・`EpubMetadata`・`EpubChapter`）。
  `Document` 型との統合は未実施で、UI 側の接続は別作業になる。
- 目次（EPUB3 の nav / EPUB2 の NCX）と章タイトルは未対応。
- 固定レイアウト・DRM 付き EPUB・非整形式の HTML は対象外。非整形式の XHTML は
  `CoreError::Archive` になる。
- 依存は pure Rust のまま増えず、native と wasm32 の両方でビルドできる。

## 検証

- ループ実走は 1 ラウンドで `VERDICT: PASS`（worker と verifier は独立セッション）。
- ゲートは緑。`cargo fmt --all -- --check`、`cargo check --all-targets`、
  `cargo test --all-targets`（core 38 件、うち epub 19 件）、
  `cargo clippy --all-targets -- -D warnings`、
  `cargo check -p openrustbooks-core --target wasm32-unknown-unknown`。
- 外部クレート視点の受け入れテストを別途作成し、次を確認したうえで削除した。
  日本語本文と数値参照（`&#x3042;`）、`br` と `pre` の改行、全角空白の正規化、
  `full-path` の先頭 `/` と `full-path` を持たない `rootfile`、本文が空の章、
  リモート・絶対・逸脱 `href` の拒否、非 EPUB、リミット超過、破損入力で panic
  しないこと。5 件すべて pass。
- プロダクションコードに `unwrap`・`expect`・`panic!`・`unsafe` が無いことを
  目視と grep で確認した。

## 解決済み

- 目次（nav / NCX）の解析と章タイトル: ADR-0006 / 0007 で実装。
- 非整形式 XHTML の扱い: ADR-0059 で OPF / nav は fail-closed、
  章本文はサニタイズに委譲すると決定。
- リモート資源を宣言する EPUB: ADR-0060 で受け付け、
  ADR-0066 で警告を表示すると決定。
- 画像の扱い: ADR-0013 で Blob URL 化、ADR-0043 〜 0045 で SVG をサニタイズ。
  EPUB 内 CSS の適用は未実装（下記「未決定」）。

## 未決定

- EPUB 内 CSS の安全な適用（ADR-0012 から継続）。
