---
type: Decision
title: "ADR-0001: デスクトップ版にOS WebViewを採用する"
description: "Tauriの標準的なOS WebView方式を採用する決定と、その影響。"
tags:
  - ADR
  - Tauri
  - WebView
generated:
  by: opencode
  at: "2026-11-17T13:19:26+09:00"
status: accepted
---

## 背景

Rustを活用して、ブラウザからも利用できるリーダー兼エディタを構想している。
デスクトップ基盤としてTauriを検討する中で、
OSのWebViewを利用する方式と、Chromiumを同梱する方式を比較した。

## 決定

デスクトップ版はTauriのOS WebView方式（wry）を採用する。

利用者の明示的な選択「WEBviewでいいわ」に基づく決定。
Chromium同梱のCEFやElectronは採用しない。

Tauriのメジャーバージョン、フロントエンドのフレームワーク、
サーバーの有無は本決定に含めない。

## 選択肢と根拠

[TauriのWebView資料][webview]では、
WindowsはWebView2、macOSはWKWebView、
LinuxはWebKitGTKを利用する構成が説明されている。

[Tauriのランタイム比較][runtime]と[CEFガイド][cef]では、
OS WebViewの利用とChromium同梱の違いを確認できる。
参照時点のCEFガイドはTauri 3.0アルファの資料。
将来のリリース状況を固定的な事実として扱わない。

Chromium同梱は描画エンジンのバージョンを揃えられる一方、
配布サイズが増える。今回、利用者はOS WebView方式を選択した。

## 影響

- 描画エンジンの別途同梱を基本構成に含めず、配布サイズを抑える。
- EPUB・Markdown・編集部品の表示や入力を、対象WebViewで確認する。
- 画面はWeb技術で構成できるが、ブラウザ版の配信と入出力は別途設計する。
- LeptosやYewを画面に採用する選択肢は残る。
- フォルダ操作などのOS連携は、Tauri側の処理として設計する。
- この選択だけで、メモリ使用量や起動速度の具体的な値は保証しない。

## 再検討の条件

対象環境のWebViewで必須の描画・入力機能を実現できず、
妥当な代替手段も取れない場合に再検討する。

再検討時は、この決定を黙って変更せず、
新しい決定記録から置き換え関係を明示する。

[webview]: https://v2.tauri.app/reference/webview-versions/
[runtime]: https://v3.tauri.app/develop/webview-runtime/
[cef]: https://v3.tauri.app/develop/cef/
