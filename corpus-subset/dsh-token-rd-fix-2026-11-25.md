---
type: incident-report
title: DSH 起動トークンがパスキー認証後に失われる問題の修正
description: 8445 の未認証リダイレクトが rd から ?token= を落とし、DSH が 401 を返していた障害の調査と修正
tags: [dsh, passkey, oauth2-proxy, caddy, token]
status: stable
generated: 2026-11-25T20:45:00+09:00
verified:
  - at: 2026-11-25T20:45:00+09:00
    level: verified
---

## 症状

- `https://<host>:8445/` を開くと `dsh web authentication required; reopen the URL printed by dsh web.` が出る。
- プライベートタブではパスキーは使えるが、その後に同エラーになる。
- キャッシュ削除でパスキー自体は成功するが、DSH 側で 401 になる。

## 原因

2 層認証の 2 段目で停止していた。

1. 第 1 層: oauth2-proxy のパスキーゲート — 成功
2. 第 2 層: DSH 本体の起動トークン → 署名 Cookie 交換 — 失敗

entry Caddy (`:14445`) → 内部 Caddy (`:18889`) の `handle_response @unauthorized` が
`rd` を固定文字列 `https://<host>:8445/` で作っており、`?token=` クエリが
ログイン後の戻り先から落ちていた。そのためパスキー後にトークンなし URL へ戻り、
DSH の `authorizeIndex()` が 401 を返した。

既知の同種事例は `docs/mobile-access.md` に记载済み。

## 修正

`passkey-gateway/trial/caddy/Caddyfile.staging` の `:18889` ブロックで `rd` に
`{uri}` を付加し、元のクエリ（起動トークン）を保持するようにした。

```caddyfile
redir * /oauth2/sign_in?rd=https://<host>:8445{uri} 302
```

`:18888` / `:18890` / `:8446` の `rd` は変更していない。

## 適用時の注意（重要）

本番ゲートは `dsh-passkey-trial` compose で、
`compose.trial.yaml` + `compose.trial.override.yaml` の **両方** が必要。
`compose.trial.yaml` のみで `up -d --force-recreate caddy` すると
`network_mode: host` の override が外れ、`:18889` がバインドできず 502 になる。
必ず次の形式で再作成する。

```bash
docker compose -p dsh-passkey-trial \
  --project-directory "$PWD/passkey-gateway/trial" \
  -f passkey-gateway/trial/compose.trial.yaml \
  -f passkey-gateway/trial/compose.trial.override.yaml \
  up -d --force-recreate --no-deps caddy
```

`admin off` のため `caddy reload` は使えない（`:2019` 接続拒否）。

## 検証

| 項目 | 結果 |
| --- | --- |
| 修正後の `rd` | `...:8445/?token=***MASKED***`（トークン保持） |
| `:14445` / `:18889` | 302（正しい未認証リダイレクト） |
| `:14181/ping` | 200 |
| `health-report.sh` | fail=0（warn 6 は既存の無関係項目） |
| DSH 直接 `/?token=` | 303 + `Set-Cookie`（30 日） |
| DSH 直接 `/`（トークンなし） | 401 |
| Betterleaks `git` | no leaks found |
| `dsh-config-export` | commit `82fc2f2` / push 完了 |

## 401 画面への案内追加（同日追記）

初回だけトークン付き URL を開く運用は維持したまま、DSH の 401 画面に案内を出すようにした。
コアモジュールは変更せず、`:18889` の `reverse_proxy` ブロックで DSH の 401 を捕捉して
案内テキストへ差し替える。

```caddyfile
@dsh_auth_required status 401
handle_response @dsh_auth_required {
    respond "{$DSH_AUTH_GUIDE}" 401
}
```

案内文言は Caddyfile に直書きせず、ローカルの
`passkey-gateway/trial/caddy/caddy.env`（git 未追跡）の `DSH_AUTH_GUIDE` に置く。
公開リポジトリには `caddy.env.example`（変数名のみ）を置き、実文言は含めない
（`dsh-config-export` commit `31e0ca3`）。Caddy は `{$VAR}` 展開を `respond` 内でも
解決するため、コンテナの `env_file` 経由で文言を渡せる（validate 済み）。

案内にはトークン取得コマンドも含める。スマホからは Termux で PC に SSH して
実行できるため、ターミナル不可を理由に案内を省く必要はない。

```bash
journalctl --user -u dsh-web-agentrouter --no-pager -o cat \
  | grep 'dsh web:' | tail -1 | grep -oE 'https://[^ ]+'
```

自動リダイレクトでトークンを付与する案も検討したが、トークンは
`processLaunchToken()` が起動ごとに `randomBytes(32)` で生成し固定化オプションがないため、
ゲート側にトークンを保管する必要が生じる。それはトークン漏洩面を増やすため採用しない。

`dsh-config-export` へは commit `ba48690` で同期済み（公開用にホスト名はサニタイズ維持）。

## 運用

トークンは DSH の設計上 URL クエリが唯一の入口だが、使い捨てである。
`authorizeIndex()` が 303 で `./` へリダイレクトし、`Referrer-Policy: no-referrer`
と `Cache-Control: no-store` を付けるため、交換後は URL に残らない。
一度 Cookie が取れれば 30 日間はトークン不要で、`:8445` をブックマークして
パスキーのみで入れる。

残る既知リスク: `dsh web` のジャーナルにトークンが平文で出る（DSH 本体の仕様）。
チャット・Issue・コミットには貼らないこと。

## 補足

`dsh-config-export` は公開リポジトリのため、ホスト名は
`<host>.<tailnet>` のサニタイズ形式を維持して同期した。
