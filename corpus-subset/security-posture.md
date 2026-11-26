# セキュリティ態勢（2026-11-08 実測）

## 入口（すべてtailnet限定・TLSはTailscaleが終端）

| 入口 | ゲート | 認証方式 | 未認証 |
|---|---|---|---|
| `https://<host>/` | oauth2-proxy-web (:14182) → Caddy :18890 | パスキー（OIDC/PKCE S256） | 302 |
| `https://<host>:8445/` | oauth2-proxy-agent (:14181) → Caddy :18889 | パスキー（OIDC/PKCE S256） | 302 |
| `https://<host>:8443/` | 入口Caddy :14443 → Pocket ID | パスキー（登録・ログイン） | `/setup` 200（初回のみ） |

- issuer/JWKS は公開HTTPSを厳格検証（`--insecure-oidc-skip-issuer-verification` 不使用）。
- メール未確認の許容なし。許可は**特定メール1件**（`--authenticated-emails-file`）。
- PKCE S256、`--prompt=login`、CSRF cookie per-request、`insecure-oidc-skip-nonce=false`。

## 分離

- ゲートCookie名は個別（staging / agent / web）。共有なし。
- DSHへ渡すHostを分けているため、DSHの署名Cookieの対象(host / host:8445)も分離。
- 内部待受（14180/14181/14182/18888/18889/18890/14443/14445）は**127.0.0.1のみ**。
- コンテナは `read_only`・`cap_drop: ALL`・`no-new-privileges`・非root・ホストネットワーク。

## ホスト

- UFW: default deny incoming。LAN許可は 8080 のみ（DSHの3080/3081は撤去済み）。
- LANからDSH直結不可（`192.168.128.111:3080/3081` は待受なし）。
- 秘密ファイル（env/backup/inbox）は 0600・0700。
- 失敗ユニット: `system-all-update-resume.service`（11/7の更新時に一部タスク失敗、DSH／ゲートとは無関係）。

## 残余リスク

1. **同一ホスト上のプロセス**は loopback(:3080/:3081) へ直接アクセスできる（エージェント自身を含む）。ホスト内の信頼境界は別問題として残る。
2. Pocket ID とアプリが同一ホスト名（ポート違い）。Cookieはポート非依存で相互に送られ得る。現在はゲートCookie名を分離し実害はないが、専用ホスト名に移すのが理想（要：DNS/証明書）。
3. 外付けバックアップはNTFSのため権限保護なし。物理保護が前提（秘密・会話履歴を含む）。
4. DSHのSameSite=Laxパッチはnpm更新で巻き戻る可能性（`ExecStartPre`で再適用中）。
5. sudo は `NOPASSWD: ALL` のまま。パスキーで入口を守っても、エージェント経由の権限は不変。

## 未実施（判断待ち）

- Timeshift：保存先にLinuxファイルシステムが必要。NTFSの外付けディスクは不適で、再フォーマットは既存VMバックアップを破壊するため**承認なしに実施しない**。
- カーネル適用（7.0.0-43）：再起動はこのGUIを一時停止させるため、タイミングの決定が必要。

## 独立監査（2026-11-08 16:30）への対応

| 指摘 | 対応 |
|---|---|
| C1: opencode API(49374) がtailnetから未認証で到達可・root権限 | **遮断**。`ts-input`より前のINPUT先頭にDROP（IPv4/IPv6）。`block-tailnet-opencode.service`で再起動後も維持。ローカル(lo)は維持 |
| C2: 8443ゲートから `/trial/agent/` でagentアプリへ横断 | **削除**。横断ルートを撤去し、未認証は302で停止、認証後も404 |
| H3: Host検査なし（オープンリダイレクト） | 3入口すべてで**ホスト名検証**（不一致は400）。リダイレクト先は固定URL |
| M1: 認証済みユーザー情報がDSHへ渡らない | `copy_headers X-Auth-Request-Email/User` を3ゲートに追加 |
| M3: `/setup` がtailnetに露出 | 入口で **404** に固定（初回登録は完了済み） |
| M4: XFF偽装でPocket IDの記録IPを汚染可能 | 入口で `X-Forwarded-For` を実接続元に**上書き** |
| L3: 監視が `/setup` を健康判定 | discoveryエンドポイント判定へ変更 |
| H2: UFWはtailnetを止めない | 方針を「UFWに頼らず明示バインド+必要ならINPUT先頭で遮断」に修正（文書化） |
| H1: ポートでCookie分離は不可 | ゲートCookie名の分離で対応中。専用ホスト名が理想である旨を維持 |
| M2: バックアップ文書の矛盾 | 文書を実態（会話履歴を含む）に修正 |

### 未対応（ユーザー判断）
- `sudo NOPASSWD: ALL` の縮小（C1の根本原因の一つ）。
  → **準備完了**: `scripts/root-helper.sh` を `/usr/local/sbin/dsh-root-helper` に配置し、
  ゲート側の sudo 依存を全て helper 経由へ移行（バックアップで実証済み）。
  適用は `docs/sudoers-proposal.md` の段階案どおり、承認後に行う。
- opencode 自体を `--hostname 127.0.0.1` で起動し直す（現在はユーザーの端末セッション由来。遮断は済み）。
- 5746/8123/18555 など他の 0.0.0.0 待受の要否整理。

## 追加対応（2026-11-08 17:10）

- **入口Caddy(:14443)にもHost検証を追加**: 異なるHostは400、公開ホストは200。discovery/ゲートは正常。
- **DSH web の起動トークンをローテーション**: `dsh-web` を再起動（agent GUIは無影響）。
  新しいURLを `inbox/dsh-launch-url-web.txt`（0600）に更新。既存ブラウザCookieは有効なまま。
- **opencode の扱い**: 公式仕様では `opencode serve` の既定は 127.0.0.1 だが、
  TUIが起動するサーバーはランダムなホスト/ポートを選ぶと明記されている。
  - 公式の防御手段: `OPENCODE_SERVER_PASSWORD`（HTTP Basic認証）。
  - 現状は **tailnetからの到達をFWで遮断**済み。恒久対応は
    (a) 起動時の `--hostname 127.0.0.1` 指定、(b) `OPENCODE_SERVER_PASSWORD` 設定 のいずれか。
  - **対応済み（2026-11-08 17:28）**: `opencode service set hostname 127.0.0.1` → `service restart`。
    待受は **127.0.0.1:49374** のみ、tailnet IPからは接続不可を確認。TUIは同一ホスト接続のため継続利用可。
    FW遮断ルールは多層防御として残置。

### 検証エージェントの残指摘
- XFFは `127.0.0.1` に固定されるためPocket IDのログは実クライアントIPにならない（仕様上の割り切り）。
- 14443のHost検証は今回追加済み。
- 49374は依然0.0.0.0バインド（FW遮断のみ）。

## 秘密の衛生（2026-11-08 17:18）

- env/秘密鍵は0600、DBは0600、ディレクトリは0700。
- 初期試験時の一時フィクスチャ（生成秘密を含む）を削除済み。
- バックアップアーカイブは0600（NTFS上は物理保護が前提）。

## セッション寿命の見直し（2026-11-08 18:16 / 2026-11-12 改定）

Google同等の「同一端末ではほぼ切れない」体験に寄せるため、次を適用した。

- oauth2-proxy（3ゲート）: `--cookie-expire=720h`（30日）、`--cookie-refresh=0`
  （2026-11-12 変更。旧 `1h`）、`--scope=openid email profile offline_access`、
  `--prompt=login` を削除。
- Pocket ID: `SESSION_DURATION=43200`（**12時間**。30日を意図する場合は `2592000`）。
- 実測（2026-11-12 変更前）: 認可URLは scope に offline_access を含み、prompt なし。
  Cookie設定は `expiry:720h refresh:after 1h0m0s`。
- 効果: ゲートCookieが有効な間は**パスキー再入力なし**。

### 2026-11-12 改定: プロアクティブ・リフレッシュ停止

- 変更: 3ゲートの `--cookie-refresh=1h` → `0`（`trial/compose.trial.yaml`）。
- 理由: 1時間リフレッシュが DSH クライアントの並列リクエストで同時発火し、Pocket ID の
  refresh token ローテーションと競合してゲートセッションが破棄されていた。
  失効直後の `/api` 呼び出しがログインへ302 → ブラウザのCORSで `Failed to fetch`
  （DSH表示: `gateway/internal: client api: session/create failed`）。
- 影響: OIDCトークンの定期更新が止まるため、Pocket IDでの失効反映はCookie期限まで遅れる。
  気になる場合は `--cookie-expire` を `168h`（7日）等に短縮する。
- 詳細: `docs/dsh-session-create-failed-2026-11-11.md`。
- セキュリティ上の前提: 端末の画面ロックを有効にし、端末紛失時はブラウザのCookie削除または
  Pocket IDユーザー無効化で失効させる。入口はtailnet限定のまま。

## dockerグループ分離（2026-11-08 18:38 適用済み）

- `koppe` を `docker` グループから削除。ゲート一式は **rootのシステムunit**で起動するため、
  通常運用でユーザー側dockerは不要。
- これにより「シェル取得 → docker run でroot」の経路を遮断（sudo縮小と合わせて実効的な権限低下）。
- 手動の docker 操作は sudo パスワードが必要（`docker` は sudo 許可リスト外）。
- 残: 自動更新の起動時unitは起動直後のaptロック競合で失敗することがある（ゲートとは無関係）。
