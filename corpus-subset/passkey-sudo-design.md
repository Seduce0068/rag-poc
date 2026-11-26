# sudoにパスキーを使う設計（Pocket ID device flow）

## 結論
**実装可能**。Pocket IDは`urn:ietf:params:oauth:grant-type:device_code`に対応しており、
公開OIDCクライアント＋PAM(`pam_exec`)で「sudo時にスマホのパスキー承認を要求」できる。
自宅の顔認証と違い、**出先（tailnet接続があれば）でも承認可能**。

## 方式
1. sudo実行 → `pam_exec.so` が `/usr/local/sbin/pocketid-sudo` を呼ぶ
2. スクリプトがPocket IDへdevice authorizationを要求
3. 端末に **URLとコード** を表示
4. ユーザーはスマホ（Tailscale接続）でURLを開き、Pocket IDにパスキーでログイン/承認
5. スクリプトがtoken endpointをポーリング → id_tokenを**JWKSで署名検証**し、
   `iss`/`aud`/`exp`/`email`（許可ユーザー）を確認 → PAM成功

## 必要なもの（初回だけ）
### A. Pocket IDに公開クライアントを作成（管理画面・要パスキー）
1. Pocket ID（`:8443`）に管理者としてログイン
2. **Administration → OIDC Clients → 追加**
   - Name: `sudo-pam`
   - **Public client**（シークレットなし）を選択
   - Grant: **Device code** を有効化（UIに項目がある場合。無ければ標準で可）
   - Callback URLs: 空で可
3. 作成後に表示される **Client ID** を控える（秘密ではない）

### B. 設定ファイル（root・600）
`/etc/pocketid-sudo.conf`:
```ini
[pocketid]
issuer = https://<host>.<tailnet>:8443
client_id = <上で控えたClient ID>
allowed_email = <user>@example.test
timeout = 120
```

### C. インストール（root）
- `/usr/local/sbin/pocketid-sudo`（0700 root）を配置
- `/etc/pam.d/sudo` の `pam_howdy` の**前**に追加:
  ```
  auth sufficient pam_exec.so /usr/local/sbin/pocketid-sudo
  ```
- 既存の顔認証・パスワードはフォールバックとして残す

## セキュリティ
- スクリプト・設定はroot所有（ユーザー書換不可）
- 公開クライアントのため秘密情報を保存しない
- id_tokenはJWKSで署名検証、issuer/audience/exp/emailを検証
- 許可ユーザー（PAM_USER=<user>）以外は拒否
- 承認コードは短寿命・1回限り。タイムアウト120秒
- 端末に表示されたコードをユーザーが確認して承認する（コードの横流しに注意）
- **ローカルkappe/エージェントも、パスキー承認なしではsudoできなくなる**（顔認証・パスワードは残す場合を除く）

## フォールバック
- 出先でPocket IDに到達できない場合に備え、`pam_howdy`（自宅顔認証）とパスワードを残す。
- どうしても使えない場合の非常用に、NOPASSWDの固定メンテナンスhelper（未導入）を用意可能。

## 現状
- `scripts/pocketid-sudo`（プロトタイプ）と`scripts/pocketid-sudo.conf.example`を作成済み。
- 未実施: Pocket IDクライアント作成、PAM設定変更（クライアントID入力後）。

## 追記（2026-11-10 02:10）: 実装時の重要知見

- **`pam_exec.so`は既定で実UID（<user>）でコマンドを実行**する。root権限が必要なスクリプトは
  PAM行に **`seteuid` オプション**を付ける（例: `pam_exec.so seteuid /usr/local/sbin/pocketid-sudo`）。
  これが無いとrootチェックでexit 1になる（今回の原因）。
- PAM順序は **顔認証 → パスキー → パスワード** が実用的:
  - 自宅: 顔（howdy timeout 20秒）で即完了
  - 出先/顔なし: 20秒後にデバイスコードURLを表示 → スマホのパスキー承認
- 承認URLは `/run/pocketid-sudo/latest`（0644）にも書き出し、エージェントがチャットで伝達可能。
- 適用スクリプト: `scripts/apply-passkey-sudo-fix.sh`（乾燥実行既定）。

## 出先での承認URLの届け方（2026-11-10 19:15）

PCのブラウザ自動オープンは**自宅用の補助**。出先では以下の経路を使う。

### 1. ユーザーがsudoする場合（SSH/スマホターミナル）
sudoを実行した端末にURLとコードが表示されるため、その端末（スマホ等）でそのまま開ける。
PC画面を見る必要はない。

### 2. エージェントがsudoする場合（推奨）
`scripts/agent-sudo.sh /tmp/sudo-url -- <command>` を使う。
- sudoをバックグラウンド実行し、承認URLを`/tmp/sudo-url`へ書き出す
- エージェントがURLを**このチャットに転記** → スマホのDSH画面から開いてパスキー承認
- 承認後、コマンドが実行され終了コードが同ファイルに記録される

これにより、**ユーザーがPCの前にいなくても**、チャット経由でパスキー承認してroot操作を許可できる。
（`/run/pocketid-sudo/latest`は0644で誰でも読めるが、承認にはパスキーが必要）
