# リモートテレオペ(SSH / Codespaces / Brev)

> 英語版(正): [../teleop_remote.md](../teleop_remote.md) — 差異があれば英語版が優先です。

so101-tool をリモートコンテナ(GitHub Codespaces、NVIDIA Brev、任意のSSH先)で動かし、
**手元のノートPCに接続したリーダーアーム**で操作できます。リモートのシミュレーション操作、
データセット収集、リモート側に接続されたフォロワー実機の駆動に使えます。

## 使い方

リモート側:

```bash
so101-tool run --scenario examples/pick_cube.yaml --teleop
# 起動時に 127.0.0.1:8765 とセッショントークンが表示されます
```

手元のPC(リーダーアーム接続済み、`pip install "so101-tool[real]"`):

```bash
ssh -L 8765:localhost:8765 <remote>   # Codespaces のポート転送UIでも可
so101-tool teleop-client --connect localhost:8765 \
    --token <表示されたトークン> --port /dev/ttyACM0 --robot-id my_leader
```

ブラウザUIの **Remote teleop** パネルが「🟢 streaming」になったら
**Enable TELEOP mode** を押してください。Record dataset パネルでの録画も
通常どおり動くため、実機テレオペのデモをリモートのデータセットに収集できます。
ハードウェアなしでの疎通確認は `--source sine`(合成軌道)を使ってください。

## セキュリティ設計

- 受信側は **127.0.0.1 のみにバインド**(コード上、それ以外のバインドを拒否)。
  ネットワークに一切露出せず、到達経路はSSHトンネルだけ — 暗号化と認証はSSHのものです
- ハンドシェイクで**セッションごとのランダムトークン**が必須(タイミング安全比較)。
  共有コンテナ上の他ユーザーもモーションを注入できません
- 固定JSONスキーマ・4KiB行上限・NaN/形状検証・関節リミットクランプ。さらに全ターゲットは
  制御ループの**安全フィルタ**(速度制限・リミット)を通ります。ストリームが0.5秒途絶えると
  アームは**その場でホールド**し、非常停止が常に優先されます
- Codespaces の転送ポートはデフォルトで *private*(自分のGitHubアカウントのみ)です。
  public にする必要はありません
