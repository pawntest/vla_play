# so101-tool

[LeRobot SO-101](https://huggingface.co/docs/lerobot/so101) アームを、ブラウザ3Dプレビュー・
ルールベース制御(関節/IK移動・ジョグ)・AI制御(lerobotポリシー推論、Claude自然言語)で
操作するツールです。シミュレーションだけで全機能が動き、実機やSSHリモートはオプションを
足すだけで有効になります。データ収集(LeRobotDataset)→ 学習 → ポリシー実行まで一貫して
このツールで完結します。

## セットアップ(5分)

必要なものは **Python 3.12 以上**だけです(GPU 不要・実機なしで全機能が試せます)。

```bash
git clone <このリポジトリのURL> && cd vla_play

uv sync                          # uv の場合(推奨)。pip の場合: pip install -e ".[dev]"

so101-tool run --scenario examples/pick_cube.yaml    # 起動確認
# → 表示される「メインUI(カメラオーバーレイ付き)」のURL(既定 http://localhost:8081)
#   を開き、アームと赤いキューブが表示されれば成功。
#   画面左にカメラ映像が縦一列で常時表示され(📷ボタンで表示/非表示)、
#   操作は画面右のパネルだけで完結します。
```

追加機能は使うときに入れます(uv は `uv sync --extra real` のように指定):

| extras | 追加されるもの |
| --- | --- |
| `.[real]` | 実機 SO-101 接続(lerobot) |
| `.[nl]` | 自然言語制御(環境変数 `ANTHROPIC_API_KEY` が必要) |
| `.[policy]` | ポリシー推論・学習(torch を含む) |

## 起動ガイド

起動は常に `so101-tool run` の1コマンドで、構成はオプションで選びます。起動後、
コンソールに表示される**メインUI**のURL(既定 http://localhost:8081、カメラの
縦一列オーバーレイ付き)を開くと、操作・データ収集・学習・シーン編集の全機能が
UIから使えます(ヘッダーの状態バナーが現在のモードを常時表示。非常停止はラッチ式)。

### 1. シミュレーションのみ(実機なし)

```bash
so101-tool run                                      # アーム単体
so101-tool run --scenario examples/pick_cube.yaml   # 物体・カメラ付きの物理シーン
```

### 2. 実機を直接つなぐ(USBシリアル)

実機がこのPCにUSB接続されている場合は `--backend real` だけです。通信はシリアル直結で、
トンネルもトークンも登場しません。初回は必ず [docs/hardware.md](docs/hardware.md) の
安全チェック(ポート権限・キャリブレーション・方向確認)を先に済ませてください。

```bash
so101-tool run --backend real --port /dev/ttyACM0 --robot-id my_follower

# 実機とシムを連動させる場合は --link を追加(--scenario と併用可)
so101-tool run --backend real --port /dev/ttyACM0 --link both \
    --scenario examples/pick_cube.yaml
```

### 3. SSH先で動かす(Codespaces / Brev など)

アプリはリモート、実機は手元PC、という構成です。リモートにはシリアル接続がないため、
手元PCで動かす **teleop-client** が実機との橋渡しをします。経路はSSHトンネルのみ・
トークン認証付きで、ポートを公開する必要はありません
(設計の詳細: [docs/teleop_remote.md](docs/teleop_remote.md))。

```bash
# リモート側: --link を付けると受信側が自動起動し、ポートとトークンが表示される
remote$ so101-tool run --scenario examples/pick_cube.yaml --link both

# 手元PC側: トンネルを張り、実機をクライアントとして接続
laptop$ ssh -L 8765:localhost:8765 <remote>   # Codespaces はポート転送UIでも可
laptop$ so101-tool teleop-client --connect localhost:8765 \
            --token <表示されたトークン> --source follower --port /dev/ttyACM0
```

- 実機なしで疎通確認: `--source sine`(合成軌道)
- リーダーアームで片方向テレオペだけしたい場合: リモート側を `--link both` の代わりに
  `--teleop`、手元側を `--source leader` にする(UIの データ→Remote teleop から有効化)

### 直結とSSHの違い

|  | 2. 直結(USB) | 3. SSHリモート |
| --- | --- | --- |
| 実機の場所 | アプリと同じPC | 手元PC(アプリはリモート) |
| 起動方法 | `--backend real --port …` | アプリ側に `--link`(または `--teleop`)、手元側で `teleop-client` |
| 通信経路 | シリアル直結 | SSHトンネル(127.0.0.1限定+セッショントークン) |
| 使える機能 | UI・録画・ポリシー・`--link` 全モード、すべて同じ | 同左 |

### 実機↔シムの連動方向(`--link`)

方向はUIヘッダーの 🔗 ドロップダウンから実行中に切り替えられます:

| モード | 動作 |
| --- | --- |
| `to_sim` | 実機→シム。実機が基準でシムが追従(シーン内の物体とも物理干渉)。トルクを切って手で動かすとシムがミラーします |
| `to_real` | シム→実機。GUI・自然言語・ポリシーの指令はシムを駆動し、同じターゲットを実機へ影として送信。実機側が不調でもシムは止まりません |
| `both` | 双方向。指令は実機へ、シムは常に実機の実測関節に追従。手で動かしても指令で動かしても同期し続けます |

### よく使う追加オプション

| オプション | 意味 |
| --- | --- |
| `--record <repo_id> --record-root <dir>` | UIからLeRobotDatasetエピソードを録画 |
| `--policy-path <path/hub-id> --policy-task "…"` | 学習済みポリシーをロードして実行 |
| `--no-nl` | APIキーがあっても自然言語制御を無効化 |
| `--viser-port <n>` | ブラウザUIのポート(既定 8080) |

## その他のコマンド

```bash
so101-tool scripted-demos --scenario my.yaml --dataset my --episodes 50   # ピックデモ自動生成
so101-tool train --dataset data/my --policy act                           # 学習(lerobot-trainラッパー)
so101-tool ik-check --n 100                                               # IK精度チェック(実機不要)
so101-tool teleop-client …                                                # 上記3.の手元PC側
```

## 模倣学習の流れ

シーン定義 → デモ収集 → 学習 → 実行のすべてがUIまたはCLIで完結します。
詳細な手順とColab(無料GPU)の使い方は [docs/data_and_training.md](docs/data_and_training.md)。

```bash
cp examples/pick_cube.yaml my_task.yaml                                   # 1. シーン定義
so101-tool scripted-demos --scenario my_task.yaml --episodes 50 \
    --dataset my_task --root data/my_task                                 # 2. デモ収集
so101-tool train --dataset data/my_task --policy act                      # 3. 学習
so101-tool run --scenario my_task.yaml \
    --policy-path outputs/train/checkpoints/last/pretrained_model         # 4. 実行
```

## Python API

CLI・GUIの土台である `so101_tool.api` をそのままライブラリとして使えます:

```python
from so101_tool import api

with api.Session(scenario="examples/tabletop_cloth.yaml") as sess:
    sess.move_to([0.25, 0.0, 0.10])   # GUIと同じ安全フィルタを通るIK移動
    sess.gripper(0.0)
    sess.generate_demos("my/pick", episodes=30, root="data/pick")
api.train(dataset="data/pick", policy="act")
```

## 安全設計

- 実機・シムを問わず、全書き込みが安全フィルタ(関節リミット・速度制限・床面チェック)を通ります
- **非常停止**はラッチ式: 解除ボタンを押すまで一切のコマンドを拒否します
- 実機の初回接続時は [docs/hardware.md](docs/hardware.md) の方向確認手順に従ってください

## ドキュメント

- [docs/hardware.md](docs/hardware.md) — 実機SO-101のセットアップ・キャリブレーション・安全手順
- [docs/teleop_remote.md](docs/teleop_remote.md) — リモート接続のセキュリティ設計と補足
- [docs/scenario_reference.md](docs/scenario_reference.md) — シーンYAML完全リファレンス(環境・布・複数/取付カメラ)
- [docs/data_and_training.md](docs/data_and_training.md) — データ収集と学習のワークフロー
- [docs/architecture.md](docs/architecture.md) — モジュール構成・スレッドモデル・座標系
- [docs/why_this_tool.md](docs/why_this_tool.md) — Isaac Sim等との比較・YAMLの表現力

アームの3Dモデルは mujoco_menagerie の
[`robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
(Apache-2.0)を `src/so101_tool/assets/so101/` に同梱しています。

## トラブルシューティング

| 症状 | 対処 |
| --- | --- |
| ブラウザに何も表示されない | 起動ログのエラーを確認。ポート競合なら `--viser-port 8081` などに変更 |
| ヘッダーに「⚠ 実機側: error…」が出る | シリアルポート権限・キャリブレーション([docs/hardware.md](docs/hardware.md))、`lerobot[feetech]` のインストールを確認。**アプリはシム単体のまま動き続けます** — 原因を直したら「🔌 実機に再接続」ボタンで復旧(再起動不要) |
| 「Missing motor IDs … Full found motor list: {}」 | モーターが1台も応答していない状態。① アーム本体の**電源アダプタ**が入っているか(USBだけではモーターは動きません) ② `--port` がフォロワー機のポートか(リーダー機と取り違えやすい) ③ モーター間ケーブル を確認 → 「🔌 実機に再接続」 |
| 実機の接続で困ったら | **`so101-tool doctor --port /dev/ttyACM0`** — ポート検出→権限→lerobot→モーター応答→キャリブレーションを順に自己診断し、直し方を表示します。`run` 中はポートの抜き差しがコンソールにリアルタイム表示され、`--log-file app.log` で接続ログを保存できます |
| 実機↔シムが連動しない | `--link` を付けているか確認(`--backend real --scenario` だけでは連動しません)。SSH構成では手元PCの `teleop-client --source follower` の接続を確認 |
| ジョグを押しても「refused」と出て動かない | 目標位置が到達不能または床面下です。仕様: 正確に並進できない場合ロボットは一切動きません |
| 画面の動きが重い | 「📷 カメラ」タブのリアルタイムプレビューをオフに |

## 開発

```bash
.venv/bin/pytest          # 実機不要
.venv/bin/ruff check src tests
```
