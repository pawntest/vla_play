# so101-tool 日本語ガイド

[LeRobot SO-101](https://huggingface.co/docs/lerobot/so101) ロボットアームを、
**ブラウザ上のリアルタイム3Dプレビュー**、従来型の**ルールベース制御**
(ツール座標系ジョグ・指定点への移動・関節移動)、**AI制御**
(lerobot 学習済みポリシー推論と Claude API による自然言語コマンド)で操作するツールです。

**シミュレーション・ファースト**: 3D表示・IK・動作プリミティブ・自然言語制御のすべてが
実機なしの仮想空間で完結します。実機 SO-101 の接続は追加機能で、接続すると同じUIが
実機の関節角を3Dにミラー表示し(入力)、すべてのコマンドをサーボへ送信します(出力)。

## クイックスタート(シミュレーション、実機不要)

```bash
pip install -e ".[dev]"          # Python 3.10 以上
so101-tool run --backend sim
# ブラウザで http://localhost:8080 を開く
```

ブラウザにアームの3Dシーンと操作パネルが表示されます:

- **Mode** — `idle` / `mirror`(プレビューのみ)/ `rule`(動作プリミティブ)/ `policy`
- **Joints** — 関節スライダー +「Move to sliders」「Home」、速度スライダー
- **Cartesian** — ターゲットギズモをドラッグして「Go to target」。ツール座標系の
  ジョグボタン(±X/±Y/±Z、ステップ幅指定)
- **EMERGENCY STOP** — ラッチ式非常停止。「Reset e-stop」を押すまで一切の動作を拒否

IKの健全性チェック(実機不要): `so101-tool ik-check --n 100`

## 実機との連携

lerobot が必要なため Python 3.12 以上が必要です。シリアルポートの権限設定・
キャリブレーション・初回接続時の安全チェックリストは [hardware.md](hardware.md) を
参照してください。

```bash
pip install -e ".[real]"
so101-tool run --backend real --port /dev/ttyACM0 --robot-id my_follower
```

3Dプレビューが実機と同期します:

- `mirror` モード — 実機の関節角をリアルタイムに3D表示(トルクを切って手でアームを
  動かし、3Dモデルが同じ方向に動くかを確認するのに使います)
- `rule` モード — GUI・自然言語からの動作指令を、シミュレーションと同一の
  安全フィルタ(関節リミット・速度制限・床面チェック)を通して実機へ送信

## 自然言語制御(Claude)

```bash
pip install -e ".[nl]"
export ANTHROPIC_API_KEY=sk-ant-...
so101-tool run --backend sim        # real でも可
```

パネルに「Natural language」欄が現れます。例:
「グリッパーを5cm前に動かして」「x=250 y=0 z=150 に移動」「グリッパーを開いてホームに戻って」。
モデルに与えられるのはボタンと同じルールベースプリミティブだけで、すべての動作は
安全フィルタを通ります(25cm を超える相対移動は拒否されます)。

## 学習済みポリシー推論(ACT / SmolVLA)

```bash
pip install -e ".[policy]"          # Python 3.12 以上、torch を含む
so101-tool run --backend real --port /dev/ttyACM0 \
    --policy-path lerobot/smolvla_base --policy-task "pick up the cube"
```

パネルで Mode を `policy` に切り替えると推論が始まります。ポリシーはカメラ画像を
必要とするため、実機バックエンド専用です(シムバックエンドの既知の制限)。

## 模倣学習: データ収集 → 学習 → 実行

シーン定義からデモ収集・学習・実行まで、模倣学習の一連の流れをこのツールだけで
完結できます。詳しい手順は [data_and_training.md](data_and_training.md)(英語)を
参照してください。

```bash
# 1. シーンとタスクを YAML で定義(オブジェクト・カメラ・指示文)
cp examples/pick_cube.yaml my_task.yaml

# 2. デモを LeRobotDataset 形式で収集(3D GUI で手動、またはスクリプト自動生成)
so101-tool run --scenario my_task.yaml --record my_task --record-root data/my_task
so101-tool scripted-demos --scenario my_task.yaml --episodes 50 \
    --dataset my_task --root data/my_task

# 3. 学習(lerobot-train のラッパー。ACT / SmolVLA / Diffusion に対応)
so101-tool train --dataset data/my_task --policy act          # ローカル GPU がある場合
so101-tool train --dataset data/my_task --policy act \
    --push-dataset --dataset-hub-repo you/so101-my-task \
    --emit-colab train.ipynb    # GPU がない場合: 生成された notebook を Colab で実行(無料 GPU)

# 4. 学習済みポリシーを実行(シム・実機どちらでも)
so101-tool run --scenario my_task.yaml \
    --policy-path outputs/train/checkpoints/last/pretrained_model
```

データセットもチェックポイントも標準の LeRobot 形式なので、ここで学習したモデルは
lerobot が動く環境ならどこでも(逆も同様に)利用できます。

## 詳細

- モジュール構成とスレッドモデル: [architecture.md](architecture.md)
- 実機接続・キャリブレーション・安全手順: [hardware.md](hardware.md)
- アームの3Dモデルは mujoco_menagerie の
  [`robotstudio_so101`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
  (Apache-2.0)を `src/so101_tool/assets/so101/` に同梱しています。
