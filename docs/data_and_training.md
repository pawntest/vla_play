# データ収集と学習: 模倣学習のワークフロー

このページでは一連の流れを最初から最後まで説明します。**シーンを定義する →
デモンストレーションを記録する → ポリシーを学習する(ローカルまたは無料の
クラウドGPU) → アームで実行する**。実機のステップより前は、ハードウェアが
一切なくても動作します。

学習のフロントエンドは `src/so101_tool/training.py` にあり、lerobot 標準の
`lerobot-train` エントリーポイントを起動します(lerobot >= 0.5.1、
Python >= 3.12)。そのため、生成されるチェックポイントはどこでも使える通常の
LeRobot ポリシーです。

## 0. 環境まとめ

インストールは [README のインストール](../README.md#インストール) を参照してください。
対応関係: シムでの収集 = `[dev]`、実機での収集 = `[real]`
([hardware.md](hardware.md) も参照)、ローカル学習 = `[policy]`、
クラウド学習 = ブラウザ(Google Colab、無料GPU)+ HuggingFace アカウントのみ。

## 1. シナリオを定義する

シナリオとは、アームの周りにある物体・カメラ・タスクを記述する小さなYAML
ファイルです。まずは [`examples/pick_cube.yaml`](../examples/pick_cube.yaml)
から始めてください。完全なスキーマ(box/sphere/cylinder/mesh オブジェクト、
リセットごとのランダム化、カメラ配置)は
[`src/so101_tool/scenario.py`](../src/so101_tool/scenario.py)
のモジュールdocstringにあります。

```yaml
name: pick_cube
task: "pick up the red cube"
objects:
  - name: cube
    type: box
    size: [0.015, 0.015, 0.015]
    pos: [0.25, 0.0, 0.02]
    rgba: [0.9, 0.1, 0.1, 1]
    pos_noise: [0.05, 0.05, 0.0]   # randomized every reset -> diverse demos
cameras:
  - {name: front, pos: [0.6, 0.0, 0.35], lookat: [0.2, 0.0, 0.1], fovy: 58}
  - {name: top,   pos: [0.25, 0.0, 0.7], lookat: [0.25, 0.0, 0.0], fovy: 58}
```

`task` の文字列とカメラ名は、記録されたデータセットの言語指示・観測キーに
そのままなります。記録を始める前に決めておきましょう。

## 2. デモンストレーションを収集する

デモンストレーションは標準の **LeRobotDataset** 形式(`meta/`, `data/`,
`videos/` を含むディレクトリ)で保存されます。そのため、他のLeRobot
データセットとまったく同じように学習・共有できます。

**シミュレーション上で、手動で(GUI記録):** 記録を有効にしてシナリオを実行し、
Cartesian ギズモ / jog ボタンでタスクをデモンストレーションします。各エピソードは
シナリオのカメラを映像ストリームとしてレンダリングした状態で保存されます。

```bash
so101-tool run --scenario examples/pick_cube.yaml --record pick_cube --record-root data/pick_cube
```

**シミュレーション上で、自動で(スクリプトによるデモ):** ピック&プレース系の
タスクであれば、シミュレータが持つ物体の正解姿勢を使ってプランニングすることで
デモを自動生成できます。数分でデータセットの土台を作るのに便利です。

```bash
so101-tool scripted-demos --scenario examples/pick_cube.yaml \
    --episodes 50 --dataset pick_cube --root data/pick_cube
```

(上記の2コマンドはいずれも `so101_tool/data` で現在仕上げが進められている
記録パイプラインの一部です。フラグは今後わずかに変わる可能性があります。)

**実機ロボット上で:** アームとカメラを接続し([hardware.md](hardware.md)参照)、
実機バックエンドで記録するか、lerobot 自身の `lerobot-record` テレオペ記録機能を
使ってください。生成されるデータセットは同じ形式で、まったく同様に学習できます。

物体の位置にばらつきを持たせながら **30〜50エピソード以上**を目指してください
(シナリオの `pos_noise` / `yaw_range` はそのためにあります)。エピソード数が
約20を下回ると、使えるACTポリシーが得られることはほとんどありません。

## 3. ローカルで学習する

```bash
pip install -e ".[policy]"     # Python >= 3.12
so101-tool train --dataset data/pick_cube --policy act
```

これは妥当なデフォルト値(2万ステップ、バッチサイズ8、wandbオフ、デバイスは
自動検出: cuda → mps → cpu)で `lerobot-train` を呼び出します。`training.py` の
`TrainArgs` に対応する有用なフラグは次の通りです。

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | (required) | local dataset dir **or** HF hub repo_id |
| `--policy` | `act` | `act` \| `smolvla` \| `diffusion` |
| `--steps` / `--batch-size` | 20000 / 8 | training length / batch |
| `--output-dir` | `outputs/train` | checkpoints land in `<dir>/checkpoints/` |
| `--device` | `auto` | `cuda`, `mps`, `cpu`, or auto-detect |
| `--resume` | off | continue from `<output-dir>/checkpoints/last` |
| `--push-to-hub --hub-repo user/repo` | off | push the trained policy to the HF hub |
| `--dataset-root DIR` | — | explicit local dataset directory (`--dataset` is then the repo_id) |
| `--extra` | — | raw passthrough to `lerobot-train` (e.g. `--extra "--save_freq=1000"`) |

フラグでカバーされていないものは、すべて `--extra` を通してそのまま渡せます。
利用可能なオプションの全体像は `lerobot-train --help` で確認できます。

補足: ACTを2万ステップ学習させるのはCPUでは一晩仕事ですが、そこそこのGPUなら
約1〜2時間です。ローカルにGPUがない場合は下記のクラウド手順を使ってください。

## 4. クラウドで学習する(無料GPU)

**Google Colab(推奨、無料のT4 GPU):** 1コマンドでローカルのデータセットを
Hugging Face hub にプッシュし(クラウドマシンからはローカルディスクが見えないため、
まず `HF_TOKEN` に *write* 権限のトークンを設定してください)、すぐに実行できる
notebookを生成します。

```bash
export HF_TOKEN=hf_...
so101-tool train --dataset data/pick_cube --policy act \
    --push-dataset --dataset-hub-repo your-hf-user/so101-pick-cube \
    --emit-colab train_pick_cube.ipynb
```

すでにプッシュ済みの場合は、アップロードを飛ばしてnotebookだけ生成できます。

```bash
so101-tool train --dataset your-hf-user/so101-pick-cube --policy act \
    --emit-colab train_pick_cube.ipynb
```

このnotebookはlerobotをインストールし、HFにログインし、ローカルで実行するのと
*まったく同じ* `lerobot-train` コマンドを実行し(デバイスは `cuda` に固定)、
学習済みチェックポイントをhubへプッシュし返します。実行前に *Runtime →
Change runtime type → T4 GPU* を選択するのを忘れないでください。

**その他のGPUマシン**(レンタルサーバー、研究室のマシン、HF Jobsなど): 本ツールは
マシン固有の処理を何も追加しないため、どこでも同じ2行で動きます。

```bash
pip install "so101-tool[policy]"     # Python >= 3.12
so101-tool train --dataset your-hf-user/so101-pick-cube --policy act \
    --push-to-hub --hub-repo your-hf-user/so101-act-pick-cube
```

## 5. 学習済みポリシーを実行する

ランナーにチェックポイントのディレクトリ(ローカル学習は
`<output-dir>/checkpoints/last/pretrained_model` に書き込みます)、または
プッシュ先のhubリポジトリを指定します。

```bash
# in sim, against the same scenario
so101-tool run --scenario examples/pick_cube.yaml \
    --policy-path outputs/train/checkpoints/last/pretrained_model

# on the real robot
so101-tool run --backend real --port /dev/ttyACM0 \
    --policy-path your-hf-user/so101-act-pick-cube --policy-task "pick up the red cube"
```

パネルの Mode を `policy` に切り替えてください。ポリシーが出力するすべての
アクションは、手動制御と同じ安全フィルタ(関節リミット・速度クランプ・床面
キープアウト・ラッチ式e-stop)を通過します。

## エンドツーエンド チートシート

```bash
# 1. scene
cp examples/pick_cube.yaml my_task.yaml           # edit objects/cameras/task

# 2. demos (sim)
so101-tool run --scenario my_task.yaml --record my_task --record-root data/my_task   # by hand, or:
so101-tool scripted-demos --scenario my_task.yaml --episodes 50 \
    --dataset my_task --root data/my_task

# 3a. train locally (GPU box)
so101-tool train --dataset data/my_task --policy act

# 3b. ...or in the cloud
export HF_TOKEN=hf_...
so101-tool train --dataset data/my_task \
    --push-dataset --dataset-hub-repo you/so101-my-task --emit-colab train.ipynb
#   -> upload train.ipynb to colab.research.google.com, Runtime -> T4 GPU, Run all

# 4. deploy
so101-tool run --scenario my_task.yaml \
    --policy-path outputs/train/checkpoints/last/pretrained_model     # sim check
so101-tool run --backend real --port /dev/ttyACM0 \
    --policy-path you/so101-act-my-task --policy-task "pick up the red cube"
```
