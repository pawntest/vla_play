# 実機SO-101のセットアップと安全性

## 前提条件

- Python >= 3.12 と `pip install -e ".[real]"`(`lerobot[feetech]` がインストール
  されます)。
- [lerobot SO-101 ガイド](https://huggingface.co/docs/lerobot/so101)に従って
  アームが組み立てられ、電源が入っていること。

## シリアルポート

```bash
ls /dev/ttyACM*            # usually /dev/ttyACM0
sudo usermod -aG dialout $USER   # then log out/in — avoids sudo for the port
```

複数のデバイスが接続されている場合は、アームを抜き差しして `ls /dev/ttyACM*`
の差分を確認してください。

## キャリブレーション

本ツールはlerobotを介して関節角度を度数で読み書きするため、事前にlerobotで
アームをキャリブレーションしておく必要があります(ロボットごとに1回、
`~/.cache/huggingface/lerobot` 以下に保存されます)。

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
```

`so101-tool run --backend real` を起動する際は同じ `--robot-id` を使用して
ください。

## 初回接続チェックリスト(この順番で実施してください)

1. **MIRRORモードでの符号チェック。** アームに通電した状態で起動し、Modeを
   `mirror` に切り替えます。このモードではループは*読み取りのみ*を行います。
   各関節をゆっくり押してみてください(トルクで動かせない場合はlerobot側で
   トルクを無効化できます。あるいは、リーダーアームで動かす/低トルクで手で
   動かす様子を観察するだけでも構いません)。3Dプレビューは実機の関節と
   **同じ方向**に動く必要があります。もしある関節が逆方向に動く、あるいは
   オフセットしている場合は、`src/so101_tool/config.py` の `JointMap.signs` /
   `offsets_deg` をデータ側で修正してください(関節ごとに、内部値 =
   `(real_deg − offset) / sign`)。修正後、再度確認します。MIRRORが一致する
   まで先に進まないでください。
2. **小さな関節移動。** `rule` に切り替え、速度を約0.2に設定し、関節スライダーを
   1つ約0.2 rad動かして *Move to sliders* を押します。電源スイッチのそばに
   手を置いておいてください。
3. **Cartesian移動。** *Snap target to TCP* を押し、ギズモを数センチドラッグして
   *Go to target (position)* を押します。
4. **自然言語(任意)。** `ANTHROPIC_API_KEY` を設定した状態で:
   「open the gripper」のように指示します。

## 安全動作

- すべてのコマンド(GUI・自然言語・ポリシー)は安全フィルタを通過します:
  関節リミットのクランプ、関節ごとの速度クランプ(デフォルト2.0 rad/s ×
  1.5倍のマージン)、床面キープアウト。
- **EMERGENCY STOP** はラッチ式で、1制御ティック(20 ms)以内に書き込みを
  停止します。サーボは最後の位置を保持します(STS3215のポジションモード)。
  真のハード停止が必要な場合は電源を切ってください。
- `disable_torque_on_disconnect=True`: 本ツールを終了するとアームのトルクが
  抜けます。アームを支えるか、先にホームポジションに戻しておかないと落下する
  可能性があります。
- 接続後最初の観測値はサニティチェックされます。ラジアンのように見える値
  (度数ではなく)が検出された場合は、乱暴な動作を指令する代わりに接続を
  中断します。

## 既知の制限

- `mirror` モードは現状トルクを自動で切り替えません。スクリプトから制御する
  ための `LeRobotBackend.set_torque(False)` が用意されています。
- POLICYモードにはカメラが必要なため、実機バックエンド専用です。スクリプトで
  設定する場合は `AppConfig.cameras`(lerobotのカメラ設定辞書)経由で
  設定してください。例:
  `{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}`。
