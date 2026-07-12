# アーキテクチャ

> 英語版(正): [../architecture.md](../architecture.md) — 差異があれば英語版が優先です。

## 単位と規約(パッケージ全体で共通)

- アーム関節: `q` = numpy の `(5,)`、**ラジアン**、順序は
  `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]`
  (= lerobotのモーター名 = MJCFの関節名)。
- グリッパー: `0..1` の分数(0が閉、1が開)。
- 姿勢: 位置(m) + ロボット基部フレームでの `wxyz` クォータニオン。TCPは
  ジョー(顎)の間にあるMJCFサイト `gripperframe` です。
- 単位変換(度 / 0〜100)を行うのは `LeRobotBackend` のみで、`JointMap`
  (関節ごとの符号/オフセットと速度リミットも管理)を介して行われます。

## スレッドモデル

```
 GUI callbacks / NL agent / CLI            ControlLoop thread (50 Hz)
 ─────────────────────────────            ───────────────────────────
  Command ──▶ commands: Queue ──────────▶  read_state()
  (MoveJ/MoveL/JogTool/SetGripper/         drain queue (SetMode, Stop, e-stop first)
   Home/Stop/SetMode)                      step active primitive → q_target
       ▲                                   SafetyFilter.filter()
       │ cmd.wait() / cmd.ok               robot.write_targets()
       │                                   publish LoopSnapshot (immutable)
  render loop (main thread, 30 Hz) ◀────── snapshot()
    RobotView.sync(q, gripper)  → viser scene
    ControlPanel.update(snap)   → readouts
```

- ロボットのバックエンドに触れるのは制御ループだけです。
- コマンドの発行元は `cmd.wait()` でブロックします。ループは
  (完了/横取り/タイムアウト/e-stop/シャットダウンのいずれかで)必ず1回だけ
  `finish()` を呼ぶため、何かがハングすることはありません。
- `Kinematics` はスレッドセーフではありません。制御ループとレンダリング経路は
  それぞれ独自のインスタンスを持ちます。
- E-stopはラッチする `threading.Event` で、いかなる書き込みよりも先に
  尊重されます。

## モード

| Mode | Writes? | Behavior |
| --- | --- | --- |
| `idle` | no | snapshots only |
| `mirror` | no | 3D preview follows the (real) robot; used for hand-posing/sign checks |
| `rule` | yes | motion primitives; MoveL/JogTool run a differential-IK servo per tick |
| `policy` | yes | `PolicyRunner.step()` provides targets each tick |

自然言語エージェントはモードではありません。言語を同じプリミティブへ翻訳する
プロデューサーであり(まず `rule` に切り替えます)、その動作は同じ安全
フィルタに従います。

## 5自由度アームでのIK

5自由度のアームは任意の6自由度姿勢を実現できません。そのため
`Kinematics.ik*` は mink を使い、`FrameTask(position_cost=1.0,
orientation_cost=0.3)` + `PostureTask(1e-2)` を用いています。位置は厳密に
守られ、向きは緩やかにトレードオフされます。`wxyz=None` の場合は向きの
指定を完全に無視します。到達可能な100個のランダムターゲットで計測したところ、
98%が5 mm未満で収束し(中央値1.2 mm)、1回のsolveあたり約1 msでした。

## シムバックエンド

`SimBackend` は速度制限つきの運動学的積分器です — ポジション制御された
STS3215が行っているのと同じ一次追従であり、物理シミュレーションではありません。
決定論的(注入可能なクロック)で、どんなdtでも安定しており、プレビュー用途と
しては実機サーボの挙動に近いものです。将来的には接触物理を同じ
`RobotInterface` の裏側に差し込むこともできます。

## 3Dビュー

`RobotView` はMuJoCoモデルから直接viserのシーンノードを構築します(ボディ
ごとに1フレーム、メッシュは `mjModel` の頂点/面配列から生成 — URDFの読み込みも
ファイルの再パースもありません)。ティックごとに `mj_kinematics` の後、
`data.xpos/xquat` をフレームハンドルへ書き込みます。MuJoCoとviserは同じ
`wxyz` クォータニオン規約を共有しているため、変換はそのまま素通しできます。
