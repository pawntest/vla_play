# シナリオ YAML リファレンス

シーン(環境・物体・カメラ・タスク)を定義するYAMLの完全リファレンスです。
座標系はアーム基部が原点、+X前方、+Z上、単位はメートル/ラジアンです。GUIの
Scenario→Edit objects/Cameras/Environment パネルは同じスキーマを対話的に編集し、
「Save scenario YAML」でこの形式に書き出します。

以下の内容はすべてGUI上でも対話的に編集できます(ギズモのドラッグ、クリックでの
追加、カメラ/環境エディタ)。編集結果はYAMLへ書き戻せます。ローダー/ビルダー:
`so101_tool.scenario.load_scenario / build_model`(`api.py` が再エクスポートしています)。

## トップレベル

| key | type | default | meaning |
| --- | --- | --- | --- |
| `name` | str | file stem | scene name (shown in the GUI) |
| `task` | str | `""` | natural-language task; stored in every dataset frame and passed to VLA policies |
| `settle_steps` | int | 100 | physics steps after each reset before control starts |
| `environment` | map | see below | static surroundings |
| `objects` | list | `[]` | manipulable objects |
| `cameras` | list | front+top | render cameras (dataset + policy observations) |

## `environment`

```yaml
environment:
  floor: {rgba: [0.35, 0.4, 0.45, 1], checker: true, rgba2: null}
  table: {size: [0.7, 0.5], height: 0.0, thickness: 0.04, pos: [0.15, 0.0],
          rgba: [0.55, 0.42, 0.3, 1]}
  props:
    - {name: wall, type: box, size: [0.01, 0.4, 0.2], pos: [-0.15, 0, 0.2],
       rgba: [0.85, 0.85, 0.88, 1]}
  light_diffuse: 0.8
```

- `floor.checker: true` にすると2色のチェッカー柄になります(2色目は `rgba2`、
  デフォルトは `rgba` を暗くした色)。`table` がある場合、床はテーブル天板より
  0.4 m 下に配置されます。
- `table`: 静的な板で、上面が `height` の高さになります(アーム基部の平面は
  z=0)。`pos` はテーブル天板中心の [x, y] です。
- `props`: 静的な `ObjectSpec`(objects と同じフィールドを持ちますが自由関節を
  持たず、ランダム化もされません)。

## `objects[]`

| field | type | default | meaning |
| --- | --- | --- | --- |
| `name` | str | — | unique; also the pick target for scripted demos |
| `type` | str | `box` | `box` \| `sphere` \| `cylinder` \| `mesh` \| `cloth` |
| `size` | [m] | — | box: half-extents xyz; sphere: [r]; cylinder: [r, half-h]; cloth: [width, height] |
| `pos` | [m] | — | spawn position (cloth: grid center) |
| `rgba` | [0-1]×4 | red | color |
| `mass` | kg | 0.05 | total mass |
| `file` | path | — | `type: mesh` only — any STL/OBJ |
| `scale` | float | 1.0 | mesh scale |
| `static` | bool | false | fixed to the world (no free joint, no randomization) |
| `pos_noise` | [m] | null | per-reset uniform noise ±[x, y, z] |
| `yaw_range` | [rad] | null | per-reset uniform yaw (rigid objects only) |
| `cloth` | map | see below | `type: cloth` physics parameters |

### 布(`type: cloth`)

MuJoCo 3.1 以降がネイティブでサポートする変形可能なシェル(`flexcomp` +
組み込みの弾性モデル — プラグイン不要)です。ブラウザのビュー、すべてのシナリオ
カメラ、したがってデータセットにもレンダリングされます。

```yaml
cloth: {resolution: 9, young: 3.0e4, poisson: 0.1, thickness: 0.01, damping: 0.01}
```

| param | default | stable range | meaning |
| --- | --- | --- | --- |
| `resolution` | 9 | 5–15 | grid vertices per side (9 → 81 vertices, 128 triangles) |
| `young` | 3e4 | 1e3 (silk-like) – 1e6 (stiff tarp) | Young's modulus, Pa |
| `poisson` | 0.1 | 0–0.45 | lateral contraction |
| `thickness` | 0.01 | 0.002–0.02 | shell thickness, m |
| `damping` | 0.01 | 0.001–0.1 | vertex velocity damping |

`resolution` を上げるほど折り目は細かくなりますが物理演算は遅くなります
(コストはおおよそ2乗で増加します)。`pos_noise` はリセットのたびに布全体を
平行移動させます。`yaw_range` は布には適用されません(無視されます)。

## `cameras[]`

```yaml
cameras:
  - {name: front, pos: [0.55, -0.4, 0.35], lookat: [0.25, 0, 0.05], fovy: 50}   # world-fixed
  - {name: wrist_cam}                                                           # built into the arm
  - {name: grip_cam, attach_to: gripper, pos: [0, -0.07, 0.01],
     lookat: [0.02, 0, -0.22], fovy: 75}                                        # eye-in-hand
```

- カメラはいくつでも追加できます。それぞれが `observation.images.<name>`
  というデータセットの特徴量になり、ポリシーの入力にもなります。解像度はグローバル
  設定(`--render-width/--render-height`)です。
- `attach_to`: `base, shoulder, upper_arm, lower_arm, wrist, gripper` のいずれか。
  指定するとそのボディのフレーム内で `pos` と `lookat` が解釈され、カメラはアームと
  一緒に動きます。グリッパーの場合、顎(jaw)はそのボディフレームの −z 方向を
  向きます。
- アームのMJCFに既に存在する名前(例: `wrist_cam`)を指定すると、そのカメラを
  そのまま再利用します。

## 実例

1. **最小構成のピックシーン** — [examples/pick_cube.yaml](../examples/pick_cube.yaml):
   ランダム化された立方体1個、カメラ3台。
2. **雑然としたテーブル上** — テーブル + props + 布 + 剛体物体 + カメラ5台
   (うち2台はアーム取付):
   [examples/tabletop_cloth.yaml](../examples/tabletop_cloth.yaml)。
3. **布のみのシーン**(折りたたみ系タスク向け): `type: cloth` の物体を1つだけ
   置き、GUIでテレオペのデモを記録します。スクリプトによるピックデモには
   剛体の物体が最低1つ必要です。

## 他のロボットへの対応(拡張ポイント)

シーンビルダーは任意のMuJoCoアームモデルを中心に構成されています:
`build_model(scenario, mjcf_path=...)` と `Kinematics(mjcf_path=...)` は別の
MJCFを受け付けます。移植時に満たす必要がある現状の前提は、qpos内でアーム関節が
先頭に来る形で6個の位置サーボアクチュエータがあること、`gripperframe` という
TCPサイトがあること、実機転送のために lerobot 形式の `<motor>.pos` という
命名規則に従っていることです。これらはデータ上の規約であり(`config.py`)、
ハードコードされたロジックではありません。別のアームへの移植は、MJCFを差し替え、
`ARM_JOINTS`/リミットを1箇所で更新するだけで済みます。
