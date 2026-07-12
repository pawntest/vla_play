# Scenario YAML reference

**日本語**: シーン(環境・物体・カメラ・タスク)を定義するYAMLの完全リファレンスです。
座標系はアーム基部が原点、+X前方、+Z上、単位はメートル/ラジアン。GUIの
Scenario→Edit objects/Cameras/Environment パネルは同じスキーマを対話的に編集し、
「Save scenario YAML」でこの形式に書き出します。

Everything below can also be edited interactively in the GUI (drag gizmos, click-to-add,
camera/environment editors) and saved back to YAML. Loader/builder:
`so101_tool.scenario.load_scenario / build_model` (`api.py` re-exports them).

## Top level

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

- `floor.checker: true` renders a two-tone checker (second color `rgba2`, default =
  darkened `rgba`). With a `table`, the floor drops 0.4 m below the tabletop.
- `table`: a static slab whose TOP surface is at `height` (the arm base plane is z=0);
  `pos` is the tabletop center [x, y].
- `props`: static `ObjectSpec`s (same fields as objects, no free joint, never randomized).

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

### Cloth (`type: cloth`)

MuJoCo ≥3.1 native deformable shells (`flexcomp` + built-in elasticity — no plugin).
Rendered in the browser view, in every scenario camera, and therefore in datasets.

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

Higher `resolution` means finer folds but slower physics (cost grows ~quadratically).
`pos_noise` translates the whole cloth per reset; `yaw_range` is ignored for cloth.

## `cameras[]`

```yaml
cameras:
  - {name: front, pos: [0.55, -0.4, 0.35], lookat: [0.25, 0, 0.05], fovy: 50}   # world-fixed
  - {name: wrist_cam}                                                           # built into the arm
  - {name: grip_cam, attach_to: gripper, pos: [0, -0.07, 0.01],
     lookat: [0.02, 0, -0.22], fovy: 75}                                        # eye-in-hand
```

- Any number of cameras; every one becomes an `observation.images.<name>` dataset feature
  and a policy input. Resolution is global (`--render-width/--render-height`).
- `attach_to`: one of `base, shoulder, upper_arm, lower_arm, wrist, gripper` — `pos` and
  `lookat` are then in that body's frame and the camera moves with the arm. For the
  gripper, the jaw points along −z of the body frame.
- A name that already exists in the arm MJCF (e.g. `wrist_cam`) reuses that camera.

## Worked examples

1. **Minimal pick scene** — [examples/pick_cube.yaml](../examples/pick_cube.yaml):
   one randomized cube, three cameras.
2. **Cluttered tabletop** — table + props + cloth + rigid objects + 5 cameras (2 attached):
   [examples/tabletop_cloth.yaml](../examples/tabletop_cloth.yaml).
3. **Cloth-only scene** (folding-style tasks): keep a single `type: cloth` object and
   record teleoperated demos through the GUI; scripted pick demos need a rigid first object.

## Other robots (extension point)

The scene builder composes around any MuJoCo arm model: `build_model(scenario, mjcf_path=...)`
and `Kinematics(mjcf_path=...)` accept a different MJCF. Current assumptions that a port
must satisfy: 6 actuated position servos with the arm joints first in qpos, a `gripperframe`
TCP site, and lerobot-style `<motor>.pos` naming for real-robot transfer. These are data
conventions (`config.py`), not hard-coded logic — porting a different arm means swapping the
MJCF + updating `ARM_JOINTS`/limits in one place.
