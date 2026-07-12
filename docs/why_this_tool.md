# Why MuJoCo + viser (and not Isaac Sim)?

**日本語要約**: 本ツールは「SO-101で模倣学習のデータを集めて学習する」という目的に最適化した
軽量スタックです。pipで数分で入り、GPU不要・ヘッドレスで動き、シーンはYAML数十行、
データは最初からLeRobotDataset形式で出ます。Isaac Sim が勝るのは、フォトリアルなレンダリング
(RTX)・数千環境並列のGPU強化学習(Isaac Lab)・USDアセット資産・LiDAR等のセンサモデルが
必要な場合です。その代わりRTX GPU必須・数十GBのインストール・起動が重い・SO-101対応や
LeRobot形式のデータパイプラインは自作になります。**卓上マニピュレーションの模倣学習を
高速に回す**なら本ツール、**大規模RLやフォトリアルsim2real**ならIsaac(またはManiSkill)が適切です。

## What this stack optimizes for

| | this tool (MuJoCo + viser) | Isaac Sim / Isaac Lab |
| --- | --- | --- |
| Install | `pip install`, ~a few hundred MB | multi-GB download, Omniverse stack |
| Hardware | any laptop, CPU-only OK (OSMesa headless) | NVIDIA RTX GPU required |
| Startup | seconds | minutes |
| Physics for manipulation | MuJoCo — de-facto standard for contact-rich research; native deformables (cloth) since 3.x | PhysX 5; strong rigid-body + GPU parallelism |
| Rendering | rasterized (viser browser view + MuJoCo offscreen cameras) | photorealistic RTX ray tracing |
| Massive parallel RL | no (single scene) | yes — thousands of envs on one GPU (Isaac Lab) |
| SO-101 support | built-in: arm model, calibration conventions, same UI drives the real robot | port the URDF and build everything yourself |
| Dataset output | LeRobotDataset natively — feeds `lerobot-train` directly | custom exporter needed |
| Scene definition | ~30 lines of YAML or interactive GUI | USD scene composition / Python API |
| Determinism of data generation | fully deterministic fake-clock generation | achievable, more moving parts |

**Honest take**: if your goal is photoreal sim2real for vision, city/warehouse-scale scenes,
lidar/depth sensor simulation, or 4096-env PPO — use Isaac Lab (or
[ManiSkill3](https://github.com/haosulab/ManiSkill), which sits in between: GPU-parallel,
manipulation-focused, lighter than Isaac). [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
is a promising new GPU sim but still stabilizing; PyBullet is simple but dated for
contact-rich work.

For **one SO-101 on a desk, imitation learning, tens-to-hundreds of demos**, none of that
horsepower is the bottleneck — iteration speed is. Here the whole loop
(edit scene → generate demos → train → run policy → drive the real arm) is one tool,
one command each, and runs on the machine you already have. lerobot itself ships
MuJoCo-based sim environments, so you stay inside the ecosystem your training stack expects.

## "Can YAML really express scenes freely?"

The YAML schema (full reference: [scenario_reference.md](scenario_reference.md)) covers:

- **Objects**: boxes/spheres/cylinders, **arbitrary STL/OBJ meshes**, and **deformable
  cloth** (MuJoCo native flex elasticity — Young's modulus, Poisson ratio, thickness,
  damping) — each with mass, color, friction, static/free, and **per-reset randomization**
  (position noise, yaw range) for domain randomization.
- **Environment**: floor (solid/checker), a table under the arm, arbitrary static props
  (walls, shelves, backdrops), light intensity.
- **Cameras**: any number, world-fixed (pos/lookat/fov) or **attached to any arm body**
  (base/shoulder/upper_arm/lower_arm/wrist/gripper) for eye-in-hand views; all of them are
  rendered into the dataset and fed to policies.
- Whatever the YAML can't express, the escape hatch is the MJCF itself:
  `build_model()` accepts any MuJoCo arm model path, and everything the builder adds is
  standard MJCF you can extend.

What it does **not** try to be: a photoreal renderer, a fluid/soft-body-everything sim,
or a multi-robot world. Scope is a tabletop and one arm, done end to end.
