# なぜ MuJoCo + viser なのか(Isaac Simではなく)?

> 英語版(正): [../why_this_tool.md](../why_this_tool.md) — 差異があれば英語版が優先です。

本ツールは「SO-101で模倣学習のデータを集めて学習する」という目的に最適化した
軽量スタックです。pipで数分でインストールでき、GPU不要・ヘッドレスで動作し、
シーンはYAML数十行で記述でき、データは最初からLeRobotDataset形式で出力されます。
Isaac Simが優れているのは、フォトリアルなレンダリング(RTX)・数千環境並列の
GPU強化学習(Isaac Lab)・USDアセット資産・LiDARなどのセンサモデルが必要な
場合です。その代わり、RTX GPUが必須で、数十GBのインストールが必要になり、
起動も重く、SO-101対応やLeRobot形式のデータパイプラインは自分で作る必要が
あります。**卓上マニピュレーションの模倣学習を高速に回す**なら本ツールが、
**大規模RLやフォトリアルなsim2real**ならIsaac(またはManiSkill)が適切です。

## このスタックが最適化しているもの

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

**正直な評価**: もし目的がビジョン向けのフォトリアルsim2real、都市/倉庫規模の
シーン、LiDAR/深度センサのシミュレーション、あるいは4096環境規模のPPOで
あるなら — Isaac Labを使ってください(あるいは、その中間に位置する
[ManiSkill3](https://github.com/haosulab/ManiSkill): GPU並列でマニピュレーションに
特化しており、Isaacより軽量です)。[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
は有望な新しいGPUシムですがまだ安定化の途上にあり、PyBulletはシンプルですが
接触の多い作業には古さを感じさせます。

**卓上に1台のSO-101を置いて、数十〜数百件のデモで模倣学習を行う**という
用途では、そうした馬力はボトルネックになりません。ボトルネックになるのは
イテレーション速度です。ここでは一連の流れ(シーンを編集する → デモを
生成する → 学習する → ポリシーを実行する → 実機アームを動かす)がすべて
1つのツール、それぞれ1コマンドで完結し、既に手元にあるマシン上で動きます。
lerobot自体がMuJoCoベースのシムを同梱しているため、学習スタックが前提と
するエコシステムの中にとどまることができます。

## 「YAMLで本当に自由にシーンを表現できるのか?」

YAMLスキーマ(完全なリファレンスは [scenario_reference.md](scenario_reference.md)
を参照)は以下をカバーしています:

- **オブジェクト**: box/sphere/cylinder、**任意のSTL/OBJメッシュ**、そして
  **変形可能な布**(MuJoCoネイティブのflex弾性 — ヤング率、ポアソン比、
  厚み、減衰)。それぞれに質量、色、摩擦、static/free、そして**リセットごとの
  ランダム化**(位置ノイズ、yaw範囲)によるドメインランダム化が設定できます。
- **環境**: 床(単色/チェッカー)、アームの下のテーブル、任意の静的props
  (壁、棚、背景)、照明の強さ。
- **カメラ**: 任意の数を設置でき、ワールド固定(pos/lookat/fov)、または
  **任意のアームボディに取り付け**(base/shoulder/upper_arm/lower_arm/wrist/
  gripper)てeye-in-handビューにできます。すべてデータセットにレンダリング
  され、ポリシーへ渡されます。
- YAMLで表現できないものについての逃げ道はMJCF自体です:
  `build_model()` は任意のMuJoCoアームモデルのパスを受け付け、ビルダーが
  追加するものはすべて標準的なMJCFなので自由に拡張できます。

本ツールが目指していないもの: フォトリアルなレンダラー、流体/ソフトボディを
何でもシミュレートできるシム、マルチロボットのワールド。スコープはあくまで
卓上と1本のアームを、一貫してエンドツーエンドで扱うことです。
