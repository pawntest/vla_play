# so101-tool

Control a [LeRobot SO-101](https://huggingface.co/docs/lerobot/so101) robot arm with a
**live browser 3D preview**, classic **rule-based motion** (tool-frame jogging, move-to-point
via IK, joint moves) and **AI control** (lerobot policy inference and natural-language
commands via the Claude API).

**Sim-first**: everything — the 3D view, IK, motion primitives, natural-language control —
runs fully in simulation with zero hardware. Connecting a real SO-101 is an add-on backend:
the same UI then mirrors the physical arm's joints in 3D (input) and sends every command to
the servos (output).

日本語のクイックスタートは [docs/README.ja.md](docs/README.ja.md) にあります。

## Quickstart (simulation, no hardware)

```bash
pip install -e ".[dev]"          # Python >= 3.10
so101-tool run --backend sim
# open http://localhost:8080
```

You get a browser 3D scene with the arm and a control panel. **Everything the tool can
do is available from this UI** — no extra terminals needed:

- **Scenario** — load a scene YAML, randomize object poses, add/remove objects
  interactively (incl. **drag gizmos** and **click-in-scene placement**), edit cameras
  (world-fixed or arm-attached) and the environment (table, floor), save back to YAML
- **Mode** — `idle` / `mirror` (preview only) / `rule` (motion primitives) / `policy`
- **Joints / Cartesian** — sliders, *Home*, target gizmo + *Go to target*, tool-frame jogs
- **Record dataset** — start/save/discard LeRobotDataset episodes while you drive the arm
- **Scripted demos** — auto-generate pick demonstrations into the same dataset
- **Policy (AI)** — load a checkpoint (local path or hub id) and run it live
- **Train** — launch/stop lerobot-train with streamed logs, or export a Colab notebook
- **Natural language** — chat box (with ANTHROPIC_API_KEY)
- **EMERGENCY STOP** — latching; no command reaches the robot until reset

Check IK health any time (no hardware): `so101-tool ik-check --n 100`

## Python API

Everything is scriptable — `so101_tool.api` is the library the CLI and GUI are built on:

```python
from so101_tool import api

with api.Session(scenario="examples/tabletop_cloth.yaml") as sess:
    sess.move_to([0.25, 0.0, 0.10])            # IK move (same safety filter as the GUI)
    sess.gripper(0.0)
    print(sess.state().tcp_position, sess.object_pose("cube"))
    sess.generate_demos("my/pick", episodes=30, root="data/pick")
api.train(dataset="data/pick", policy="act")
```

`api.Session` also drives the real robot (`backend="real"`) and can serve the browser UI
on top of itself (`sess.open_ui()`).

## Real robot

Requires Python >= 3.12 (lerobot). See [docs/hardware.md](docs/hardware.md) for port
permissions, calibration and the first-connection safety checklist.

```bash
pip install -e ".[real]"
so101-tool run --backend real --port /dev/ttyACM0 --robot-id my_follower
```

The 3D preview is now synchronized with the physical arm: `mirror` mode shows the real
joint positions live (hand-pose the arm with torque off to verify the model matches);
`rule` mode drives the servos through the same safety filter as the sim.

## Natural-language control (Claude)

```bash
pip install -e ".[nl]"
export ANTHROPIC_API_KEY=sk-ant-...
so101-tool run --backend sim        # or real
```

A *Natural language* box appears in the panel. Try: *"move the gripper 5 cm forward"*,
*"go to x=250 y=0 z=150"*, *"open the gripper and go home"*. The model only gets the same
rule-based primitives you have as buttons — every motion passes through the safety filter,
and steps larger than 25 cm are refused.

## Trained policy inference (ACT / SmolVLA)

```bash
pip install -e ".[policy]"          # Python >= 3.12, pulls torch
so101-tool run --backend real --port /dev/ttyACM0 \
    --policy-path lerobot/smolvla_base --policy-task "pick up the cube"
```

Switch the mode to `policy` in the panel. Policies need camera observations, so this mode
requires the real backend (documented limitation of the sim backend).

## Remote teleop (Codespaces / Brev / any SSH box)

Run the app on a remote container and drive it with the leader arm on your desk —
securely (127.0.0.1 + session token, reached only through your SSH tunnel):

```bash
remote$ so101-tool run --scenario examples/pick_cube.yaml --teleop
laptop$ ssh -L 8765:localhost:8765 <remote>
laptop$ so101-tool teleop-client --connect localhost:8765 --token <printed> --port /dev/ttyACM0
```

See [docs/teleop_remote.md](docs/teleop_remote.md). Test without hardware: `--source sine`.

`--link {to_sim,to_real,both}` couples the real arm and the MuJoCo sim in either
or both directions (実機→シム / シム→実機 / 双方向), switchable live from the UI
header — works with a local serial arm (`--backend real`) or over the same SSH
tunnel (client runs with `--source follower`).

## Imitation learning: record → train → deploy

The tool covers the full imitation-learning loop, sim-first and with a free-GPU cloud
option — see [docs/data_and_training.md](docs/data_and_training.md) for the whole story.
The entire loop below can also be driven from the browser UI (Scenario → Record/Scripted
demos → Train → Policy folders); the CLI equivalents are:

```bash
# 1. describe the scene + task in YAML (objects, cameras, instruction)
cp examples/pick_cube.yaml my_task.yaml

# 2. record demonstrations as a LeRobotDataset (by hand in the 3D GUI, or scripted)
so101-tool run --scenario my_task.yaml --record my_task --record-root data/my_task
so101-tool scripted-demos --scenario my_task.yaml --episodes 50 \
    --dataset my_task --root data/my_task

# 3. train (wraps lerobot-train; ACT / SmolVLA / Diffusion)
so101-tool train --dataset data/my_task --policy act          # local GPU box
so101-tool train --dataset data/my_task --policy act \
    --push-dataset --dataset-hub-repo you/so101-my-task \
    --emit-colab train.ipynb                                  # free GPU: open in Colab

# 4. run the trained policy (sim or real)
so101-tool run --scenario my_task.yaml \
    --policy-path outputs/train/checkpoints/last/pretrained_model
```

Datasets and checkpoints are standard LeRobot formats, so anything trained here runs
anywhere lerobot runs (and vice versa).

## Architecture (short version)

| Module | Role |
| --- | --- |
| `api.py` | public Python API: `Session` (motion, scenes, demos, policies) + `train()` |
| `kinematics.py` | MuJoCo FK + [mink](https://github.com/kevinzakka/mink) differential IK (TCP = `gripperframe` site) |
| `scenario.py` | YAML scenes: environment (table/floor/props), rigid + **cloth** objects w/ randomization, world/arm-attached cameras |
| `robot/sim.py`, `robot/physics_sim.py`, `robot/lerobot_backend.py` | Interchangeable backends behind `RobotInterface`: kinematic sim, contact-physics sim (rendered cameras), real robot |
| `control/loop.py` | 50 Hz thread owning all robot I/O: command queue in, immutable snapshots out, latching e-stop |
| `control/safety.py` | Joint-limit clamp, velocity limit, floor check on every write |
| `viz/` | viser 3D scene driven by MuJoCo body poses + GUI panel |
| `nl/agent.py` | Claude tool-use agent mapping language → motion primitives |
| `policy/runner.py` | lerobot policy checkpoint → joint targets (sim or real cameras) |
| `data/recorder.py`, `demo/scripted.py` | LeRobotDataset recording; deterministic scripted demo generation |
| `training.py` | `so101-tool train`: local lerobot-train wrapper + Colab notebook export |

Details in [docs/architecture.md](docs/architecture.md). The arm model is
[`robotstudio_so101` from mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
(Apache-2.0), vendored under `src/so101_tool/assets/so101/`.

## Safety notes

- Every write to the robot (sim or real) goes through the safety filter: joint limits,
  velocity clamp, floor keep-out.
- The **EMERGENCY STOP** button latches: motion commands are refused until *Reset e-stop*.
- On first real-robot use, follow the sign-check procedure in
  [docs/hardware.md](docs/hardware.md) before commanding any motion.

## See also

- [docs/why_this_tool.md](docs/why_this_tool.md) — honest comparison with Isaac Sim & friends
- [docs/scenario_reference.md](docs/scenario_reference.md) — full scene YAML reference
  (environment, objects incl. **cloth**, multi/attached cameras, randomization)

## Development

```bash
.venv/bin/pytest          # no hardware needed
.venv/bin/ruff check src tests
```
