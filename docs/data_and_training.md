# Data collection & training: the imitation-learning workflow

日本語版: [docs/ja/data_and_training.md](ja/data_and_training.md)

This page walks the full loop end to end: **define a scene → record demonstrations →
train a policy (locally or on a free cloud GPU) → run it on the arm**. Everything up to
the real-robot steps works with zero hardware.

The training front-end lives in `src/so101_tool/training.py`; it drives lerobot's
standard `lerobot-train` entry point (lerobot >= 0.5.1, Python >= 3.12), so checkpoints
are plain LeRobot policies usable anywhere.

## 0. Environments at a glance

| Step | Needs |
| --- | --- |
| Scenario + sim recording | `pip install -e ".[dev]"` (Python >= 3.10) |
| Real-robot recording | `pip install -e ".[real]"` (Python >= 3.12), see [hardware.md](hardware.md) |
| Local training | `pip install -e ".[policy]"` (Python >= 3.12, pulls torch) |
| Cloud training | a browser (Google Colab, free GPU) + an HF account |

## 1. Define a scenario

A scenario is a small YAML file describing the objects, cameras and task around the arm.
Start from [`examples/pick_cube.yaml`](../examples/pick_cube.yaml); the full schema
(box/sphere/cylinder/mesh objects, per-reset randomization, camera placement) is in the
module docstring of [`src/so101_tool/scenario.py`](../src/so101_tool/scenario.py):

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

The `task` string and the camera names become the language instruction and the
observation keys of the recorded dataset, so choose them before you start recording.

## 2. Collect demonstrations

Demonstrations are saved in the standard **LeRobotDataset** format (a directory with
`meta/`, `data/`, `videos/`), so they train and share exactly like any other LeRobot
dataset.

**In simulation, by hand (GUI recording):** run the scenario with recording enabled and
demonstrate the task with the Cartesian gizmo / jog buttons; each episode is saved with
the scenario cameras rendered as video streams:

```bash
so101-tool run --scenario examples/pick_cube.yaml --record pick_cube --record-root data/pick_cube
```

**In simulation, automatically (scripted demos):** for pick-and-place style tasks the
tool can generate demonstrations by planning with the simulator's ground-truth object
poses — useful to bootstrap a dataset in minutes:

```bash
so101-tool scripted-demos --scenario examples/pick_cube.yaml \
    --episodes 50 --dataset pick_cube --root data/pick_cube
```

(Both commands above are part of the recording pipeline currently being finalized in
`so101_tool/data`; flags may still move slightly.)

**On the real robot:** connect the arm and cameras (see [hardware.md](hardware.md)) and
record with the real backend, or use lerobot's own `lerobot-record` teleop recorder —
the resulting dataset is the same format and trains identically.

Aim for **30–50+ episodes** with varied object positions (that's what `pos_noise` /
`yaw_range` in the scenario are for). Fewer than ~20 episodes rarely produces a usable
ACT policy.

## 3. Train locally

```bash
pip install -e ".[policy]"     # Python >= 3.12
so101-tool train --dataset data/pick_cube --policy act
```

This shells out to `lerobot-train` with sensible defaults (20k steps, batch size 8,
wandb off, device auto-detected: cuda → mps → cpu). Useful flags, mirroring
`TrainArgs` in `training.py`:

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

Anything not covered by a flag can be passed verbatim through `--extra` — the full
option surface is `lerobot-train --help`.

Note: ACT at 20k steps is an overnight job on CPU but ~1–2 h on a modest GPU. If you
don't have a local GPU, use the cloud path below.

## 4. Train in the cloud (free GPU)

**Google Colab (recommended, free T4 GPU):** one command pushes your local dataset to
the Hugging Face hub (cloud machines can't see your disk — set `HF_TOKEN` to a *write*
token first) and emits a ready-to-run notebook:

```bash
export HF_TOKEN=hf_...
so101-tool train --dataset data/pick_cube --policy act \
    --push-dataset --dataset-hub-repo your-hf-user/so101-pick-cube \
    --emit-colab train_pick_cube.ipynb
```

Already pushed? Skip the upload and just emit the notebook:

```bash
so101-tool train --dataset your-hf-user/so101-pick-cube --policy act \
    --emit-colab train_pick_cube.ipynb
```

The notebook installs lerobot, logs you into HF, runs the *exact same*
`lerobot-train` command you would run locally (device pinned to `cuda`), and pushes the
trained checkpoint back to the hub. Remember to select *Runtime → Change runtime type →
T4 GPU* before running.

**Any other GPU box** (rented server, lab machine, HF Jobs): the tool adds nothing
machine-specific, so the same two lines work anywhere:

```bash
pip install "so101-tool[policy]"     # Python >= 3.12
so101-tool train --dataset your-hf-user/so101-pick-cube --policy act \
    --push-to-hub --hub-repo your-hf-user/so101-act-pick-cube
```

## 5. Run the trained policy

Point the runner at the checkpoint directory (local training writes
`<output-dir>/checkpoints/last/pretrained_model`) or at the hub repo you pushed to:

```bash
# in sim, against the same scenario
so101-tool run --scenario examples/pick_cube.yaml \
    --policy-path outputs/train/checkpoints/last/pretrained_model

# on the real robot
so101-tool run --backend real --port /dev/ttyACM0 \
    --policy-path your-hf-user/so101-act-pick-cube --policy-task "pick up the red cube"
```

Switch the panel Mode to `policy`. Every policy action still passes through the same
safety filter (joint limits, velocity clamp, floor keep-out, latching e-stop) as manual
control.

## End-to-end cheat sheet

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
