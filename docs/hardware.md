# Real SO-101 setup & safety

日本語版: [docs/ja/hardware.md](ja/hardware.md)

## Prerequisites

- Python >= 3.12 and `pip install -e ".[real]"` (installs `lerobot[feetech]`).
- The arm assembled and powered per the
  [lerobot SO-101 guide](https://huggingface.co/docs/lerobot/so101).

## Serial port

```bash
ls /dev/ttyACM*            # usually /dev/ttyACM0
sudo usermod -aG dialout $USER   # then log out/in — avoids sudo for the port
```

If several devices are attached, unplug/replug the arm and diff `ls /dev/ttyACM*`.

## Calibration

This tool reads/writes joint angles in degrees through lerobot, so the arm must be
calibrated with lerobot first (once per robot, stored under `~/.cache/huggingface/lerobot`):

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower
```

Use the same `--robot-id` when launching `so101-tool run --backend real`.

## First-connection checklist (do these in order)

1. **Sign check in MIRROR mode.** Launch with the arm powered, switch Mode to `mirror`.
   The loop only *reads* in this mode. Gently push each joint (if torque prevents it,
   lerobot's torque can be disabled — or just watch while moving the arm with the leader
   arm / by hand at low torque). The 3D preview must move the **same direction** as the
   physical joint. If a joint moves the opposite way, or is offset, fix it in data — edit
   `JointMap.signs` / `offsets_deg` in `src/so101_tool/config.py` (per-joint, internal =
   `(real_deg − offset) / sign`) — and re-check. Do not proceed until MIRROR matches.
2. **Small joint move.** Switch to `rule`, set speed ≈ 0.2, move one joint slider by
   ~0.2 rad, press *Move to sliders*. Keep a hand near the power switch.
3. **Cartesian move.** Press *Snap target to TCP*, drag the gizmo a few centimeters,
   *Go to target (position)*.
4. **Natural language (optional).** With `ANTHROPIC_API_KEY` set: "open the gripper".

## Safety behavior

- Every command (GUI, natural language, policy) passes the safety filter: joint-limit
  clamp, per-joint velocity clamp (default 2.0 rad/s × 1.5 margin), floor keep-out.
- **EMERGENCY STOP** latches and stops writes within one control tick (20 ms). The servos
  hold their last position (STS3215 position mode); power off for a true hard stop.
- `disable_torque_on_disconnect=True`: quitting the tool relaxes the arm — support it or
  home it first so it doesn't drop.
- The first observation after connect is sanity-checked: values that look like radians
  instead of degrees abort the connection instead of commanding a violent move.

## Known limitations

- `mirror` mode does not currently toggle torque automatically;
  `LeRobotBackend.set_torque(False)` exists for scripting it.
- POLICY mode requires cameras, hence the real backend; configure them via
  `AppConfig.cameras` (lerobot camera config dicts) when scripting, e.g.
  `{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}`.
