# Architecture

日本語版: [docs/ja/architecture.md](ja/architecture.md)

## Units & conventions (everywhere inside the package)

- Arm joints: `q` = numpy `(5,)`, **radians**, order `[shoulder_pan, shoulder_lift,
  elbow_flex, wrist_flex, wrist_roll]` (= lerobot motor names = MJCF joint names).
- Gripper: fraction `0..1` (0 closed, 1 open).
- Poses: position (m) + `wxyz` quaternion in the robot base frame; TCP is the MJCF site
  `gripperframe` between the jaws.
- Only `LeRobotBackend` converts units (degrees / 0–100), via `JointMap` (which also owns
  per-joint signs/offsets and velocity limits).

## Thread model

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

- The control loop is the **only** thing that touches the robot backend.
- Producers block on `cmd.wait()`; the loop always calls `finish()` exactly once
  (completed / preempted / timeout / estop / shutdown), so nothing hangs.
- `Kinematics` is not thread-safe: the loop and the render path each own an instance.
- E-stop is a latching `threading.Event`, honored before any write.

## Modes

| Mode | Writes? | Behavior |
| --- | --- | --- |
| `idle` | no | snapshots only |
| `mirror` | no | 3D preview follows the (real) robot; used for hand-posing/sign checks |
| `rule` | yes | motion primitives; MoveL/JogTool run a differential-IK servo per tick |
| `policy` | yes | `PolicyRunner.step()` provides targets each tick |

The NL agent is not a mode: it is a producer that translates language into the same
primitives (and switches to `rule` first). Its motions obey the same safety filter.

## IK on a 5-DOF arm

A 5-DOF arm cannot realize arbitrary 6-DOF poses. `Kinematics.ik*` therefore uses mink
with `FrameTask(position_cost=1.0, orientation_cost=0.3)` + `PostureTask(1e-2)`: position
is hard, orientation trades off gracefully; `wxyz=None` drops orientation entirely.
Measured on 100 random reachable targets: 98% converge < 5 mm (median 1.2 mm), ~1 ms/solve.

## Sim backend

`SimBackend` is a velocity-limited kinematic integrator — the same first-order tracking a
position-controlled STS3215 performs — not a physics sim. Deterministic (injectable clock),
stable at any dt, and behaviorally close to the real servos for preview purposes. Contact
physics could later be swapped in behind the same `RobotInterface`.

## 3D view

`RobotView` builds viser scene nodes directly from the MuJoCo model (one frame per body,
meshes from `mjModel` vertex/face arrays — no URDF, no file re-parsing) and per tick writes
`data.xpos/xquat` into the frame handles after `mj_kinematics`. MuJoCo and viser share the
wxyz quaternion convention, so transforms pass through untouched.
