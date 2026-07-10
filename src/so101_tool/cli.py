"""Command-line entry point: `so101-tool run` / `so101-tool ik-check`."""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import tyro

from .config import ARM_LIMITS_HI, ARM_LIMITS_LO, AppConfig


@dataclasses.dataclass
class RunArgs:
    """Start the control loop + 3D preview web UI."""

    backend: str = "sim"
    """Robot backend: 'sim' (no hardware needed) or 'real' (SO-101 via lerobot)."""
    port: str = "/dev/ttyACM0"
    """Serial port of the real robot (backend=real)."""
    robot_id: str = "so101_follower"
    """lerobot calibration id (backend=real)."""
    viser_port: int = 8080
    """Port for the browser 3D UI."""
    control_hz: float = 50.0
    render_hz: float = 30.0
    policy_path: str | None = None
    """lerobot policy checkpoint (hub id or local path) for POLICY mode."""
    policy_task: str | None = None
    """Task instruction passed to VLA policies (e.g. 'pick up the cube')."""
    nl_model: str | None = None
    """Claude model for natural-language control (default: claude-opus-4-8)."""
    no_nl: bool = False
    """Disable natural-language control even if ANTHROPIC_API_KEY is set."""
    scenario: str | None = None
    """Scenario YAML (objects/cameras/task): runs the PHYSICS simulation."""
    record: str | None = None
    """Record demonstrations to this LeRobotDataset repo_id (adds a GUI panel)."""
    record_fps: int = 15
    """Dataset sampling rate."""
    record_root: str | None = None
    """Local directory for the dataset (default: HF cache)."""
    record_resume: bool = False
    """Append episodes to an existing dataset."""
    render_width: int = 320
    render_height: int = 240
    """Scenario camera resolution (policy obs + recording)."""


def _run(args: RunArgs) -> None:
    import viser
    from loop_rate_limiters import RateLimiter

    from .control.loop import ControlLoop
    from .kinematics import Kinematics
    from .viz.panel import ControlPanel
    from .viz.robot_view import RobotView

    config = AppConfig(
        backend=args.backend,
        port=args.port,
        robot_id=args.robot_id,
        viser_port=args.viser_port,
        control_hz=args.control_hz,
        render_hz=args.render_hz,
        policy_path=args.policy_path,
        policy_task=args.policy_task,
    )
    if args.nl_model:
        config.nl_model = args.nl_model

    scenario = None
    if args.scenario:
        from .robot.physics_sim import PhysicsBackend
        from .scenario import load_scenario

        scenario = load_scenario(args.scenario)
        robot = PhysicsBackend(
            scenario,
            config.joint_map,
            render_width=args.render_width,
            render_height=args.render_height,
        )
        print(f"scenario: {scenario.name} — task: {scenario.task or '-'}")
    elif config.backend == "real":
        from .robot.lerobot_backend import LeRobotBackend

        robot = LeRobotBackend(config)
    elif config.backend == "sim":
        from .robot.sim import SimBackend

        robot = SimBackend(config.joint_map)
    else:
        raise SystemExit(f"unknown backend {config.backend!r} (use 'sim' or 'real')")

    print(f"connecting to {config.backend} robot…")
    robot.connect()

    policy_runner = None
    if config.policy_path:
        from .policy.runner import PolicyRunner

        policy_runner = PolicyRunner(config, robot)

    loop = ControlLoop(robot, Kinematics(), config, policy_runner=policy_runner)
    loop.start()

    server = viser.ViserServer(port=config.viser_port)
    kin_viz = Kinematics()  # render thread owns its own instance
    view = RobotView(server, robot.model if scenario is not None else kin_viz)

    nl_agent = None
    if not args.no_nl and os.environ.get("ANTHROPIC_API_KEY"):
        from .nl.agent import NLAgent

        nl_agent = NLAgent(loop, kin_viz, model=config.nl_model)
        print(f"natural-language control enabled ({config.nl_model})")

    recorder = None
    if args.record:
        if not hasattr(robot, "get_camera_frames"):
            raise SystemExit("--record needs cameras: use --scenario or --backend real")
        from .viz.recording import RecorderBridge

        def _make_recorder():
            from .data.recorder import DatasetRecorder

            frames = robot.get_camera_frames()
            cameras = {name: img.shape[:2] for name, img in frames.items()}
            return DatasetRecorder(
                repo_id=args.record,
                fps=args.record_fps,
                cameras=cameras,
                joint_map=config.joint_map,
                root=args.record_root,
                task=scenario.task if scenario else "",
                resume=args.record_resume,
            )

        recorder = RecorderBridge(_make_recorder, robot.get_camera_frames, args.record_fps)
        print(f"recording to dataset: {args.record} (fps {args.record_fps})")

    panel = ControlPanel(
        server, loop, kin_viz, config,
        nl_agent=nl_agent, recorder=recorder, scenario=scenario, backend=robot,
    )

    print()
    print(f"  ▶ 3D preview: http://localhost:{config.viser_port}")
    print("    Ctrl-C to exit.")
    print()

    rate = RateLimiter(frequency=config.render_hz, warn=False)
    try:
        while True:
            snap = loop.snapshot()
            if snap is not None:
                if snap.qpos_full is not None:
                    view.sync_qpos(snap.qpos_full)
                else:
                    view.sync(snap.q, snap.gripper)
                panel.update(snap)
                if recorder is not None:
                    recorder.tick(snap)
            rate.sleep()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        if recorder is not None:
            recorder.finalize()
        loop.stop()
        loop.join(timeout=2.0)
        robot.disconnect()
        server.stop()


@dataclasses.dataclass
class ScriptedDemosArgs:
    """Auto-generate pick-and-lift demonstrations in the physics sim (no GUI)."""

    scenario: str
    """Scenario YAML with at least one graspable object."""
    dataset: str
    """LeRobotDataset repo_id to record into."""
    episodes: int = 20
    object_name: str | None = None
    """Object to pick (default: first object in the scenario)."""
    fps: int = 15
    root: str | None = None
    """Local dataset directory (default: HF cache)."""
    resume: bool = False
    seed: int = 0
    keep_failures: bool = False
    """Also save episodes where the object was not lifted."""
    render_width: int = 320
    render_height: int = 240
    max_attempts_factor: int = 3
    """Stop after episodes*factor attempts even if fewer successes."""


def _scripted_demos(args: ScriptedDemosArgs) -> None:
    from .control.loop import ControlLoop
    from .data.recorder import DatasetRecorder
    from .demo.scripted import PickParams, ScriptedPick
    from .kinematics import Kinematics
    from .robot.physics_sim import PhysicsBackend
    from .scenario import load_scenario

    scenario = load_scenario(args.scenario)
    if not scenario.objects:
        raise SystemExit("scenario has no objects to pick")
    object_name = args.object_name or scenario.objects[0].name

    config = AppConfig()
    backend = PhysicsBackend(
        scenario, config.joint_map, args.render_width, args.render_height, seed=args.seed
    )
    backend.connect()
    loop = ControlLoop(backend, Kinematics(), config)
    loop.start()
    recorder = DatasetRecorder(
        repo_id=args.dataset,
        fps=args.fps,
        cameras={n: (args.render_height, args.render_width) for n in backend.camera_names},
        joint_map=config.joint_map,
        root=args.root,
        task=scenario.task,
        resume=args.resume,
    )
    pick = ScriptedPick(
        loop, backend, Kinematics(), PickParams(object_name=object_name), sample_hz=args.fps
    )

    def sample():
        snap = loop.snapshot()
        if snap is None:
            return
        recorder.add_frame(
            q=snap.q,
            gripper=snap.gripper,
            q_cmd=snap.q_cmd if snap.q_cmd is not None else snap.q,
            gripper_cmd=snap.gripper_cmd if snap.gripper_cmd is not None else snap.gripper,
            images=backend.get_camera_frames(),
        )

    saved = attempts = 0
    try:
        while saved < args.episodes and attempts < args.episodes * args.max_attempts_factor:
            attempts += 1
            backend.reset(randomize=True)
            recorder.start_episode()
            ok = pick.run_episode(sample)
            keep = ok or args.keep_failures
            recorder.end_episode(save=keep)
            saved += keep
            print(f"episode {attempts}: {'SUCCESS' if ok else 'fail'} — saved {saved}/{args.episodes}")
    finally:
        root = recorder.finalize()
        loop.stop()
        loop.join(timeout=2.0)
        backend.disconnect()
    print(f"\ndataset written: {root} ({saved} episodes)")
    print(f"train with:  so101-tool train --dataset {args.dataset}"
          + (f" --dataset-root {root}" if args.root else ""))


@dataclasses.dataclass
class IkCheckArgs:
    """FK→IK round-trip accuracy check (no hardware needed)."""

    n: int = 100
    position_only: bool = False
    seed: int = 0


def _ik_check(args: IkCheckArgs) -> None:
    from .kinematics import IKError, Kinematics

    kin = Kinematics()
    rng = np.random.default_rng(args.seed)
    errors, failures = [], 0
    for _ in range(args.n):
        q = rng.uniform(ARM_LIMITS_LO * 0.7, ARM_LIMITS_HI * 0.7)
        target = kin.fk(q)
        if args.position_only:
            target.wxyz = None
        t0 = time.perf_counter()
        try:
            q_sol = kin.ik(target, np.zeros(5))
        except IKError:
            failures += 1
            continue
        dt = time.perf_counter() - t0
        errors.append((np.linalg.norm(kin.fk(q_sol).position - target.position), dt))
    if errors:
        pos = np.array([e[0] for e in errors]) * 1e3
        times = np.array([e[1] for e in errors]) * 1e3
        print(
            f"{len(errors)}/{args.n} converged | position error "
            f"median {np.median(pos):.2f} mm, max {pos.max():.2f} mm | "
            f"solve median {np.median(times):.1f} ms"
        )
    if failures:
        print(f"{failures} targets unreached (IKError)")


def main() -> None:
    from .config import ensure_headless_gl

    ensure_headless_gl()  # before anything imports mujoco
    from .training import TrainArgs, train_cli

    args = tyro.extras.subcommand_cli_from_dict(
        {
            "run": RunArgs,
            "scripted-demos": ScriptedDemosArgs,
            "train": TrainArgs,
            "ik-check": IkCheckArgs,
        }
    )
    if isinstance(args, RunArgs):
        _run(args)
    elif isinstance(args, ScriptedDemosArgs):
        _scripted_demos(args)
    elif isinstance(args, TrainArgs):
        train_cli(args)
    else:
        _ik_check(args)


if __name__ == "__main__":
    main()
