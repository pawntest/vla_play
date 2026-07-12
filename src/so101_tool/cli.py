"""Command-line entry point: `so101-tool run` / `so101-tool ik-check`."""

from __future__ import annotations

import dataclasses
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

    from .app import App, run_app

    config = AppConfig(
        backend=args.backend,
        port=args.port,
        robot_id=args.robot_id,
        viser_port=args.viser_port,
        control_hz=args.control_hz,
        render_hz=args.render_hz,
        policy_path=args.policy_path,
        policy_task=args.policy_task,
        render_width=args.render_width,
        render_height=args.render_height,
        no_nl=args.no_nl,
    )
    if args.nl_model:
        config.nl_model = args.nl_model

    server = viser.ViserServer(port=config.viser_port)
    app = App(
        server,
        config,
        scenario_path=args.scenario,
        record_repo=args.record,
        record_root=args.record_root,
        record_fps=args.record_fps,
    )
    print()
    print(f"  ▶ 3D preview: http://localhost:{config.viser_port}")
    print("    Ctrl-C to exit.")
    print()
    run_app(app, config.render_hz)


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
    from .demo.generate import generate_demo_dataset
    from .scenario import load_scenario

    def progress(attempt, saved, ok):
        print(f"episode {attempt}: {'SUCCESS' if ok else 'fail'} — saved {saved}/{args.episodes}")

    result = generate_demo_dataset(
        load_scenario(args.scenario), args.dataset, args.episodes,
        root=args.root, object_name=args.object_name, fps=args.fps, seed=args.seed,
        keep_failures=args.keep_failures, resume=args.resume,
        render_width=args.render_width, render_height=args.render_height,
        joint_map=AppConfig().joint_map, max_attempts_factor=args.max_attempts_factor,
        on_progress=progress,
    )
    print(f"\ndataset written: {result.dataset_root} ({result.saved} episodes)")
    print(f"train with:  so101-tool train --dataset {args.dataset}"
          + (f" --dataset-root {result.dataset_root}" if args.root else ""))


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
