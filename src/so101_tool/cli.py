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

    if config.backend == "real":
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
    view = RobotView(server, kin_viz)

    nl_agent = None
    if not args.no_nl and os.environ.get("ANTHROPIC_API_KEY"):
        from .nl.agent import NLAgent

        nl_agent = NLAgent(loop, kin_viz, model=config.nl_model)
        print(f"natural-language control enabled ({config.nl_model})")
    panel = ControlPanel(server, loop, kin_viz, config, nl_agent=nl_agent)

    print()
    print(f"  ▶ 3D preview: http://localhost:{config.viser_port}")
    print("    Ctrl-C to exit.")
    print()

    rate = RateLimiter(frequency=config.render_hz, warn=False)
    try:
        while True:
            snap = loop.snapshot()
            if snap is not None:
                view.sync(snap.q, snap.gripper)
                panel.update(snap)
            rate.sleep()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        loop.stop()
        loop.join(timeout=2.0)
        robot.disconnect()
        server.stop()


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
    args = tyro.extras.subcommand_cli_from_dict({"run": RunArgs, "ik-check": IkCheckArgs})
    if isinstance(args, RunArgs):
        _run(args)
    else:
        _ik_check(args)


if __name__ == "__main__":
    main()
