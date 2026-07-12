"""Public Python API for so101_tool.

Everything the CLI and the browser UI can do, as a library:

    from so101_tool import api

    # --- simulate & move (no hardware, no GUI) ------------------------------
    with api.Session(scenario="examples/pick_cube.yaml") as sess:
        sess.home()
        sess.move_to([0.25, 0.0, 0.10])          # cartesian, position-only IK
        sess.gripper(0.0)                        # close
        print(sess.state().tcp_position)

        # --- collect demonstrations & train ---------------------------------
        result = sess.generate_demos("my/pick", episodes=30, root="data/pick")
        api.train(dataset="data/pick", policy="act", steps=20000)

        # --- run a trained policy -------------------------------------------
        sess.run_policy("outputs/train/checkpoints/last/pretrained_model")

    # --- same code drives the real robot ------------------------------------
    with api.Session(backend="real", port="/dev/ttyACM0") as sess:
        sess.move_joints([0.2, -0.3, 0.4, 0.0, 0.0])

    # --- open the browser UI on top of a session ----------------------------
    sess = api.Session(scenario="examples/pick_cube.yaml")
    sess.open_ui(port=8080)                      # blocks; Ctrl-C to exit

Motion methods are blocking by default (wait=False returns the Command).
All motion goes through the same 50 Hz control loop + safety filter as the GUI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AppConfig, JointMap  # noqa: F401  (re-export)
from .control.commands import (
    Command,
    Home,
    JogTool,
    Mode,
    MoveJ,
    MoveL,
    SetGripper,
    SetMode,
    Stop,
)
from .control.loop import ControlLoop, LoopSnapshot
from .kinematics import IKError, Kinematics, SE3Pose  # noqa: F401  (re-export)
from .scenario import (  # noqa: F401  (re-export)
    CameraSpec,
    EnvironmentSpec,
    FloorSpec,
    ObjectSpec,
    Scenario,
    TableSpec,
    build_model,
    load_scenario,
    save_scenario,
)
from .training import TrainArgs, emit_colab, push_dataset, train_local  # noqa: F401

_DEFAULT_TIMEOUT = 30.0


class Session:
    """A running robot session: one backend + one control loop, no GUI required.

    backend: "sim" (kinematic), "real" (SO-101 via lerobot), or pass `scenario`
    (path or Scenario) for the contact-physics simulation with cameras.
    """

    def __init__(
        self,
        backend: str = "sim",
        scenario: str | Path | Scenario | None = None,
        config: AppConfig | None = None,
        connect: bool = True,
        **config_overrides,
    ):
        self.config = config or AppConfig(backend=backend, **config_overrides)
        self.scenario: Scenario | None = None
        if scenario is not None:
            self.scenario = (
                scenario if isinstance(scenario, Scenario) else load_scenario(scenario)
            )
            from .robot.physics_sim import PhysicsBackend

            self.robot = PhysicsBackend(
                self.scenario, self.config.joint_map,
                self.config.render_width, self.config.render_height,
            )
        elif self.config.backend == "real":
            from .robot.lerobot_backend import LeRobotBackend

            self.robot = LeRobotBackend(self.config)
        else:
            from .robot.sim import SimBackend

            self.robot = SimBackend(self.config.joint_map)
        self.kinematics = Kinematics()
        self.loop = ControlLoop(self.robot, Kinematics(), self.config)
        self._policy_runner = None
        if connect:
            self.connect()

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> "Session":
        self.robot.connect()
        if not self.loop.is_alive():
            self.loop.start()
        return self

    def close(self) -> None:
        self.loop.stop()
        self.loop.join(timeout=2.0)
        self.robot.disconnect()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- state ------------------------------------------------------------------

    def state(self, wait: float = 2.0) -> LoopSnapshot:
        """Latest loop snapshot (q, gripper, TCP pose, mode, estop...)."""
        import time

        deadline = time.monotonic() + wait
        while (snap := self.loop.snapshot()) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if snap is None:
            raise RuntimeError("control loop did not publish a snapshot in time")
        return snap

    def camera_frames(self) -> dict[str, np.ndarray]:
        if not hasattr(self.robot, "get_camera_frames"):
            raise RuntimeError("this backend has no cameras (use scenario= or real)")
        return self.robot.get_camera_frames()

    # -- motion -------------------------------------------------------------------

    def _submit(self, cmd: Command, wait: bool, timeout: float) -> Command:
        self.loop.commands.put(cmd)
        if wait:
            if not cmd.wait(timeout):
                raise TimeoutError(f"{type(cmd).__name__} did not finish in {timeout}s")
            if not cmd.ok:
                raise RuntimeError(f"{type(cmd).__name__} failed: {cmd.error}")
        return cmd

    def set_mode(self, mode: str | Mode, wait: bool = True) -> Command:
        mode = Mode(mode) if isinstance(mode, str) else mode
        return self._submit(SetMode(mode=mode), wait, 10.0)

    def _ensure_rule(self) -> None:
        snap = self.loop.snapshot()
        if snap is None or snap.mode is not Mode.RULE:
            self.set_mode(Mode.RULE)

    def move_joints(self, q, gripper: float | None = None, speed: float = 0.5,
                    wait: bool = True, timeout: float = _DEFAULT_TIMEOUT) -> Command:
        """Joint-space move (radians, ARM_JOINTS order)."""
        self._ensure_rule()
        return self._submit(
            MoveJ(q=np.asarray(q, dtype=float), gripper=gripper, speed=speed), wait, timeout
        )

    def move_to(self, position, wxyz=None, speed: float = 0.5, wait: bool = True,
                timeout: float = _DEFAULT_TIMEOUT) -> Command:
        """Cartesian TCP move (meters, base frame). wxyz=None -> position-only."""
        self._ensure_rule()
        return self._submit(
            MoveL(position=np.asarray(position, dtype=float),
                  wxyz=None if wxyz is None else np.asarray(wxyz, dtype=float),
                  speed=speed),
            wait, timeout,
        )

    def jog(self, dpos, frame: str = "tool", speed: float = 0.5, wait: bool = True,
            timeout: float = _DEFAULT_TIMEOUT) -> Command:
        """Relative TCP step (meters) in 'tool' or 'base' frame."""
        self._ensure_rule()
        return self._submit(
            JogTool(dpos=np.asarray(dpos, dtype=float), frame=frame, speed=speed),
            wait, timeout,
        )

    def gripper(self, fraction: float, wait: bool = True,
                timeout: float = _DEFAULT_TIMEOUT) -> Command:
        """0 = closed .. 1 = open."""
        self._ensure_rule()
        return self._submit(SetGripper(fraction=fraction), wait, timeout)

    def home(self, speed: float = 0.5, wait: bool = True,
             timeout: float = _DEFAULT_TIMEOUT) -> Command:
        self._ensure_rule()
        return self._submit(Home(speed=speed), wait, timeout)

    def stop_motion(self) -> None:
        self._submit(Stop(), wait=True, timeout=5.0)

    def estop(self) -> None:
        self.loop.estop()

    def reset_estop(self) -> None:
        self.loop.reset_estop()

    # -- scene ------------------------------------------------------------------------

    def reset_scene(self, randomize: bool = True) -> None:
        if hasattr(self.robot, "reset"):
            self.robot.reset(randomize=randomize)

    def object_pose(self, name: str):
        return self.robot.object_pose(name)

    def set_object_pose(self, name: str, pos, wxyz=None) -> None:
        self.robot.set_object_pose(name, pos, wxyz)

    # -- data & policy ------------------------------------------------------------------

    def generate_demos(self, dataset: str, episodes: int, **kwargs):
        """Scripted pick demos into a LeRobotDataset (see demo.generate)."""
        from .demo.generate import generate_demo_dataset

        if self.scenario is None:
            raise RuntimeError("generate_demos needs a scenario session")
        return generate_demo_dataset(
            self.scenario, dataset, episodes,
            joint_map=self.config.joint_map,
            render_width=self.config.render_width,
            render_height=self.config.render_height,
            **kwargs,
        )

    def run_policy(self, checkpoint: str, task: str | None = None) -> None:
        """Load a lerobot policy and switch the loop to POLICY mode (blocking load)."""
        import dataclasses

        from .policy.runner import PolicyRunner

        cfg = dataclasses.replace(self.config, policy_path=checkpoint,
                                  policy_task=task or (self.scenario.task if self.scenario else None))
        runner = PolicyRunner(cfg, self.robot)
        runner.reset()
        self._policy_runner = runner
        self.loop.set_policy_runner(runner)
        self.set_mode(Mode.POLICY)

    def stop_policy(self) -> None:
        self.set_mode(Mode.IDLE)

    # -- GUI ---------------------------------------------------------------------------

    def open_ui(self, port: int = 8080) -> None:
        """Attach the browser UI to this session and serve it (blocking)."""
        import viser

        from .app import App, run_app

        self.config.viser_port = port
        server = viser.ViserServer(port=port)
        app = App.from_session(server, self)
        print(f"\n  ▶ 3D preview: http://localhost:{port}\n")
        run_app(app, self.config.render_hz)


def train(dataset: str, policy: str = "act", **kwargs) -> int:
    """Train a lerobot policy locally (wraps lerobot-train). Returns exit code."""
    return train_local(TrainArgs(dataset=dataset, policy=policy, **kwargs))
