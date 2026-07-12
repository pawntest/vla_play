"""50 Hz control loop.

Thread model:
  - ControlLoop (a Thread) is the ONLY thing that touches the robot backend.
  - Producers (viser GUI, NL agent, CLI) enqueue Command objects on .commands.
  - Consumers read the latest immutable LoopSnapshot via .snapshot().
  - estop() is latching: no robot writes until reset_estop().
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
from loop_rate_limiters import RateLimiter

from ..config import AppConfig
from ..kinematics import Kinematics, SE3Pose
from ..robot.base import RobotInterface, RobotState
from .commands import Command, Home, JogTool, Mode, MoveJ, MoveL, SetGripper, SetMode, Stop
from .safety import SafetyFilter

_JOINT_TOL = 0.02  # rad
_GRIPPER_TOL = 0.02
_GRIPPER_VMAX = 2.0  # fraction/s
_MOVEL_POS_TOL = 3e-3  # m


@dataclass(frozen=True)
class LoopSnapshot:
    t: float
    mode: Mode
    q: np.ndarray  # (5,) rad — current measured joints
    gripper: float  # 0..1
    tcp_position: np.ndarray  # (3,) m
    tcp_wxyz: np.ndarray  # (4,)
    estop: bool
    error: str | None  # last error message, if any
    active_command: str | None  # short description of the running primitive
    backend_is_real: bool
    # extras for scene view / data recording
    qpos_full: np.ndarray | None = None  # full model qpos (scene backends)
    q_cmd: np.ndarray | None = None  # last commanded arm targets (rad)
    gripper_cmd: float | None = None  # last commanded gripper fraction


class _Executor:
    """Steps one motion primitive. Returns (q_cmd, gripper_cmd, done)."""

    def __init__(self, cmd: Command, deadline: float):
        self.cmd = cmd
        self.deadline = deadline

    def step(self, state: RobotState, dt: float):  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> str:
        return type(self.cmd).__name__


class _MoveJExec(_Executor):
    def __init__(self, cmd: MoveJ, state: RobotState, q_cmd: np.ndarray, vmax: np.ndarray):
        self.q_target = np.asarray(cmd.q, dtype=float)
        self.g_target = cmd.gripper
        self.speed = float(np.clip(cmd.speed, 0.05, 1.0))
        self.vmax = vmax
        self.q_cmd = q_cmd.copy()
        self.g_cmd = state.gripper
        nominal = float(np.max(np.abs(self.q_target - state.q) / (vmax * self.speed)))
        super().__init__(cmd, time.monotonic() + 3.0 * nominal + 1.0)

    def step(self, state: RobotState, dt: float):
        step = self.vmax * self.speed * dt
        self.q_cmd += np.clip(self.q_target - self.q_cmd, -step, step)
        if self.g_target is not None:
            dg = np.clip(self.g_target - self.g_cmd, -_GRIPPER_VMAX * dt, _GRIPPER_VMAX * dt)
            self.g_cmd = float(self.g_cmd + dg)
        done = np.max(np.abs(state.q - self.q_target)) < _JOINT_TOL and (
            self.g_target is None or abs(state.gripper - self.g_target) < _GRIPPER_TOL
        )
        return self.q_cmd, self.g_cmd, done


class _MoveLExec(_Executor):
    """Differential-IK servo toward a TCP pose (also used for tool jogs)."""

    def __init__(self, target: SE3Pose, cmd: Command, state: RobotState, kin: Kinematics,
                 speed: float, timeout: float = 10.0):
        self.target = target
        self.kin = kin
        self.speed = float(np.clip(speed, 0.05, 1.0))
        self.q_cmd = state.q.copy()
        self.g_cmd = state.gripper
        super().__init__(cmd, time.monotonic() + timeout)

    def step(self, state: RobotState, dt: float):
        q_next = self.kin.ik_velocity(self.target, self.q_cmd, dt * self.speed)
        tick_motion = float(np.max(np.abs(q_next - self.q_cmd)))
        self.q_cmd = q_next
        # Completion is judged on the MEASURED state: the safety filter
        # rate-limits what actually reaches the robot, so q_cmd can lead it.
        pos_err = float(np.linalg.norm(self.kin.fk(state.q).position - self.target.position))
        settled = float(np.max(np.abs(state.q - self.q_cmd))) < _JOINT_TOL
        done = pos_err < _MOVEL_POS_TOL and tick_motion < 1e-4 and settled
        return self.q_cmd, self.g_cmd, done


class _SetGripperExec(_Executor):
    def __init__(self, cmd: SetGripper, state: RobotState, q_cmd: np.ndarray):
        self.target = float(np.clip(cmd.fraction, 0.0, 1.0))
        self.q_cmd = q_cmd.copy()
        self.g_cmd = state.gripper
        nominal = abs(self.target - state.gripper) / _GRIPPER_VMAX
        super().__init__(cmd, time.monotonic() + 3.0 * nominal + 1.0)

    def step(self, state: RobotState, dt: float):
        dg = np.clip(self.target - self.g_cmd, -_GRIPPER_VMAX * dt, _GRIPPER_VMAX * dt)
        self.g_cmd = float(self.g_cmd + dg)
        return self.q_cmd, self.g_cmd, abs(state.gripper - self.target) < _GRIPPER_TOL


def _jog_target(kin: Kinematics, state: RobotState, cmd: JogTool) -> SE3Pose:
    pose = kin.fk(state.q)
    dpos = np.asarray(cmd.dpos, dtype=float)
    if cmd.frame == "tool":
        mat = np.empty(9)
        import mujoco

        mujoco.mju_quat2Mat(mat, pose.wxyz)
        dpos = mat.reshape(3, 3) @ dpos
    return SE3Pose(position=pose.position + dpos, wxyz=None)


class ControlLoop(threading.Thread):
    def __init__(
        self,
        robot: RobotInterface,
        kinematics: Kinematics,
        config: AppConfig,
        policy_runner=None,  # optional object with .reset() and .step(state) -> (q, g) | None
    ):
        super().__init__(daemon=True, name="so101-control-loop")
        self.commands: queue.Queue[Command] = queue.Queue()
        self._robot = robot
        self._kin = kinematics
        self._config = config
        self._policy = policy_runner
        self._safety = SafetyFilter(config.joint_map)
        self._mode = Mode.IDLE
        self._active: _Executor | None = None
        self._estop = threading.Event()
        self._run = threading.Event()
        self._run.set()
        self._error: str | None = None
        self._snap: LoopSnapshot | None = None

    # -- thread-safe public API ---------------------------------------------

    def snapshot(self) -> LoopSnapshot | None:
        return self._snap

    def set_policy_runner(self, runner) -> None:
        """Swap the policy runner at runtime (GIL-atomic reference assignment)."""
        self._policy = runner

    def estop(self) -> None:
        """Latching emergency stop (takes effect within one tick)."""
        self._estop.set()

    def reset_estop(self) -> None:
        self._estop.clear()

    def stop(self) -> None:
        """Shut down the loop thread (join afterwards). Does not disconnect the robot."""
        self._run.clear()

    # -- loop internals -------------------------------------------------------

    def _abort_active(self, reason: str) -> None:
        if self._active is not None:
            self._active.cmd.finish(error=reason)
            self._active = None

    def _start_motion(self, cmd: Command, state: RobotState, q_cmd: np.ndarray) -> None:
        if self._mode is not Mode.RULE:
            cmd.finish(error=f"not in RULE mode (mode={self._mode.value})")
            return
        self._abort_active("preempted")
        vmax = self._config.joint_map.vmax_rad_s
        try:
            if isinstance(cmd, Home):
                mj = MoveJ(q=np.zeros(5), speed=cmd.speed)
                self._active = _MoveJExec(mj, state, q_cmd, vmax)
                self._active.cmd = cmd  # finish the original Home command
            elif isinstance(cmd, MoveJ):
                self._active = _MoveJExec(cmd, state, q_cmd, vmax)
            elif isinstance(cmd, MoveL):
                target = SE3Pose(np.asarray(cmd.position, dtype=float),
                                 None if cmd.wxyz is None else np.asarray(cmd.wxyz, float))
                self._active = _MoveLExec(target, cmd, state, self._kin, cmd.speed)
            elif isinstance(cmd, JogTool):
                self._active = _MoveLExec(_jog_target(self._kin, state, cmd), cmd, state,
                                          self._kin, cmd.speed)
            elif isinstance(cmd, SetGripper):
                self._active = _SetGripperExec(cmd, state, q_cmd)
        except Exception as exc:  # bad target etc. — never kill the loop
            cmd.finish(error=str(exc))
            self._active = None
            self._error = str(exc)

    def _handle_command(self, cmd: Command, state: RobotState, q_cmd: np.ndarray) -> None:
        if isinstance(cmd, SetMode):
            if cmd.mode is Mode.POLICY and self._policy is None:
                cmd.finish(error="no policy loaded (--policy-path)")
                return
            if self._mode is Mode.RULE and cmd.mode is not Mode.RULE:
                self._abort_active("mode changed")
            self._mode = cmd.mode
            self._error = None
            if cmd.mode is Mode.POLICY:
                try:
                    self._policy.reset()
                except Exception as exc:
                    self._mode = Mode.IDLE
                    cmd.finish(error=f"policy reset failed: {exc}")
                    self._error = str(exc)
                    return
            cmd.finish()
        elif isinstance(cmd, Stop):
            self._abort_active("stopped")
            cmd.finish()
        elif self._estop.is_set():
            cmd.finish(error="estop")
        else:
            self._start_motion(cmd, state, q_cmd)

    def run(self) -> None:
        rate = RateLimiter(frequency=self._config.control_hz, warn=False)
        q_cmd = None
        g_cmd: float | None = None
        last_t = time.monotonic()
        while self._run.is_set():
            try:
                state = self._robot.read_state()
            except Exception as exc:
                self._error = f"robot read failed: {exc}"
                rate.sleep()
                continue
            now = time.monotonic()
            dt = min(max(now - last_t, 1e-4), 0.1)
            last_t = now
            if q_cmd is None:
                q_cmd = state.q.copy()

            if self._estop.is_set():
                self._abort_active("estop")

            while True:
                try:
                    cmd = self.commands.get_nowait()
                except queue.Empty:
                    break
                self._handle_command(cmd, state, q_cmd)

            q_t = g_t = None
            if self._mode is Mode.RULE and self._active is not None:
                if time.monotonic() > self._active.deadline:
                    self._abort_active("timeout")
                else:
                    try:
                        q_t, g_t, done = self._active.step(state, dt)
                        q_cmd = q_t.copy()
                        g_cmd = g_t
                        if done:
                            self._active.cmd.finish()
                            self._active = None
                    except Exception as exc:
                        self._abort_active(f"primitive failed: {exc}")
                        self._error = str(exc)
            elif self._mode is Mode.POLICY:
                try:
                    result = self._policy.step(state)
                    if result is not None:
                        q_t, g_t = result
                        q_cmd = np.asarray(q_t, dtype=float).copy()
                        g_cmd = g_t
                except Exception as exc:
                    self._error = f"policy failed: {exc}"
                    self._mode = Mode.IDLE

            if q_t is not None and not self._estop.is_set():
                q_safe, g_safe = self._safety.filter(q_t, g_t, state, dt)
                try:
                    self._robot.write_targets(q_safe, g_safe)
                except Exception as exc:
                    self._error = f"robot write failed: {exc}"
            self._publish(state, q_cmd, g_cmd)
            rate.sleep()

        # shutdown: never leave a producer blocked on wait()
        self._abort_active("shutdown")
        while True:
            try:
                self.commands.get_nowait().finish(error="shutdown")
            except queue.Empty:
                break

    def _publish(self, state: RobotState, q_cmd=None, g_cmd=None) -> None:
        tcp = self._kin.fk(state.q, state.gripper)
        self._snap = LoopSnapshot(
            t=state.t,
            mode=self._mode,
            q=state.q,
            gripper=state.gripper,
            tcp_position=tcp.position,
            tcp_wxyz=tcp.wxyz,
            estop=self._estop.is_set(),
            error=self._error,
            active_command=self._active.describe() if self._active else None,
            backend_is_real=self._robot.is_real,
            qpos_full=state.qpos_full,
            q_cmd=None if q_cmd is None else q_cmd.copy(),
            gripper_cmd=g_cmd,
        )
