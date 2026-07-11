"""Scripted expert for automatic demonstration generation in the physics sim.

Runs a pick-and-lift sequence SYNCHRONOUSLY on a fake clock: each control tick
advances simulated time by exactly 1/control_hz regardless of how long camera
rendering takes, so datasets are deterministic and generation is immune to CPU
speed (and can run faster than real time). This is intentionally decoupled
from the interactive ControlLoop, which serves the live GUI on wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config import JointMap
from ..control.safety import SafetyFilter
from ..kinematics import IKError, Kinematics, SE3Pose
from ..robot.base import RobotState
from ..robot.physics_sim import PhysicsBackend

# cb(state, q_cmd, gripper_cmd) — called at the sampling rate in sim time
SampleCallback = Callable[[RobotState, np.ndarray, float], None]

_GRIPPER_VMAX = 2.0  # fraction/s


@dataclass
class PickParams:
    object_name: str = "cube"
    approach_height: float = 0.07  # m above the object for the pre-grasp pose
    grasp_offset: tuple = (0.0, 0.0, 0.008)  # TCP offset from object center at grasp
    open_fraction: float = 1.0
    close_fraction: float = 0.0
    close_duration: float = 1.2  # s: jaws stall on the object, so run a fixed time
    lift_height: float = 0.12
    success_lift: float = 0.05  # object must rise this much above its start z
    speed: float = 0.7
    phase_timeout: float = 8.0  # sim-seconds per motion phase


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class ScriptedPick:
    """Synchronous scripted pick. The backend MUST be built with clock=<FakeClock>
    (make_backend_and_pick() wires this up)."""

    def __init__(
        self,
        backend: PhysicsBackend,
        clock: FakeClock,
        kinematics: Kinematics | None = None,
        params: PickParams | None = None,
        joint_map: JointMap | None = None,
        control_hz: float = 50.0,
        sample_hz: float = 15.0,
    ):
        self._backend = backend
        self._clock = clock
        self._kin = kinematics or Kinematics()
        self.params = params or PickParams()
        self._jm = joint_map or backend.joint_map
        self._safety = SafetyFilter(self._jm)
        self._dt = 1.0 / control_hz
        self._ticks_per_sample = max(1, round(control_hz / sample_hz))

    # -- internals -----------------------------------------------------------

    def _servo_to(
        self,
        q_target: np.ndarray,
        g_target: float | None,
        q_cmd: np.ndarray,
        g_cmd: float,
        duration: float | None,
        sample_cb: SampleCallback | None,
        tick_offset: int,
    ) -> tuple[np.ndarray, float, bool, int]:
        """Velocity-limited ramp toward (q_target, g_target) in sim time.

        Runs until settled (or `duration` sim-seconds when given). Returns
        (q_cmd, g_cmd, settled, tick_offset).
        """
        p = self.params
        max_ticks = int((duration if duration is not None else p.phase_timeout) / self._dt)
        vmax = self._jm.vmax_rad_s
        state = None
        for _ in range(max_ticks):
            self._clock.t += self._dt
            step = vmax * p.speed * self._dt
            q_cmd = q_cmd + np.clip(q_target - q_cmd, -step, step)
            if g_target is not None:
                dg = np.clip(g_target - g_cmd, -_GRIPPER_VMAX * self._dt, _GRIPPER_VMAX * self._dt)
                g_cmd = float(g_cmd + dg)
            prev = self._backend.read_state()
            q_safe, g_safe = self._safety.filter(q_cmd, g_cmd, prev, self._dt)
            self._backend.write_targets(q_safe, g_safe)
            state = self._backend.read_state()
            tick_offset += 1
            if sample_cb is not None and tick_offset % self._ticks_per_sample == 0:
                sample_cb(state, q_cmd.copy(), g_cmd)
            if duration is None and np.max(np.abs(state.q - q_target)) < 0.02 and (
                g_target is None or abs(state.gripper - g_target) < 0.05
            ):
                return q_cmd, g_cmd, True, tick_offset
        return q_cmd, g_cmd, duration is not None, tick_offset

    def _ik(self, position: np.ndarray, seed: np.ndarray) -> np.ndarray:
        return self._kin.ik(SE3Pose(position=position, wxyz=None), seed)

    # -- public API --------------------------------------------------------------

    def run_episode(self, sample_cb: SampleCallback | None = None) -> bool:
        """One pick-and-lift attempt. Returns True on success (object lifted)."""
        p = self.params
        obj_pos, _ = self._backend.object_pose(p.object_name)
        start_z = obj_pos[2]
        grasp = obj_pos + np.asarray(p.grasp_offset)
        above = grasp + [0.0, 0.0, p.approach_height]

        state = self._backend.read_state()
        q_cmd, g_cmd = state.q.copy(), state.gripper
        try:
            q_above = self._ik(above, q_cmd)
            q_grasp = self._ik(grasp, q_above)
            q_lift = self._ik(grasp + [0.0, 0.0, p.lift_height], q_grasp)
        except IKError:
            return False

        ticks = 0
        q_cmd, g_cmd, ok, ticks = self._servo_to(
            q_above, p.open_fraction, q_cmd, g_cmd, None, sample_cb, ticks
        )
        if not ok:
            return False
        q_cmd, g_cmd, ok, ticks = self._servo_to(q_grasp, None, q_cmd, g_cmd, None, sample_cb, ticks)
        if not ok:
            return False
        # Close on the object for a fixed time: the jaws stall against it,
        # which IS the grasp — a settle check would never pass.
        q_cmd, g_cmd, _, ticks = self._servo_to(
            q_cmd.copy(), p.close_fraction, q_cmd, g_cmd, p.close_duration, sample_cb, ticks
        )
        q_cmd, g_cmd, ok, ticks = self._servo_to(q_lift, None, q_cmd, g_cmd, None, sample_cb, ticks)
        if not ok:
            return False
        end_z = self._backend.object_pose(p.object_name)[0][2]
        return bool(end_z - start_z > p.success_lift)


def make_backend_and_pick(
    scenario,
    joint_map: JointMap | None = None,
    render_width: int = 320,
    render_height: int = 240,
    seed: int = 0,
    params: PickParams | None = None,
    control_hz: float = 50.0,
    sample_hz: float = 15.0,
) -> tuple[PhysicsBackend, ScriptedPick]:
    """Build a fake-clocked PhysicsBackend + ScriptedPick pair for data generation."""
    clock = FakeClock()
    backend = PhysicsBackend(
        scenario, joint_map, render_width, render_height, seed=seed, clock=clock
    )
    backend.connect()
    pick = ScriptedPick(
        backend, clock, params=params, joint_map=joint_map,
        control_hz=control_hz, sample_hz=sample_hz,
    )
    return backend, pick
