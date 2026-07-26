"""Simulated robot backend: a kinematic integrator emulating position-controlled
STS3215 servos. Each read_state() advances joints toward the last commanded
targets at the JointMap velocity limits, using wall-clock dt from an injectable
clock (tests pass a fake clock to step time deterministically).
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from ..config import ARM_LIMITS_HI, ARM_LIMITS_LO, JointMap
from .base import RobotInterface, RobotState

_GRIPPER_VMAX = 2.0  # fraction/s
_DT_CAP = 0.1  # s; survive stalls (debugger, GC) without a huge jump


class SimBackend(RobotInterface):
    def __init__(
        self,
        joint_map: JointMap | None = None,
        q0: np.ndarray | None = None,
        gripper0: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.joint_map = joint_map or JointMap()
        self._clock = clock
        self._lock = threading.Lock()
        self._q = np.zeros(5) if q0 is None else np.asarray(q0, dtype=float).copy()
        self._gripper = float(np.clip(gripper0, 0.0, 1.0))
        self._q_target = self._q.copy()
        self._gripper_target = self._gripper
        self._last_t: float | None = None
        self.connected = False

    def connect(self) -> None:
        with self._lock:
            self.connected = True
            self._last_t = self._clock()

    def disconnect(self) -> None:
        with self._lock:
            self.connected = False

    def read_state(self) -> RobotState:
        with self._lock:
            if not self.connected:
                raise RuntimeError("SimBackend not connected")
            now = self._clock()
            dt = min(now - (self._last_t if self._last_t is not None else now), _DT_CAP)
            self._last_t = now
            if dt > 0:
                vmax = self.joint_map.vmax_rad_s
                self._q += np.clip(self._q_target - self._q, -vmax * dt, vmax * dt)
                dg = self._gripper_target - self._gripper
                self._gripper += float(np.clip(dg, -_GRIPPER_VMAX * dt, _GRIPPER_VMAX * dt))
            return RobotState(q=self._q.copy(), gripper=self._gripper, t=now, connected=True)

    def write_targets(self, q: np.ndarray, gripper: float) -> None:
        with self._lock:
            if not self.connected:
                raise RuntimeError("SimBackend not connected")
            self._q_target = np.clip(np.asarray(q, dtype=float), ARM_LIMITS_LO, ARM_LIMITS_HI)
            self._gripper_target = float(np.clip(gripper, 0.0, 1.0))

    @property
    def is_real(self) -> bool:
        return False
