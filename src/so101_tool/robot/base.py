"""Robot backend interface. All implementations speak canonical units:
arm q (5,) in radians (ARM_JOINTS order), gripper fraction in [0, 1].
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class RobotState:
    q: np.ndarray  # (5,) radians, ARM_JOINTS order
    gripper: float  # 0..1
    t: float  # time.monotonic() at read
    connected: bool = True
    # Full model qpos for scene-aware backends (arm first, then object free
    # joints); None for arm-only backends.
    qpos_full: np.ndarray | None = None


class RobotInterface(abc.ABC):
    """A position-controlled robot (real or simulated)."""

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def read_state(self) -> RobotState: ...

    @abc.abstractmethod
    def write_targets(self, q: np.ndarray, gripper: float) -> None:
        """Command joint position targets (radians) + gripper fraction."""

    @property
    @abc.abstractmethod
    def is_real(self) -> bool: ...
