"""Commands enqueued into the control loop (from GUI, CLI, or the NL agent).

Thread contract: producers construct a command, put it on ControlLoop.commands,
and may block on cmd.wait(). The control loop is the only consumer; it calls
cmd.finish() exactly once when the command completes, fails, or is preempted.
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass, field

import numpy as np


class Mode(enum.Enum):
    IDLE = "idle"  # no writes to the robot
    MIRROR = "mirror"  # read real joints -> 3D preview only (hand-pose the robot)
    RULE = "rule"  # execute motion primitives (also used by the NL agent)
    POLICY = "policy"  # lerobot policy inference drives the robot


@dataclass
class Command:
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    error: str | None = None  # set before finish() on failure/preemption

    def finish(self, error: str | None = None) -> None:
        self.error = error
        self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until finished. Returns False on timeout."""
        return self._done.wait(timeout)

    @property
    def ok(self) -> bool:
        return self._done.is_set() and self.error is None


@dataclass
class MoveJ(Command):
    """Joint-space move: interpolate to q (radians) at speed x vmax."""

    q: np.ndarray = None  # (5,)
    gripper: float | None = None
    speed: float = 1.0  # fraction of JointMap.vmax_rad_s


@dataclass
class MoveL(Command):
    """Cartesian move of the TCP (gripperframe site) via differential IK servo."""

    position: np.ndarray = None  # (3,) meters, base frame
    wxyz: np.ndarray | None = None  # (4,) target orientation; None = position-only
    speed: float = 1.0


@dataclass
class JogTool(Command):
    """Relative TCP step. Frame: 'tool' (TCP axes) or 'base'."""

    dpos: np.ndarray = None  # (3,) meters
    frame: str = "tool"
    speed: float = 1.0


@dataclass
class SetGripper(Command):
    fraction: float = 1.0  # 0 closed .. 1 open


@dataclass
class Home(Command):
    """MoveJ to the home pose (all zeros, gripper unchanged)."""

    speed: float = 0.5


@dataclass
class Stop(Command):
    """Abort the active primitive and hold position (does NOT latch e-stop)."""


@dataclass
class SetMode(Command):
    mode: Mode = Mode.IDLE
