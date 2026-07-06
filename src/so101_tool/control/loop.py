"""50 Hz control loop (frozen public API — workstream B implements).

Thread model:
  - ControlLoop (a Thread) is the ONLY thing that touches the robot backend.
  - Producers (viser GUI, NL agent, CLI) enqueue Command objects on .commands.
  - Consumers read the latest immutable LoopSnapshot via .snapshot().
  - estop() is latching: robot writes stop until reset_estop().

Per tick:
  state = robot.read_state()
  drain commands (SetMode/Stop/estop take effect immediately; a new motion
  primitive preempts the active one, finishing it with error="preempted")
  RULE:   q_t, g_t = active primitive step (MoveJ interp; MoveL/JogTool via
          Kinematics.ik_velocity servo)
  POLICY: q_t, g_t from the policy runner
  MIRROR/IDLE: no write
  robot.write_targets(*safety.filter(...)) unless estopped
  publish LoopSnapshot
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

from ..config import AppConfig
from ..kinematics import Kinematics
from ..robot.base import RobotInterface
from .commands import Command, Mode


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
    active_command: str | None  # repr of the running primitive, if any
    backend_is_real: bool


class ControlLoop(threading.Thread):
    def __init__(
        self,
        robot: RobotInterface,
        kinematics: Kinematics,
        config: AppConfig,
        policy_runner=None,  # optional object with .step(state) -> (q, gripper)
    ):
        super().__init__(daemon=True, name="so101-control-loop")
        self.commands: queue.Queue[Command] = queue.Queue()
        raise NotImplementedError  # workstream B

    def snapshot(self) -> LoopSnapshot: ...

    def estop(self) -> None:
        """Latching emergency stop (thread-safe, takes effect within one tick)."""

    def reset_estop(self) -> None: ...

    def stop(self) -> None:
        """Shut down the loop thread cleanly (join afterwards)."""
