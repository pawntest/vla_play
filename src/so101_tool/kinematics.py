"""FK/IK for the SO-101 arm. MuJoCo provides FK; mink provides differential IK.

Contract (frozen — workstream A implements, B/C code against it):
  - q is always (5,) radians in ARM_JOINTS order; gripper is a 0..1 fraction.
  - Poses are (position (3,) meters, wxyz (4,) unit quaternion) in the base frame,
    for the TCP site (config.TCP_SITE, between the gripper jaws).
  - Not thread-safe: each thread owns its own Kinematics instance.

Verified feasibility (this environment): mink FrameTask(position_cost=1.0,
orientation_cost=0.3) + PostureTask(1e-2), solver="daqp", converges to ~1 mm
position error on this 5-DOF arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SE3Pose:
    position: np.ndarray  # (3,)
    wxyz: np.ndarray  # (4,) unit quaternion, w first (MuJoCo/viser convention)


class IKError(RuntimeError):
    """Raised when IK cannot reach the requested pose within tolerance."""


class Kinematics:
    def __init__(self, mjcf_path: Path | None = None):
        """Loads the (vendored) MJCF; builds mjModel/mjData and a mink Configuration."""
        raise NotImplementedError  # workstream A

    def fk(self, q: np.ndarray, gripper: float = 0.0) -> SE3Pose:
        """TCP pose for joint config q."""
        raise NotImplementedError

    def set_qpos(self, q: np.ndarray, gripper: float) -> None:
        """Write qpos + mj_kinematics (visualization path). Exposes .model/.data."""
        raise NotImplementedError

    def ik(
        self,
        target: SE3Pose,
        q_seed: np.ndarray,
        *,
        position_only: bool = False,
        dt: float = 0.02,
        iters: int = 100,
        pos_tol: float = 5e-3,
    ) -> np.ndarray:
        """Full solve to a target pose. Returns q (5,). Raises IKError."""
        raise NotImplementedError

    def ik_velocity(
        self, target: SE3Pose, q_now: np.ndarray, dt: float, *, position_only: bool = False
    ) -> np.ndarray:
        """One differential-IK step toward target; returns next q (5,).
        Used by the 50 Hz servo for MoveL / tool jog."""
        raise NotImplementedError
