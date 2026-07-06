"""FK/IK for the SO-101 arm. MuJoCo provides FK; mink provides differential IK.

Contract:
  - q is always (5,) radians in ARM_JOINTS order; gripper is a 0..1 fraction.
  - Poses are (position (3,) meters, wxyz (4,) unit quaternion) in the base frame,
    for the TCP site (config.TCP_SITE, between the gripper jaws).
  - Not thread-safe: each thread owns its own Kinematics instance.

The 5-DOF arm cannot realize arbitrary 6-DOF poses, so the frame task uses a
soft orientation cost (0.3) that trades off orientation gracefully, plus a
posture task for redundancy/regularization. position_only=True drops the
orientation objective entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np

from .config import TCP_SITE, gripper_fraction_to_rad, so101_mjcf_path


@dataclass
class SE3Pose:
    position: np.ndarray  # (3,)
    wxyz: np.ndarray  # (4,) unit quaternion, w first (MuJoCo/viser convention)


class IKError(RuntimeError):
    """Raised when IK cannot reach the requested pose within tolerance."""


_IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


class Kinematics:
    def __init__(self, mjcf_path: Path | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path or so101_mjcf_path()))
        self.data = mujoco.MjData(self.model)
        self._tcp_sid = self.model.site(TCP_SITE).id
        self._conf = mink.Configuration(self.model)
        self._task = mink.FrameTask(
            frame_name=TCP_SITE,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.3,
            lm_damping=1.0,
        )
        self._task_pos = mink.FrameTask(
            frame_name=TCP_SITE,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        self._posture = mink.PostureTask(self.model, cost=1e-2)
        self._posture.set_target(np.zeros(self.model.nq))
        self._limits = [mink.ConfigurationLimit(self.model)]

    # -- helpers -----------------------------------------------------------

    def _full_q(self, q: np.ndarray, gripper: float) -> np.ndarray:
        return np.concatenate([np.asarray(q, dtype=float), [gripper_fraction_to_rad(gripper)]])

    def _tcp_pose_from(self, data: mujoco.MjData) -> SE3Pose:
        pos = data.site_xpos[self._tcp_sid].copy()
        wxyz = np.empty(4)
        mujoco.mju_mat2Quat(wxyz, data.site_xmat[self._tcp_sid].reshape(-1))
        return SE3Pose(position=pos, wxyz=wxyz)

    def _target_se3(self, target: SE3Pose) -> mink.SE3:
        wxyz = target.wxyz if target.wxyz is not None else _IDENTITY_WXYZ
        return mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(wxyz, dtype=float)), np.asarray(target.position, dtype=float)
        )

    # -- public API ---------------------------------------------------------

    def set_qpos(self, q: np.ndarray, gripper: float) -> None:
        """Write qpos + mj_kinematics (visualization path; exposes .model/.data)."""
        self.data.qpos[:] = self._full_q(q, gripper)
        mujoco.mj_kinematics(self.model, self.data)

    def fk(self, q: np.ndarray, gripper: float = 0.0) -> SE3Pose:
        self.set_qpos(q, gripper)
        return self._tcp_pose_from(self.data)

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
        if target.wxyz is None:
            position_only = True
        task = self._task_pos if position_only else self._task
        task.set_target(self._target_se3(target))
        self._conf.update(self._full_q(q_seed, 0.0))
        tasks = [task, self._posture]
        for _ in range(iters):
            vel = mink.solve_ik(self._conf, tasks, dt, solver="daqp", limits=self._limits)
            vel[-1] = 0.0  # never let IK drive the gripper jaw
            self._conf.integrate_inplace(vel, dt)
            err = task.compute_error(self._conf)
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(vel[:-1]) < 1e-3:
                break
        q = self._conf.q[:5].copy()
        reached = self.fk(q).position
        pos_err = float(np.linalg.norm(reached - target.position))
        if pos_err > pos_tol:
            raise IKError(
                f"IK did not converge: position error {pos_err * 1e3:.1f} mm "
                f"(target {np.round(target.position, 3)})"
            )
        return q

    def ik_velocity(
        self, target: SE3Pose, q_now: np.ndarray, dt: float, *, position_only: bool = False
    ) -> np.ndarray:
        """One differential-IK step toward target; returns next q (5,).
        Used by the 50 Hz servo for MoveL / tool jog."""
        if target.wxyz is None:
            position_only = True
        task = self._task_pos if position_only else self._task
        task.set_target(self._target_se3(target))
        self._conf.update(self._full_q(q_now, 0.0))
        vel = mink.solve_ik(self._conf, [task, self._posture], dt, solver="daqp", limits=self._limits)
        vel[-1] = 0.0
        self._conf.integrate_inplace(vel, dt)
        return self._conf.q[:5].copy()
