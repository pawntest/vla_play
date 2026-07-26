"""Safety filter applied to every target before it reaches the robot.

Pure/stateless: each call depends only on its arguments. Order of operations:
joint-limit clamp -> velocity rate-limit -> gripper clamp -> workspace check.
"""

from __future__ import annotations

import numpy as np

from ..config import ARM_LIMITS_HI, ARM_LIMITS_LO, JointMap
from ..kinematics import Kinematics
from ..robot.base import RobotState


class SafetyFilter:
    def __init__(
        self,
        joint_map: JointMap,
        kinematics: Kinematics | None = None,
        floor_z: float = 0.005,
        vel_margin: float = 1.5,
    ):
        self.joint_map = joint_map
        self.kinematics = kinematics
        self.floor_z = floor_z
        self.vel_margin = vel_margin

    def filter(
        self, q_target: np.ndarray, gripper_target: float, state: RobotState, dt: float
    ) -> tuple[np.ndarray, float]:
        """Return (q_safe, gripper_safe) to send to the robot."""
        gripper_safe = float(np.clip(gripper_target, 0.0, 1.0))
        if dt <= 0:
            return state.q.copy(), gripper_safe

        q_safe = np.clip(np.asarray(q_target, dtype=float), ARM_LIMITS_LO, ARM_LIMITS_HI)
        max_step = self.joint_map.vmax_rad_s * dt * self.vel_margin
        q_safe = state.q + np.clip(q_safe - state.q, -max_step, max_step)

        if self.kinematics is not None:
            if self.kinematics.fk(q_safe).position[2] < self.floor_z:
                return state.q.copy(), gripper_safe  # hold: step would hit the floor
        return q_safe, gripper_safe
