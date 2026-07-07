"""Real SO-101 backend wrapping lerobot's SO101Follower.

lerobot (and torch) are heavy and require Python >= 3.12, so they are imported
only inside connect(). All conversions between lerobot's wire format
({"<motor>.pos": degrees, "gripper.pos": 0..100}) and the canonical internal
units (radians / 0..1 fraction) happen here and nowhere else.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import ARM_JOINTS, AppConfig
from .base import RobotInterface, RobotState

_INSTALL_HINT = (
    "lerobot is not installed. Install the real-robot extra with:\n"
    '    pip install "so101-tool[real]"\n'
    "(requires Python >= 3.12; see docs/hardware.md)"
)


class LeRobotBackend(RobotInterface):
    def __init__(self, config: AppConfig):
        self._config = config
        self._robot = None
        self.last_observation: dict | None = None  # read by the policy runner

    @property
    def lerobot_robot(self):
        """Underlying SO101Follower (for the policy runner). None until connect()."""
        return self._robot

    def connect(self) -> None:
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc

        cfg = SO101FollowerConfig(
            port=self._config.port,
            id=self._config.robot_id,
            use_degrees=True,
            cameras=self._config.cameras,
            disable_torque_on_disconnect=True,
        )
        self._robot = SO101Follower(cfg)
        self._robot.connect()

        obs = self._robot.get_observation()
        arm_deg = np.array([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=float)
        # With use_degrees=True a resting arm reads tens of degrees; if every
        # value is tiny the bus is probably reporting radians or normalized
        # units (lerobot version / config drift) and blind conversion would
        # command a violent move.
        if np.all(np.abs(arm_deg) < 3.2) and np.any(np.abs(arm_deg) > 0.01):
            self._robot.disconnect()
            self._robot = None
            raise RuntimeError(
                f"Suspicious joint readings {arm_deg}: expected degrees but the "
                "values look like radians/normalized units. Check that lerobot "
                ">= 0.5.1 is installed and the robot is calibrated (use_degrees=True)."
            )
        self.last_observation = obs

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None

    def _require_robot(self):
        if self._robot is None:
            raise RuntimeError("LeRobotBackend is not connected")
        return self._robot

    def read_state(self) -> RobotState:
        robot = self._require_robot()
        obs = robot.get_observation()
        self.last_observation = obs
        arm_deg = np.array([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=float)
        return RobotState(
            q=self._config.joint_map.from_real_deg(arm_deg),
            gripper=float(np.clip(obs["gripper.pos"] / 100.0, 0.0, 1.0)),
            t=time.monotonic(),
        )

    def write_targets(self, q: np.ndarray, gripper: float) -> None:
        robot = self._require_robot()
        deg = self._config.joint_map.to_real_deg(np.asarray(q, dtype=float))
        action = {f"{j}.pos": float(deg[i]) for i, j in enumerate(ARM_JOINTS)}
        action["gripper.pos"] = float(np.clip(gripper, 0.0, 1.0) * 100.0)
        robot.send_action(action)

    def set_torque(self, enabled: bool) -> None:
        """Best-effort torque toggle so MIRROR mode can hand-pose the arm."""
        robot = self._require_robot()
        try:
            bus = robot.bus
            if enabled:
                bus.enable_torque()
            else:
                bus.disable_torque()
        except Exception as exc:
            raise RuntimeError(f"torque toggle not supported by this lerobot version: {exc}")

    @property
    def is_real(self) -> bool:
        return True
