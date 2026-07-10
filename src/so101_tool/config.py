"""Central configuration and joint conventions for so101_tool.

Canonical internal units, used EVERYWHERE inside this package:
  - arm joints: numpy array q of shape (5,), radians, ordered as ARM_JOINTS
  - gripper: float in [0, 1] (0 = fully closed, 1 = fully open)
Only backend boundaries (e.g. the lerobot backend) convert to other units.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Joint order matches both the vendored MJCF (robotstudio_so101) and
# lerobot's SO101 motor names (IDs 1..6).
ARM_JOINTS: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_JOINT = "gripper"
ALL_JOINTS: list[str] = ARM_JOINTS + [GRIPPER_JOINT]

# Joint limits in radians, from the vendored MJCF (robotstudio_so101/so101.xml).
ARM_LIMITS_LO = np.array([-1.91986, -1.7453293, -1.69, -1.658063, -2.7438473])
ARM_LIMITS_HI = np.array([1.91986, 1.7453293, 1.69, 1.658063, 2.7438473])
GRIPPER_LIMIT_LO = -0.174533  # rad, MJCF gripper joint (0.0 fraction maps here)
GRIPPER_LIMIT_HI = 1.7453292  # rad (1.0 fraction maps here)

# TCP site name in the MJCF (between the gripper jaws).
TCP_SITE = "gripperframe"
BASE_SITE = "baseframe"


def so101_mjcf_path() -> Path:
    """Path to the vendored SO-101 MJCF model."""
    return Path(__file__).parent / "assets" / "so101" / "so101.xml"


def ensure_headless_gl() -> None:
    """Pick a MUJOCO_GL that works without a display.

    MUST run before the first `import mujoco` in the process (mujoco binds its
    GL platform at import time). OSMesa (pure CPU) is the safe default; EGL is
    faster but needs a GPU/EGL device — set MUJOCO_GL=egl explicitly for that.
    """
    import ctypes.util

    if os.environ.get("MUJOCO_GL") or os.environ.get("DISPLAY"):
        return
    if ctypes.util.find_library("OSMesa"):
        os.environ["MUJOCO_GL"] = "osmesa"
    elif ctypes.util.find_library("EGL"):
        os.environ["MUJOCO_GL"] = "egl"


def gripper_fraction_to_rad(fraction: float) -> float:
    f = float(np.clip(fraction, 0.0, 1.0))
    return GRIPPER_LIMIT_LO + f * (GRIPPER_LIMIT_HI - GRIPPER_LIMIT_LO)


def gripper_rad_to_fraction(rad: float) -> float:
    f = (rad - GRIPPER_LIMIT_LO) / (GRIPPER_LIMIT_HI - GRIPPER_LIMIT_LO)
    return float(np.clip(f, 0.0, 1.0))


@dataclass
class JointMap:
    """Mapping between internal joint convention and a real robot.

    real_deg = sign * rad2deg(q_internal) + offset_deg   (per arm joint)
    Real-robot gripper is 0..100 (lerobot); internal is 0..1.
    A sign/offset mismatch on real hardware is fixed HERE, in data, not code.
    """

    signs: np.ndarray = field(default_factory=lambda: np.ones(5))
    offsets_deg: np.ndarray = field(default_factory=lambda: np.zeros(5))
    # Conservative velocity limit per arm joint (rad/s). STS3215 can do far
    # more; keep motions gentle by default.
    vmax_rad_s: np.ndarray = field(default_factory=lambda: np.full(5, 2.0))

    def to_real_deg(self, q: np.ndarray) -> np.ndarray:
        return self.signs * np.degrees(q) + self.offsets_deg

    def from_real_deg(self, deg: np.ndarray) -> np.ndarray:
        return np.radians((deg - self.offsets_deg) / self.signs)


@dataclass
class AppConfig:
    backend: str = "sim"  # "sim" | "real"
    port: str = "/dev/ttyACM0"  # serial port for the real robot
    robot_id: str = "so101_follower"  # lerobot calibration id
    viser_port: int = 8080
    control_hz: float = 50.0
    render_hz: float = 30.0
    joint_map: JointMap = field(default_factory=JointMap)
    # Natural-language control (enabled when an API key resolves)
    nl_model: str = field(
        default_factory=lambda: os.environ.get("SO101_NL_MODEL", "claude-opus-4-8")
    )
    # Optional lerobot policy checkpoint (POLICY mode)
    policy_path: str | None = None
    policy_task: str | None = None  # task instruction for VLA policies
    # lerobot camera configs passed through to SO101FollowerConfig, e.g.
    # {"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}
    cameras: dict = field(default_factory=dict)
