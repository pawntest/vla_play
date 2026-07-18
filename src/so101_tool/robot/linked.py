"""LinkedBackend: couple the MuJoCo simulation with a real arm, both ways.

One backend pair behind the normal RobotInterface, with a runtime-switchable
link direction:

  - "to_sim"  (実機→Mujoco): the real arm is the source of truth and the sim
    arm follows it (the sim's position servos chase the real joints, so the
    sim arm physically interacts with scene objects). Loop writes are dropped
    — hand-pose the real arm (torque off) or drive it externally.
  - "to_real" (Mujoco→実機): commands drive the SIM, and the same targets are
    shadowed to the real arm. State comes from the sim (preview-first).
  - "both"    (双方向): the real arm is read as ground truth and the sim
    always follows it; commands go to the real arm (the sim inherits them by
    following). Hand-moving the real arm shows up in the sim; GUI/policy/NL
    moves both.

The "real" side is any RobotInterface — a local LeRobotBackend (serial) or a
RemoteArmBackend (the operator's arm across an SSH tunnel). The sim side must
expose `.model` (and provides qpos_full/cameras for the 3D view and datasets).
"""

from __future__ import annotations

import threading

import numpy as np

from .base import RobotInterface, RobotState

LINK_MODES = ("to_sim", "to_real", "both")


class LinkedBackend(RobotInterface):
    def __init__(self, sim: RobotInterface, real: RobotInterface, link: str = "both"):
        if link not in LINK_MODES:
            raise ValueError(f"link must be one of {LINK_MODES}")
        self.sim = sim
        self.real = real
        self._link = link
        self._lock = threading.Lock()

    # -- link mode (runtime switchable from the GUI) ------------------------

    @property
    def link(self) -> str:
        return self._link

    def set_link(self, link: str) -> None:
        if link not in LINK_MODES:
            raise ValueError(f"link must be one of {LINK_MODES}")
        with self._lock:
            self._link = link

    # -- RobotInterface -----------------------------------------------------

    def connect(self) -> None:
        self.sim.connect()
        self.real.connect()

    def disconnect(self) -> None:
        for side in (self.real, self.sim):
            try:
                side.disconnect()
            except Exception:
                pass

    def read_state(self) -> RobotState:
        with self._lock:
            link = self._link
        if link == "to_real":
            # sim is the source; the real arm just shadows written targets
            return self.sim.read_state()
        # to_sim / both: real is ground truth, sim chases it every tick
        real_state = self.real.read_state()
        self.sim.write_targets(real_state.q, real_state.gripper)
        sim_state = self.sim.read_state()  # advances sim physics
        return RobotState(
            q=real_state.q,
            gripper=real_state.gripper,
            t=real_state.t,
            connected=real_state.connected,
            qpos_full=sim_state.qpos_full,  # scene objects come from the sim
        )

    def write_targets(self, q: np.ndarray, gripper: float) -> None:
        with self._lock:
            link = self._link
        if link == "to_sim":
            return  # real arm is the source: commands are dropped
        if link == "to_real":
            self.sim.write_targets(q, gripper)
        # to_real and both: shadow/drive the real arm
        try:
            self.real.write_targets(q, gripper)
        except Exception:
            if link == "to_real":
                return  # sim keeps working even if the real link hiccups
            raise

    @property
    def is_real(self) -> bool:
        return True

    # -- sim-side passthrough (3D view, cameras, scene) ------------------------

    @property
    def model(self):
        return self.sim.model

    def get_camera_frames(self):
        return self.sim.get_camera_frames()

    def reset(self, randomize: bool = True) -> None:
        if hasattr(self.sim, "reset"):
            self.sim.reset(randomize=randomize)

    def object_pose(self, name: str):
        return self.sim.object_pose(name)

    def set_object_pose(self, name: str, pos, wxyz=None) -> None:
        self.sim.set_object_pose(name, pos, wxyz)
