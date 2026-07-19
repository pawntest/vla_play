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

Failure policy: the real arm must never take the app down. connect() brings
the sim up immediately and dials the real side in a BACKGROUND thread
(lerobot's import alone takes tens of seconds); until it succeeds — and
whenever it later fails — the link degrades to plain sim and `real_status`
reports why, so the UI keeps running and shows what's wrong instead of dying.
"""

from __future__ import annotations

import threading
import time

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
        self._real_ok = False
        self._real_live = False  # last read reported connected (e.g. teleop stream fresh)
        self.real_status = "disconnected"

    # -- link mode (runtime switchable from the GUI) ------------------------

    @property
    def link(self) -> str:
        return self._link

    def set_link(self, link: str) -> None:
        if link not in LINK_MODES:
            raise ValueError(f"link must be one of {LINK_MODES}")
        with self._lock:
            self._link = link

    # -- real-side health ----------------------------------------------------

    @property
    def real_ok(self) -> bool:
        with self._lock:
            return self._real_ok

    @property
    def real_live(self) -> bool:
        """Connected AND delivering state (a remote arm with no fresh stream
        is ok-but-not-live; a serial arm is live whenever it is ok)."""
        with self._lock:
            return self._real_ok and self._real_live

    def wait_real(self, timeout: float = 60.0) -> bool:
        """Block until the real side is connected (scripts/tests). False on
        timeout or connection error."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.real_ok:
                return True
            if self.real_status.startswith("error"):
                return False
            time.sleep(0.05)
        return False

    def _mark_real_down(self, exc: Exception) -> None:
        with self._lock:
            self._real_ok = False
        self.real_status = f"error: {exc}"

    # -- RobotInterface -----------------------------------------------------

    def connect(self) -> None:
        self.sim.connect()
        self.real_status = "connecting"
        threading.Thread(target=self._connect_real, daemon=True,
                         name="so101-link-connect").start()

    def _connect_real(self) -> None:
        try:
            self.real.connect()
        except Exception as exc:
            self._mark_real_down(exc)
            return
        with self._lock:
            self._real_ok = True
        self.real_status = "connected"

    def reconnect_real(self) -> None:
        """Retry the real-side connection (e.g. after plugging in the motor
        power supply or fixing the port) without restarting the app."""
        if self.real_status == "connecting":
            return  # already dialing
        with self._lock:
            self._real_ok = False
            self._real_live = False
        try:
            self.real.disconnect()
        except Exception:
            pass
        self.real_status = "connecting"
        threading.Thread(target=self._connect_real, daemon=True,
                         name="so101-link-reconnect").start()

    def disconnect(self) -> None:
        with self._lock:
            self._real_ok = False
        for side in (self.real, self.sim):
            try:
                side.disconnect()
            except Exception:
                pass

    def read_state(self) -> RobotState:
        with self._lock:
            link, real_ok = self._link, self._real_ok
        if link == "to_real" or not real_ok:
            # sim is the source — either by mode, or as the degraded fallback
            # while the real arm is still connecting / has dropped out
            return self.sim.read_state()
        # to_sim / both: real is ground truth, sim chases it every tick
        try:
            real_state = self.real.read_state()
        except Exception as exc:
            self._mark_real_down(exc)
            return self.sim.read_state()
        with self._lock:
            self._real_live = real_state.connected
        if not real_state.connected:
            # e.g. a remote arm whose stream hasn't started / went stale:
            # don't drag the sim to the placeholder pose — sim is the source
            return self.sim.read_state()
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
            link, real_ok = self._link, self._real_ok
            live = self._real_live
        if link == "to_sim":
            return  # real arm is the source: commands are dropped
        if link == "to_real" or not (real_ok and live):
            # drive the sim directly — in "both" this is the degraded path
            # that keeps the preview responding while the real arm is away
            self.sim.write_targets(q, gripper)
        if not real_ok:
            return
        try:
            self.real.write_targets(q, gripper)
        except Exception as exc:
            self._mark_real_down(exc)
            if link == "both":
                self.sim.write_targets(q, gripper)  # don't lose the command

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
