"""Physics-simulated backend: MuJoCo mj_step over a Scenario model.

Unlike SimBackend (pure kinematic integrator), this backend steps real contact
physics so objects can be pushed, grasped and lifted. The arm is driven by the
MJCF position actuators (write_targets -> data.ctrl); read_state() advances the
simulation by wall-clock time. Cameras defined in the scenario (plus the arm's
built-in wrist_cam) render offscreen for policy inference and data recording.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import mujoco
import numpy as np

from ..config import (
    ARM_LIMITS_HI,
    ARM_LIMITS_LO,
    JointMap,
    ensure_headless_gl,
    gripper_fraction_to_rad,
    gripper_rad_to_fraction,
)
from ..scenario import Scenario, build_model, randomize_object_qpos
from .base import RobotInterface, RobotState

_DT_CAP = 0.1


class PhysicsBackend(RobotInterface):
    def __init__(
        self,
        scenario: Scenario,
        joint_map: JointMap | None = None,
        render_width: int = 320,
        render_height: int = 240,
        seed: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.scenario = scenario
        self.joint_map = joint_map or JointMap()
        self.model = build_model(scenario)
        self._data = mujoco.MjData(self.model)
        self._render_wh = (render_width, render_height)
        self._rng = np.random.default_rng(seed)
        self._clock = clock
        self._lock = threading.Lock()
        self._renderer: mujoco.Renderer | None = None
        self._last_t: float | None = None
        self._sim_leftover = 0.0
        self.connected = False
        # arm joints are the first 6 qpos / actuators (arm spec compiles first)
        self._camera_names = [c.name for c in scenario.cameras]
        model_cams = {self.model.camera(i).name for i in range(self.model.ncam)}
        self._camera_names = [c for c in self._camera_names if c in model_cams]
        self.reset(randomize=True)

    # -- RobotInterface -------------------------------------------------------

    def connect(self) -> None:
        with self._lock:
            self.connected = True
            self._last_t = self._clock()

    def disconnect(self) -> None:
        with self._lock:
            self.connected = False

    def read_state(self) -> RobotState:
        with self._lock:
            if not self.connected:
                raise RuntimeError("PhysicsBackend not connected")
            now = self._clock()
            dt = min(now - (self._last_t if self._last_t is not None else now), _DT_CAP)
            self._last_t = now
            self._advance(dt)
            return self._state_locked(now)

    def write_targets(self, q: np.ndarray, gripper: float) -> None:
        with self._lock:
            if not self.connected:
                raise RuntimeError("PhysicsBackend not connected")
            self._data.ctrl[:5] = np.clip(np.asarray(q, dtype=float), ARM_LIMITS_LO, ARM_LIMITS_HI)
            self._data.ctrl[5] = gripper_fraction_to_rad(gripper)

    @property
    def is_real(self) -> bool:
        return False

    # -- physics/scene extras ---------------------------------------------------

    def _advance(self, dt: float) -> None:
        self._sim_leftover += dt
        timestep = self.model.opt.timestep
        steps = int(self._sim_leftover / timestep)
        self._sim_leftover -= steps * timestep
        for _ in range(min(steps, int(_DT_CAP / timestep))):
            mujoco.mj_step(self.model, self._data)

    def _state_locked(self, now: float) -> RobotState:
        return RobotState(
            q=self._data.qpos[:5].copy(),
            gripper=gripper_rad_to_fraction(float(self._data.qpos[5])),
            t=now,
            qpos_full=self._data.qpos.copy(),
        )

    def reset(self, randomize: bool = True) -> None:
        """Reset arm to home and objects to their (optionally randomized) spawn poses."""
        with self._lock:
            mujoco.mj_resetData(self.model, self._data)
            if randomize:
                randomize_object_qpos(self.scenario, self.model, self._data, self._rng)
            self._data.ctrl[:] = self._data.qpos[: self.model.nu]
            for _ in range(self.scenario.settle_steps):
                mujoco.mj_step(self.model, self._data)
            self._sim_leftover = 0.0
            self._last_t = self._clock()

    def qpos_full(self) -> np.ndarray:
        with self._lock:
            return self._data.qpos.copy()

    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World (position, wxyz) of a scenario object (cloth: vertex centroid)."""
        with self._lock:
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, f"obj_{name}")
            if fid >= 0:
                adr, num = self.model.flex_vertadr[fid], self.model.flex_vertnum[fid]
                center = self._data.flexvert_xpos[adr : adr + num].mean(axis=0)
                return center.copy(), np.array([1.0, 0.0, 0.0, 0.0])
            bid = self.model.body(f"obj_{name}").id
            return self._data.xpos[bid].copy(), self._data.xquat[bid].copy()

    def set_object_pose(self, name: str, pos, wxyz=None) -> None:
        """Teleport an object (GUI drag). Cloth is translated rigidly."""
        from ..scenario import _shift_cloth

        pos = np.asarray(pos, dtype=float)
        with self._lock:
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, f"obj_{name}")
            if fid >= 0:
                adr, num = self.model.flex_vertadr[fid], self.model.flex_vertnum[fid]
                center = self._data.flexvert_xpos[adr : adr + num].mean(axis=0)
                _shift_cloth(self.model, self._data, f"obj_{name}", pos - center)
            else:
                jid = self.model.joint(f"free_{name}").id
                adr = self.model.jnt_qposadr[jid]
                self._data.qpos[adr : adr + 3] = pos
                if wxyz is not None:
                    self._data.qpos[adr + 3 : adr + 7] = np.asarray(wxyz, dtype=float)
                self._data.qvel[self.model.jnt_dofadr[jid] : self.model.jnt_dofadr[jid] + 6] = 0.0
            mujoco.mj_forward(self.model, self._data)

    @property
    def camera_names(self) -> list[str]:
        return list(self._camera_names)

    def get_camera_frames(self) -> dict[str, np.ndarray]:
        """Render all scenario cameras (uint8 HxWx3). Call from ONE thread only."""
        if self._renderer is None:
            ensure_headless_gl()
            h, w = self._render_wh[1], self._render_wh[0]
            self._renderer = mujoco.Renderer(self.model, h, w)
        frames = {}
        with self._lock:
            for name in self._camera_names:
                self._renderer.update_scene(self._data, camera=name)
                frames[name] = self._renderer.render()
        return frames
