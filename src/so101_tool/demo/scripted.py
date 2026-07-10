"""Scripted expert for automatic demonstration generation in the physics sim.

Runs a pick-and-lift sequence against a scenario object using the same motion
primitives a human would trigger from the GUI, sampling frames at a fixed rate
into a recorder callback. This turns any pick-style scenario into a labelled
imitation-learning dataset without teleoperation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..control.commands import Command, Mode, MoveJ, SetGripper, SetMode
from ..kinematics import IKError, Kinematics, SE3Pose
from ..robot.physics_sim import PhysicsBackend

# sample_cb() is called at the recording rate while motions execute
SampleCallback = Callable[[], None]


@dataclass
class PickParams:
    object_name: str = "cube"
    approach_height: float = 0.07  # m above the object for the pre-grasp pose
    grasp_offset: tuple = (0.0, 0.0, 0.008)  # TCP offset from object center at grasp
    open_fraction: float = 1.0
    close_fraction: float = 0.0  # jaws stall on the object — timeout counts as grasped
    lift_height: float = 0.12
    success_lift: float = 0.05  # object must rise this much above its start z
    speed: float = 0.7
    step_timeout: float = 12.0


class ScriptedPick:
    def __init__(
        self,
        loop,
        backend: PhysicsBackend,
        kinematics: Kinematics,
        params: PickParams | None = None,
        sample_hz: float = 15.0,
    ):
        self._loop = loop
        self._backend = backend
        self._kin = kinematics  # own instance, not the loop's
        self.params = params or PickParams()
        self._sample_dt = 1.0 / sample_hz

    # -- helpers ---------------------------------------------------------------

    def _run(self, cmd: Command, sample_cb: SampleCallback | None) -> bool:
        """Execute one primitive, sampling frames while it runs.

        Returns True if the command completed cleanly, False on error/timeout.
        """
        self._loop.commands.put(cmd)
        deadline = time.monotonic() + self.params.step_timeout
        next_sample = 0.0
        while not cmd.wait(0.01):
            now = time.monotonic()
            if now >= next_sample and sample_cb is not None:
                sample_cb()
                next_sample = now + self._sample_dt
            if now > deadline:
                return False
        return cmd.ok

    def _ik_movej(self, position: np.ndarray, gripper: float | None) -> MoveJ:
        snap = self._loop.snapshot()
        seed = snap.q if snap is not None else np.zeros(5)
        q = self._kin.ik(SE3Pose(position=position, wxyz=None), seed)
        return MoveJ(q=q, gripper=gripper, speed=self.params.speed)

    # -- public API --------------------------------------------------------------

    def run_episode(self, sample_cb: SampleCallback | None = None) -> bool:
        """One pick-and-lift attempt. Returns True on success (object lifted)."""
        p = self.params
        obj_pos, _ = self._backend.object_pose(p.object_name)
        start_z = obj_pos[2]
        grasp = obj_pos + np.asarray(p.grasp_offset)
        above = grasp + [0.0, 0.0, p.approach_height]

        mode = SetMode(mode=Mode.RULE)
        self._loop.commands.put(mode)
        if not (mode.wait(5.0) and mode.ok):
            return False
        try:
            steps = [
                self._ik_movej(above, p.open_fraction),
                self._ik_movej(grasp, None),
            ]
        except IKError:
            return False
        for cmd in steps:
            if not self._run(cmd, sample_cb):
                return False
        # Close on the object: the jaws stall against it, so a timeout here
        # usually MEANS a firm grasp. Treat both outcomes as "try lifting".
        self._run(SetGripper(fraction=p.close_fraction), sample_cb)
        try:
            lift = self._ik_movej(grasp + [0.0, 0.0, p.lift_height], None)
        except IKError:
            return False
        if not self._run(lift, sample_cb):
            return False
        end_z = self._backend.object_pose(p.object_name)[0][2]
        return bool(end_z - start_z > p.success_lift)
