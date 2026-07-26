"""Tests for so101_tool.control.safety.SafetyFilter."""

from __future__ import annotations

import numpy as np
import pytest

from so101_tool.config import ARM_LIMITS_HI, ARM_LIMITS_LO, JointMap
from so101_tool.control.safety import SafetyFilter
from so101_tool.kinematics import Kinematics
from so101_tool.robot.base import RobotState


def make_state(q=None, gripper=0.5) -> RobotState:
    return RobotState(q=np.zeros(5) if q is None else np.asarray(q, float), gripper=gripper, t=0.0)


@pytest.fixture()
def filt() -> SafetyFilter:
    return SafetyFilter(JointMap())


def test_joint_limit_clamp(filt: SafetyFilter):
    # Big dt so the rate limit doesn't bind; targets beyond limits clamp to them.
    q_safe, _ = filt.filter(ARM_LIMITS_HI + 1.0, 0.5, make_state(), dt=10.0)
    np.testing.assert_allclose(q_safe, ARM_LIMITS_HI)
    q_safe, _ = filt.filter(ARM_LIMITS_LO - 1.0, 0.5, make_state(), dt=10.0)
    np.testing.assert_allclose(q_safe, ARM_LIMITS_LO)


def test_velocity_rate_limit_exact():
    jm = JointMap()
    filt = SafetyFilter(jm, vel_margin=1.5)
    dt = 0.02
    state = make_state(q=np.full(5, 0.1))
    target = np.full(5, 1.0)  # far beyond one tick's reach
    q_safe, _ = filt.filter(target, 0.5, state, dt)
    np.testing.assert_allclose(q_safe, state.q + jm.vmax_rad_s * dt * 1.5)
    # Negative direction, and a target within reach passes through unchanged.
    q_safe, _ = filt.filter(-target, 0.5, state, dt)
    np.testing.assert_allclose(q_safe, state.q - jm.vmax_rad_s * dt * 1.5)
    near = state.q + 0.01
    q_safe, _ = filt.filter(near, 0.5, state, dt)
    np.testing.assert_allclose(q_safe, near)


def test_gripper_clamp(filt: SafetyFilter):
    state = make_state()
    assert filt.filter(np.zeros(5), 1.7, state, 0.02)[1] == 1.0
    assert filt.filter(np.zeros(5), -0.3, state, 0.02)[1] == 0.0
    assert filt.filter(np.zeros(5), 0.4, state, 0.02)[1] == 0.4


def test_floor_rejection():
    kin = Kinematics()
    filt = SafetyFilter(JointMap(), kinematics=kin, floor_z=0.005)
    # Pitch the arm down hard: TCP goes well below z=0.
    q_low = np.array([0.0, 1.7, 0.0, 0.0, 0.0])
    assert kin.fk(q_low).position[2] < 0.0
    state = make_state(q=q_low * 0.999)  # already near the bad pose
    q_safe, _ = filt.filter(q_low, 0.5, state, dt=0.02)
    np.testing.assert_array_equal(q_safe, state.q)  # held, not advanced
    # A safe target well above the floor is not rejected.
    state0 = make_state()
    q_safe, _ = filt.filter(np.array([0.1, 0.0, 0.0, 0.0, 0.0]), 0.5, state0, dt=1.0)
    assert not np.array_equal(q_safe, state0.q)


def test_dt_zero_guard(filt: SafetyFilter):
    state = make_state(q=np.full(5, 0.2))
    q_safe, g = filt.filter(np.ones(5), 2.0, state, dt=0.0)
    np.testing.assert_array_equal(q_safe, state.q)
    assert g == 1.0
    q_safe, _ = filt.filter(np.ones(5), 0.5, state, dt=-0.01)
    np.testing.assert_array_equal(q_safe, state.q)
