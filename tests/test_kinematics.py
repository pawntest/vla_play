"""Tests for so101_tool.kinematics (FK/IK against the vendored MJCF)."""

from __future__ import annotations

import numpy as np
import pytest

from so101_tool.config import ARM_LIMITS_HI, ARM_LIMITS_LO
from so101_tool.kinematics import IKError, Kinematics, SE3Pose


@pytest.fixture(scope="module")
def kin() -> Kinematics:
    return Kinematics()


def test_fk_at_zeros(kin: Kinematics):
    pose = kin.fk(np.zeros(5))
    assert np.all(np.isfinite(pose.position))
    assert np.all(np.isfinite(pose.wxyz))
    np.testing.assert_allclose(pose.position, [0.391, -0.001, 0.246], atol=1e-2)
    assert np.isclose(np.linalg.norm(pose.wxyz), 1.0, atol=1e-6)


def test_fk_ik_round_trip(kin: Kinematics):
    rng = np.random.default_rng(42)
    lo, hi = 0.7 * ARM_LIMITS_LO, 0.7 * ARM_LIMITS_HI
    failures = 0
    for _ in range(15):
        q_ref = rng.uniform(lo, hi)
        target = kin.fk(q_ref)
        try:
            q_sol = kin.ik(target, q_seed=np.zeros(5))
        except IKError:
            failures += 1
            continue
        err = np.linalg.norm(kin.fk(q_sol).position - target.position)
        assert err < 5e-3, f"position error {err * 1e3:.1f} mm for q_ref={q_ref}"
    assert failures <= 1, f"{failures} IK failures out of 15"


def test_ik_position_only_with_none_wxyz(kin: Kinematics):
    target_pos = kin.fk(np.array([0.3, -0.4, 0.5, 0.2, 0.0])).position
    q = kin.ik(SE3Pose(position=target_pos, wxyz=None), q_seed=np.zeros(5))
    assert q.shape == (5,)
    assert np.linalg.norm(kin.fk(q).position - target_pos) < 5e-3


def test_ik_unreachable_raises(kin: Kinematics):
    with pytest.raises(IKError):
        kin.ik(SE3Pose(position=np.array([1.5, 0.0, 0.5]), wxyz=None), q_seed=np.zeros(5))


def test_ik_velocity_reduces_distance(kin: Kinematics):
    q_now = np.zeros(5)
    target = kin.fk(np.array([0.4, -0.3, 0.4, 0.1, 0.0]))
    d0 = np.linalg.norm(kin.fk(q_now).position - target.position)
    q_next = kin.ik_velocity(target, q_now, dt=0.02)
    assert q_next.shape == (5,)
    d1 = np.linalg.norm(kin.fk(q_next).position - target.position)
    assert d1 < d0
