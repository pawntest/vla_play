"""Tests for so101_tool.robot.sim.SimBackend using a fake controllable clock."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from so101_tool.config import ARM_LIMITS_HI, JointMap
from so101_tool.robot.sim import SimBackend


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture()
def rig():
    clock = FakeClock()
    backend = SimBackend(joint_map=JointMap(), clock=clock)
    backend.connect()
    return backend, clock


def test_converges_in_expected_time(rig):
    backend, clock = rig
    target = np.array([1.0, -0.5, 0.8, 0.3, -1.2])
    backend.write_targets(target, 0.0)
    # Farthest joint is 1.2 rad -> 0.6 s at 2 rad/s. Step at 50 Hz.
    dt = 0.02
    for _ in range(int(0.6 / dt)):
        clock.advance(dt)
        state = backend.read_state()
    np.testing.assert_allclose(state.q, target, atol=1e-9)
    # And it wasn't there one full tick earlier: check monotonic approach.
    assert state.t == pytest.approx(0.6)


def test_never_exceeds_per_tick_vmax(rig):
    backend, clock = rig
    vmax = backend.joint_map.vmax_rad_s
    backend.write_targets(np.array([1.5, 1.5, 1.5, 1.5, 1.5]), 1.0)
    dt = 0.02
    prev = backend.read_state().q
    for _ in range(100):
        clock.advance(dt)
        q = backend.read_state().q
        assert np.all(np.abs(q - prev) <= vmax * dt + 1e-12)
        prev = q


def test_dt_capped_on_stall(rig):
    backend, clock = rig
    backend.write_targets(np.array([1.5, 0, 0, 0, 0]), 0.0)
    clock.advance(5.0)  # long stall: dt must cap at 0.1 s -> max 0.2 rad moved
    q = backend.read_state().q
    assert q[0] == pytest.approx(0.2)


def test_gripper_tracks(rig):
    backend, clock = rig
    backend.write_targets(np.zeros(5), 1.0)
    clock.advance(0.25)  # 2.0 fraction/s -> 0.5 in 0.25 s, but dt caps at 0.1
    assert backend.read_state().gripper == pytest.approx(0.2)
    for _ in range(4):
        clock.advance(0.1)
        state = backend.read_state()
    assert state.gripper == pytest.approx(1.0)


def test_targets_clamped(rig):
    backend, clock = rig
    backend.write_targets(np.full(5, 10.0), 3.0)
    clock.advance(100.0)
    for _ in range(200):
        clock.advance(0.1)
        state = backend.read_state()
    np.testing.assert_allclose(state.q, ARM_LIMITS_HI)
    assert state.gripper == 1.0


def test_disconnected_raises():
    backend = SimBackend()
    with pytest.raises(RuntimeError):
        backend.read_state()
    with pytest.raises(RuntimeError):
        backend.write_targets(np.zeros(5), 0.0)
    backend.connect()
    backend.read_state()
    backend.disconnect()
    with pytest.raises(RuntimeError):
        backend.read_state()


def test_concurrent_read_write_smoke(rig):
    backend, clock = rig
    errors: list[Exception] = []
    stop = threading.Event()

    def reader():
        try:
            while not stop.is_set():
                s = backend.read_state()
                assert s.q.shape == (5,)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def writer():
        try:
            rng = np.random.default_rng(0)
            for _ in range(500):
                backend.write_targets(rng.uniform(-1, 1, 5), rng.uniform(0, 1))
                clock.advance(0.001)
        except Exception as e:  # pragma: no cover
            errors.append(e)
        finally:
            stop.set()

    threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    assert all(not t.is_alive() for t in threads)
