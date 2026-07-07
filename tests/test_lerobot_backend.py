"""LeRobotBackend unit tests with a fake SO101Follower (no lerobot install)."""

import sys
import types

import numpy as np
import pytest

from so101_tool.config import ARM_JOINTS, AppConfig, JointMap
from so101_tool.robot.lerobot_backend import LeRobotBackend

HOME_DEG = {f"{j}.pos": v for j, v in zip(ARM_JOINTS, [10.0, -35.0, 40.0, 20.0, -5.0])}


class FakeFollower:
    def __init__(self, cfg):
        self.cfg = cfg
        self.connected = False
        self.sent: list[dict] = []
        self.obs = dict(HOME_DEG, **{"gripper.pos": 50.0})

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        return dict(self.obs)

    def send_action(self, action):
        self.sent.append(action)


@pytest.fixture
def fake_lerobot(monkeypatch):
    mod = types.ModuleType("lerobot.robots.so_follower")
    mod.SO101Follower = FakeFollower
    mod.SO101FollowerConfig = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.robots", types.ModuleType("lerobot.robots"))
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", mod)
    return mod


def test_read_write_conversions(fake_lerobot):
    backend = LeRobotBackend(AppConfig(backend="real"))
    backend.connect()
    state = backend.read_state()
    expected_rad = np.radians([10.0, -35.0, 40.0, 20.0, -5.0])
    assert np.allclose(state.q, expected_rad)
    assert state.gripper == pytest.approx(0.5)

    backend.write_targets(expected_rad, 0.25)
    sent = backend.lerobot_robot.sent[-1]
    for j, deg in zip(ARM_JOINTS, [10.0, -35.0, 40.0, 20.0, -5.0]):
        assert sent[f"{j}.pos"] == pytest.approx(deg)
    assert sent["gripper.pos"] == pytest.approx(25.0)
    backend.disconnect()
    assert backend.lerobot_robot is None


def test_joint_map_signs_offsets(fake_lerobot):
    jm = JointMap(signs=np.array([-1.0, 1.0, 1.0, 1.0, 1.0]),
                  offsets_deg=np.array([5.0, 0.0, 0.0, 0.0, 0.0]))
    backend = LeRobotBackend(AppConfig(backend="real", joint_map=jm))
    backend.connect()
    q = backend.read_state().q
    # real 10 deg with sign -1, offset 5: internal = (10-5)/-1 = -5 deg
    assert q[0] == pytest.approx(np.radians(-5.0))
    backend.write_targets(q, 0.0)
    assert backend.lerobot_robot.sent[-1]["shoulder_pan.pos"] == pytest.approx(10.0)


def test_radianlike_observation_rejected(fake_lerobot):
    fake_lerobot.SO101Follower.obs = None  # not used; patch per instance below

    class RadianFollower(FakeFollower):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.obs = {f"{j}.pos": v for j, v in zip(ARM_JOINTS, [0.17, -0.6, 0.7, 0.35, -0.1])}
            self.obs["gripper.pos"] = 50.0

    fake_lerobot.SO101Follower = RadianFollower
    backend = LeRobotBackend(AppConfig(backend="real"))
    with pytest.raises(RuntimeError, match="Suspicious joint readings"):
        backend.connect()


def test_missing_lerobot_message():
    backend = LeRobotBackend(AppConfig(backend="real"))
    assert "lerobot" not in sys.modules or True  # env has no lerobot either way
    with pytest.raises(RuntimeError, match="so101-tool\\[real\\]"):
        backend.connect()


def test_read_requires_connect():
    backend = LeRobotBackend(AppConfig(backend="real"))
    with pytest.raises(RuntimeError, match="not connected"):
        backend.read_state()
