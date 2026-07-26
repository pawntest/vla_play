import numpy as np
import pytest

from so101_tool.config import AppConfig
from so101_tool.control.commands import Mode, MoveJ, SetMode
from so101_tool.control.loop import ControlLoop
from so101_tool.kinematics import Kinematics
from so101_tool.robot.physics_sim import PhysicsBackend
from so101_tool.scenario import ObjectSpec, Scenario


@pytest.fixture(scope="module")
def scenario():
    return Scenario(
        name="t",
        task="pick",
        objects=[ObjectSpec(name="cube", pos=[0.25, 0.0, 0.02], pos_noise=[0.02, 0.02, 0.0])],
        settle_steps=100,
    )


def test_physics_tracks_targets(scenario):
    backend = PhysicsBackend(scenario, seed=0)
    backend.connect()
    fake_now = [0.0]
    backend._clock = lambda: fake_now[0]
    backend._last_t = 0.0
    target = np.array([0.4, -0.5, 0.6, 0.2, 0.1])
    backend.write_targets(target, 0.5)
    for _ in range(40):
        fake_now[0] += 0.05
        state = backend.read_state()
    assert np.max(np.abs(state.q - target)) < 0.08  # position servos ~ track
    assert state.qpos_full is not None and len(state.qpos_full) == 13
    backend.disconnect()


def test_reset_randomizes_and_settles(scenario):
    backend = PhysicsBackend(scenario, seed=1)
    backend.connect()
    positions = []
    for _ in range(3):
        backend.reset(randomize=True)
        pos, _wxyz = backend.object_pose("cube")
        assert pos[2] < 0.05  # settled on the floor, not exploded
        positions.append(pos[:2].copy())
    assert np.std([p[0] for p in positions]) + np.std([p[1] for p in positions]) > 0
    backend.disconnect()


def test_camera_frames(scenario):
    backend = PhysicsBackend(scenario, render_width=64, render_height=48, seed=0)
    backend.connect()
    frames = backend.get_camera_frames()
    assert set(frames) == {"front", "top"}
    for img in frames.values():
        assert img.shape == (48, 64, 3) and img.dtype == np.uint8
        assert img.std() > 0  # not a blank frame
    backend.disconnect()


def test_full_loop_movej_with_physics(scenario):
    backend = PhysicsBackend(scenario, seed=0)
    backend.connect()
    loop = ControlLoop(backend, Kinematics(), AppConfig(control_hz=100.0))
    loop.start()
    try:
        mode = SetMode(mode=Mode.RULE)
        loop.commands.put(mode)
        assert mode.wait(2.0) and mode.ok
        mv = MoveJ(q=np.array([0.3, -0.3, 0.4, 0.1, 0.0]), speed=1.0)
        loop.commands.put(mv)
        assert mv.wait(20.0), "MoveJ did not finish under physics"
        assert mv.ok, mv.error
        snap = loop.snapshot()
        assert snap.qpos_full is not None
        assert snap.q_cmd is not None
    finally:
        loop.stop()
        loop.join(timeout=2.0)
        backend.disconnect()
