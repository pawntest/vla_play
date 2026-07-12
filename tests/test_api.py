"""Public API (so101_tool.api.Session) tests — headless, no GUI."""

import numpy as np
import pytest

from so101_tool import api


def test_session_sim_motion():
    with api.Session(backend="sim", control_hz=200.0) as sess:
        sess.home()
        sess.move_joints([0.3, -0.2, 0.4, 0.1, 0.0], gripper=0.5, speed=1.0)
        snap = sess.state()
        assert np.max(np.abs(snap.q - [0.3, -0.2, 0.4, 0.1, 0.0])) < 0.05
        target = snap.tcp_position + [0.0, 0.0, 0.03]
        sess.move_to(target, speed=1.0)
        assert np.linalg.norm(sess.state().tcp_position - target) < 0.01


def test_session_scenario_objects():
    sc = api.Scenario(
        objects=[api.ObjectSpec(name="cube", pos=[0.25, 0, 0.02])], cameras=[]
    )
    with api.Session(scenario=sc, control_hz=200.0) as sess:
        pos, _ = sess.object_pose("cube")
        assert abs(pos[0] - 0.25) < 0.02
        sess.set_object_pose("cube", [0.3, 0.1, 0.05])
        pos, _ = sess.object_pose("cube")
        assert np.allclose(pos, [0.3, 0.1, 0.05], atol=1e-3)
        sess.reset_scene(randomize=False)


def test_session_motion_failure_raises():
    with api.Session(backend="sim", control_hz=200.0) as sess:
        with pytest.raises((RuntimeError, TimeoutError)):
            sess.move_to([1.5, 0.0, 0.5], timeout=12.0)  # unreachable


def test_session_camera_frames_require_cameras():
    with api.Session(backend="sim") as sess:
        with pytest.raises(RuntimeError, match="cameras"):
            sess.camera_frames()
