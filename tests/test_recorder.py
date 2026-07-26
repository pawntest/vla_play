"""Tests for so101_tool.data.recorder against a real lerobot install.

Skipped entirely when lerobot is not installed (e.g. the py3.11 dev venv).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from so101_tool.config import JointMap  # noqa: E402
from so101_tool.data.recorder import DatasetRecorder  # noqa: E402

FPS = 15
CAMERAS = {"front": (96, 128)}  # name -> (height, width)
STATE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def make_joint_map() -> JointMap:
    # Nontrivial signs/offsets so unit conversion is actually exercised.
    return JointMap(
        signs=np.array([1.0, -1.0, 1.0, -1.0, 1.0]),
        offsets_deg=np.array([0.0, 10.0, -5.0, 3.0, 90.0]),
    )


def synthetic_frame(i: int, rng: np.random.Generator):
    q = np.linspace(-0.4, 0.4, 5) * (i + 1) / 8.0  # ramping joints, radians
    gripper = i / 10.0
    q_cmd = q + 0.01
    gripper_cmd = min(gripper + 0.05, 1.0)
    img = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    return q, gripper, q_cmd, gripper_cmd, {"front": img}


def record_episodes(rec: DatasetRecorder, n_episodes: int, n_frames: int, seed: int = 0):
    """Record episodes of synthetic data; returns the first frame's inputs."""
    rng = np.random.default_rng(seed)
    first = None
    for _ in range(n_episodes):
        rec.start_episode()
        for i in range(n_frames):
            q, g, q_cmd, g_cmd, images = synthetic_frame(i, rng)
            if first is None:
                first = (q, g, q_cmd, g_cmd)
            rec.add_frame(q, g, q_cmd, g_cmd, images)
        rec.end_episode(save=True)
    return first


def test_record_and_reopen(tmp_path):
    jm = make_joint_map()
    root = tmp_path / "ds"
    rec = DatasetRecorder(
        "test/so101_rec", fps=FPS, cameras=CAMERAS, joint_map=jm, root=root, task="pick"
    )
    q0, g0, q_cmd0, g_cmd0 = record_episodes(rec, n_episodes=2, n_frames=8)
    out = rec.finalize()
    assert out == root

    ds = LeRobotDataset("test/so101_rec", root=root, video_backend="pyav")
    assert ds.num_episodes == 2
    assert ds.num_frames == 16
    assert ds.meta.fps == FPS

    # Feature schema: SO101-style names, units checked below.
    assert list(ds.meta.features["observation.state"]["names"]) == STATE_NAMES
    assert list(ds.meta.features["action"]["names"]) == STATE_NAMES
    cam_key = "observation.images.front"
    assert cam_key in ds.meta.features
    assert ds.meta.features[cam_key]["dtype"] == "video"
    assert tuple(ds.meta.features[cam_key]["shape"]) == (96, 128, 3)

    # First frame round-trips through the joint map: degrees + gripper 0..100.
    item = ds[0]
    expected_state = np.append(jm.to_real_deg(q0), np.clip(g0, 0, 1) * 100.0)
    expected_action = np.append(jm.to_real_deg(q_cmd0), np.clip(g_cmd0, 0, 1) * 100.0)
    np.testing.assert_allclose(np.asarray(item["observation.state"]), expected_state, atol=1e-4)
    np.testing.assert_allclose(np.asarray(item["action"]), expected_action, atol=1e-4)

    # Camera frame decodes with the right pixel dimensions (channel-first or -last).
    img = np.asarray(item[cam_key])
    assert img.shape in ((3, 96, 128), (96, 128, 3))


def test_discarded_episode_not_saved(tmp_path):
    jm = make_joint_map()
    root = tmp_path / "ds"
    rec = DatasetRecorder(
        "test/so101_disc", fps=FPS, cameras=CAMERAS, joint_map=jm, root=root, task="pick"
    )
    record_episodes(rec, n_episodes=1, n_frames=8)

    rng = np.random.default_rng(1)
    rec.start_episode()
    for i in range(5):
        q, g, q_cmd, g_cmd, images = synthetic_frame(i, rng)
        rec.add_frame(q, g, q_cmd, g_cmd, images)
    assert rec.num_frames_in_episode == 5
    rec.end_episode(save=False)  # discard
    rec.finalize()

    ds = LeRobotDataset("test/so101_disc", root=root, video_backend="pyav")
    assert ds.num_episodes == 1
    assert ds.num_frames == 8


def test_resume_appends(tmp_path):
    jm = make_joint_map()
    root = tmp_path / "ds"
    rec = DatasetRecorder(
        "test/so101_res", fps=FPS, cameras=CAMERAS, joint_map=jm, root=root, task="pick"
    )
    record_episodes(rec, n_episodes=2, n_frames=8)
    rec.finalize()

    # Same root without resume must fail loudly.
    with pytest.raises(RuntimeError, match="resume=True"):
        DatasetRecorder(
            "test/so101_res", fps=FPS, cameras=CAMERAS, joint_map=jm, root=root, task="pick"
        )

    rec2 = DatasetRecorder(
        "test/so101_res",
        fps=FPS,
        cameras=CAMERAS,
        joint_map=jm,
        root=root,
        task="pick",
        resume=True,
    )
    assert rec2.num_episodes == 2
    record_episodes(rec2, n_episodes=1, n_frames=8, seed=2)
    rec2.finalize()

    ds = LeRobotDataset("test/so101_res", root=root, video_backend="pyav")
    assert ds.num_episodes == 3
    assert ds.num_frames == 24
