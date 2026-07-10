"""Record demonstrations into a lerobot ``LeRobotDataset``.

Dataset conventions match what ``lerobot-record`` produces for a real SO-101
follower, so policies trained on these datasets transfer:
  - ``observation.state`` / ``action``: float32 (6,) named
    ``["shoulder_pan.pos", ..., "gripper.pos"]``; arm joints in real-robot
    degrees (via ``JointMap.to_real_deg``), gripper in 0..100.
  - ``observation.images.<name>``: "video" dtype, (h, w, 3) uint8 frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101_tool.config import ALL_JOINTS, JointMap

_LEROBOT_HINT = (
    'data recording requires lerobot: pip install "so101-tool[data]" (Python >= 3.12)'
)

STATE_NAMES = [f"{joint}.pos" for joint in ALL_JOINTS]


class DatasetRecorder:
    """Buffered, episode-oriented writer around ``LeRobotDataset``."""

    def __init__(
        self,
        repo_id: str,
        fps: int,
        cameras: dict[str, tuple[int, int]],  # name -> (height, width)
        joint_map: JointMap,
        root: str | Path | None = None,
        task: str = "",
        resume: bool = False,
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.utils.constants import HF_LEROBOT_HOME
        except ImportError as e:
            raise RuntimeError(_LEROBOT_HINT) from e

        self._joint_map = joint_map
        self._default_task = task
        self._camera_names = list(cameras)
        self._episode_open = False
        self._frames_in_episode = 0

        root_path = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        exists = (root_path / "meta" / "info.json").exists()
        if exists and not resume:
            raise RuntimeError(
                f"dataset '{repo_id}' already exists at {root_path}; "
                "pass resume=True to append episodes, or choose a different root/repo_id"
            )
        if exists:
            self._ds = LeRobotDataset.resume(repo_id, root=root_path)
        else:
            features: dict[str, dict] = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (len(STATE_NAMES),),
                    "names": list(STATE_NAMES),
                },
                "action": {
                    "dtype": "float32",
                    "shape": (len(STATE_NAMES),),
                    "names": list(STATE_NAMES),
                },
            }
            for name, (h, w) in cameras.items():
                features[f"observation.images.{name}"] = {
                    "dtype": "video",
                    "shape": (int(h), int(w), 3),
                    "names": ["height", "width", "channels"],
                }
            self._ds = LeRobotDataset.create(
                repo_id,
                fps,
                features=features,
                root=root_path,
                robot_type="so101_follower",
            )

    # ── Introspection ────────────────────────────────────────────────

    @property
    def num_episodes(self) -> int:
        """Episodes saved to the dataset so far (buffered episode excluded)."""
        return self._ds.num_episodes

    @property
    def num_frames_in_episode(self) -> int:
        """Frames buffered in the currently open episode."""
        return self._frames_in_episode

    # ── Recording ────────────────────────────────────────────────────

    def start_episode(self) -> None:
        if self._episode_open:
            raise RuntimeError("episode already started; call end_episode() first")
        self._episode_open = True
        self._frames_in_episode = 0

    def _to_state(self, q, gripper) -> np.ndarray:
        deg = self._joint_map.to_real_deg(np.asarray(q, dtype=np.float64).reshape(5))
        grip = float(np.clip(gripper, 0.0, 1.0)) * 100.0
        return np.append(deg, grip).astype(np.float32)

    def add_frame(
        self,
        q,
        gripper,
        q_cmd,
        gripper_cmd,
        images: dict[str, np.ndarray],
        task: str | None = None,
    ) -> None:
        if not self._episode_open:
            raise RuntimeError("no episode in progress; call start_episode() first")
        if set(images) != set(self._camera_names):
            raise ValueError(f"expected images for {self._camera_names}, got {list(images)}")
        frame: dict = {
            "observation.state": self._to_state(q, gripper),
            "action": self._to_state(q_cmd, gripper_cmd),
            "task": self._default_task if task is None else task,
        }
        for name, img in images.items():
            frame[f"observation.images.{name}"] = np.ascontiguousarray(img)
        self._ds.add_frame(frame)
        self._frames_in_episode += 1

    def end_episode(self, save: bool = True) -> None:
        if not self._episode_open:
            raise RuntimeError("no episode in progress; call start_episode() first")
        if save and self._frames_in_episode > 0:
            self._ds.save_episode()
        else:
            self._ds.clear_episode_buffer()
        self._episode_open = False
        self._frames_in_episode = 0

    def finalize(self) -> Path:
        """Flush pending metadata and close writers. Returns the dataset root."""
        if self._episode_open:
            self.end_episode(save=False)
        self._ds.finalize()
        return Path(self._ds.root)
