"""Trained-policy inference (ACT / SmolVLA and other lerobot policies).

Loaded lazily: torch/lerobot are imported on first reset(). Policies need
camera observations, so this only works with the real backend (LeRobotBackend)
— the sim backend has no cameras, which is a documented limitation.

The control loop calls step(state) each tick and routes the returned targets
through the safety filter like any other motion source.
"""

from __future__ import annotations

import numpy as np

from ..config import ARM_JOINTS, AppConfig
from ..robot.base import RobotInterface, RobotState

_INSTALL_HINT = 'policy inference requires: pip install "so101-tool[policy]" (Python >= 3.12)'


class PolicyRunner:
    def __init__(self, config: AppConfig, backend: RobotInterface):
        self._config = config
        self._backend = backend
        self._policy = None
        self._preprocess = None
        self._postprocess = None

    def _load(self) -> None:
        if not self._backend.is_real:
            raise RuntimeError(
                "POLICY mode needs camera observations and therefore the real "
                "backend (--backend real). The sim backend has no cameras."
            )
        path = self._config.policy_path
        if not path:
            raise RuntimeError("no policy checkpoint configured (--policy-path)")
        try:
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
            from lerobot.policies.pretrained import PreTrainedConfig
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc

        cfg = PreTrainedConfig.from_pretrained(path)
        policy_cls = get_policy_class(cfg.type)
        self._policy = policy_cls.from_pretrained(path)
        self._policy.eval()
        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config, path
        )

    def reset(self) -> None:
        if self._policy is None:
            self._load()
        self._policy.reset()

    def step(self, state: RobotState) -> tuple[np.ndarray, float] | None:
        """One inference tick -> (q_target rad, gripper fraction), or None to hold."""
        if self._policy is None:
            raise RuntimeError("PolicyRunner.reset() was not called")
        obs = getattr(self._backend, "last_observation", None)
        if obs is None:
            return None
        import torch  # already imported transitively by lerobot

        try:
            from lerobot.utils.control_utils import build_inference_frame

            frame = build_inference_frame(
                observation=obs, task=self._config.policy_task, robot_type="so101_follower"
            )
        except ImportError:
            # Older/newer lerobot layouts: fall back to passing the raw
            # observation dict (+ task) straight into the preprocessor.
            frame = dict(obs)
            if self._config.policy_task is not None:
                frame["task"] = self._config.policy_task
        batch = self._preprocess(frame)
        with torch.inference_mode():
            action = self._policy.select_action(batch)
        action = self._postprocess(action)
        # action: {"<motor>.pos": degrees, "gripper.pos": 0..100}
        deg = np.array([float(action[f"{j}.pos"]) for j in ARM_JOINTS])
        q = self._config.joint_map.from_real_deg(deg)
        gripper = float(np.clip(float(action["gripper.pos"]) / 100.0, 0.0, 1.0))
        return q, gripper
