"""Trained-policy inference (ACT / SmolVLA and other lerobot policies).

Works with ANY backend that provides camera frames: the physics simulation
(`--scenario`, rendered MuJoCo cameras) or the real robot (lerobot cameras).
Loaded lazily: torch/lerobot are imported on first reset().

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
        self._features = None

    def _load(self) -> None:
        if not hasattr(self._backend, "get_camera_frames"):
            raise RuntimeError(
                "POLICY mode needs camera observations: run with --scenario "
                "(physics sim with rendered cameras) or --backend real."
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
        self._preprocess.reset()
        self._postprocess.reset()

    def _build_observation(self, state: RobotState) -> dict:
        deg = self._config.joint_map.to_real_deg(state.q)
        obs = {f"{j}.pos": float(deg[i]) for i, j in enumerate(ARM_JOINTS)}
        obs["gripper.pos"] = float(np.clip(state.gripper, 0.0, 1.0) * 100.0)
        obs.update(self._backend.get_camera_frames())
        return obs

    def _features_for(self, obs: dict) -> dict:
        """LeRobot dataset-feature spec matching our observation/action dicts
        (the same shape the recorder writes, so checkpoints line up)."""
        from lerobot.utils.feature_utils import hw_to_dataset_features

        hw_obs = {
            k: (v.shape if isinstance(v, np.ndarray) else float) for k, v in obs.items()
        }
        hw_act = {f"{j}.pos": float for j in ARM_JOINTS}
        hw_act["gripper.pos"] = float
        feats = hw_to_dataset_features(hw_obs, "observation", use_video=True)
        feats.update(hw_to_dataset_features(hw_act, "action", use_video=True))
        return feats

    def step(self, state: RobotState) -> tuple[np.ndarray, float] | None:
        """One inference tick -> (q_target rad, gripper fraction), or None to hold."""
        if self._policy is None:
            raise RuntimeError("PolicyRunner.reset() was not called")
        import torch  # already imported transitively by lerobot
        from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
        from lerobot.utils.feature_utils import build_dataset_frame

        obs = self._build_observation(state)
        if self._features is None:
            self._features = self._features_for(obs)
        frame = build_dataset_frame(self._features, obs, prefix="observation")
        with torch.inference_mode():
            observation = prepare_observation_for_inference(
                frame, torch.device("cpu"), self._config.policy_task, "so101_follower"
            )
            observation = self._preprocess(observation)
            action = self._policy.select_action(observation)
            action = self._postprocess(action)
        action_dict = make_robot_action(action.squeeze(0).cpu(), self._features)
        # action_dict: {"<motor>.pos": degrees, "gripper.pos": 0..100}
        deg = np.array([float(action_dict[f"{j}.pos"]) for j in ARM_JOINTS])
        q = self._config.joint_map.from_real_deg(deg)
        gripper = float(np.clip(float(action_dict["gripper.pos"]) / 100.0, 0.0, 1.0))
        return q, gripper
