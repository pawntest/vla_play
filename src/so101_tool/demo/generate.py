"""Shared demonstration-dataset generation, used by the CLI, the GUI and the
Python API (`so101_tool.api.Session.generate_demos`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import JointMap
from ..scenario import Scenario
from .scripted import PickParams, make_backend_and_pick

ProgressCallback = Callable[[int, int, bool], None]  # (attempt, saved, success)


@dataclass
class DemoResult:
    saved: int
    attempts: int
    dataset_root: Path


def generate_demo_dataset(
    scenario: Scenario,
    dataset: str,
    episodes: int,
    *,
    root: str | Path | None = None,
    object_name: str | None = None,
    fps: int = 15,
    seed: int = 0,
    keep_failures: bool = False,
    resume: bool = False,
    render_width: int = 320,
    render_height: int = 240,
    joint_map: JointMap | None = None,
    max_attempts_factor: int = 3,
    pick_params: PickParams | None = None,
    on_progress: ProgressCallback | None = None,
) -> DemoResult:
    """Run the scripted expert until `episodes` successful demos are saved.

    Fully synchronous and deterministic (fake-clock physics): identical inputs
    produce byte-identical datasets regardless of machine speed.
    """
    from ..data.recorder import DatasetRecorder

    if not scenario.objects:
        raise ValueError("scenario has no objects to pick")
    target = object_name or scenario.objects[0].name
    joint_map = joint_map or JointMap()

    backend, pick = make_backend_and_pick(
        scenario, joint_map, render_width, render_height, seed=seed,
        params=pick_params or PickParams(object_name=target), sample_hz=fps,
    )
    recorder = DatasetRecorder(
        repo_id=dataset, fps=fps,
        cameras={n: (render_height, render_width) for n in backend.camera_names},
        joint_map=joint_map, root=root, task=scenario.task, resume=resume,
    )

    def sample(state, q_cmd, gripper_cmd):
        recorder.add_frame(q=state.q, gripper=state.gripper, q_cmd=q_cmd,
                           gripper_cmd=gripper_cmd, images=backend.get_camera_frames())

    saved = attempts = 0
    try:
        while saved < episodes and attempts < episodes * max_attempts_factor:
            attempts += 1
            backend.reset(randomize=True)
            recorder.start_episode()
            ok = pick.run_episode(sample)
            keep = ok or keep_failures
            recorder.end_episode(save=keep)
            saved += keep
            if on_progress is not None:
                on_progress(attempts, saved, ok)
    finally:
        dataset_root = recorder.finalize()
        backend.disconnect()
    return DemoResult(saved=saved, attempts=attempts, dataset_root=Path(dataset_root))
