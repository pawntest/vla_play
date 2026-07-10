"""Bridge between the render loop / GUI and a DatasetRecorder.

The recorder itself is created lazily (lerobot import happens on the first
Start click / first episode), so the viewer still works without lerobot and
errors surface as status text instead of crashes. tick() is called from the
render loop and downsamples to the dataset fps.
"""

from __future__ import annotations

import time
from typing import Callable

from ..control.loop import LoopSnapshot


class RecorderBridge:
    def __init__(self, make_recorder: Callable, frames_getter: Callable[[], dict], fps: float):
        self._make = make_recorder
        self._frames_getter = frames_getter
        self._fps = fps
        self._recorder = None
        self._next_sample = 0.0
        self.recording = False
        self.status = "idle"

    @property
    def episodes(self) -> int:
        return 0 if self._recorder is None else self._recorder.num_episodes

    def start(self) -> None:
        if self.recording:
            return
        try:
            if self._recorder is None:
                self._recorder = self._make()
            self._recorder.start_episode()
        except Exception as exc:
            self.status = f"error: {exc}"
            return
        self.recording = True
        self._next_sample = 0.0
        self.status = "recording…"

    def stop(self, save: bool = True) -> None:
        if not self.recording:
            return
        self.recording = False
        try:
            self._recorder.end_episode(save=save)
            self.status = f"{'saved' if save else 'discarded'} (episodes: {self.episodes})"
        except Exception as exc:
            self.status = f"error: {exc}"

    def tick(self, snap: LoopSnapshot | None) -> None:
        if not self.recording or snap is None:
            return
        now = time.monotonic()
        if now < self._next_sample:
            return
        self._next_sample = now + 1.0 / self._fps
        try:
            self._recorder.add_frame(
                q=snap.q,
                gripper=snap.gripper,
                q_cmd=snap.q_cmd if snap.q_cmd is not None else snap.q,
                gripper_cmd=snap.gripper_cmd if snap.gripper_cmd is not None else snap.gripper,
                images=self._frames_getter(),
            )
        except Exception as exc:
            self.recording = False
            self.status = f"error: {exc}"

    def finalize(self) -> None:
        if self._recorder is not None:
            if self.recording:
                self.stop(save=True)
            self._recorder.finalize()
