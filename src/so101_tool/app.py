"""Application session: owns the viser server + all live components and lets the
GUI rebuild or drive them at runtime.

Everything heavy (scene rebuild, policy load, scripted demo generation,
training) runs in worker threads and reports progress through status strings
that the panel polls at render rate. `lock` guards component swaps: the render
loop grabs the current components under the lock each frame, so a rebuild in a
worker thread never races the renderer.
"""

from __future__ import annotations

import dataclasses
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import viser

from .api import Session
from .config import AppConfig
from .control.commands import Mode, SetMode
from .kinematics import Kinematics
from .scenario import Scenario
from .viz.recording import RecorderBridge
from .viz.robot_view import RobotView


class App:
    def __init__(self, server: viser.ViserServer, config: AppConfig,
                 scenario_path: str | None = None,
                 record_repo: str | None = None, record_root: str | None = None,
                 record_fps: int = 15):
        self.server = server
        self.config = config
        self.record_repo = record_repo  # panel initial values (--record flags)
        self.record_root = record_root
        self.record_fps = record_fps
        self.lock = threading.RLock()
        self.session: Session | None = None
        self.view: RobotView | None = None
        self.panel = None
        self.recorder: RecorderBridge | None = None
        self.nl_agent = None

        # worker-task status strings, polled by the panel
        self.scene_status = ""
        self.policy_status = "no policy loaded"
        self.demo_status = "idle"
        self.train_status = "idle"
        self._train_proc: subprocess.Popen | None = None
        self._busy = threading.Lock()  # one heavy worker at a time

        self._build(scenario_path)

    # -- component aliases (the panel reads these) --------------------------------

    @property
    def scenario(self) -> Scenario | None:
        return self.session.scenario if self.session else None

    @property
    def backend(self):
        return self.session.robot if self.session else None

    @property
    def loop(self):
        return self.session.loop if self.session else None

    # -- construction / rebuild -------------------------------------------------

    @classmethod
    def from_session(cls, server: viser.ViserServer, session: Session) -> "App":
        """Attach the GUI to an already-running api.Session."""
        app = cls.__new__(cls)
        app.server = server
        app.config = session.config
        app.record_repo = app.record_root = None
        app.record_fps = 15
        app.lock = threading.RLock()
        app.session = None
        app.view = None
        app.panel = None
        app.recorder = None
        app.nl_agent = None
        app.scene_status = ""
        app.policy_status = "no policy loaded"
        app.demo_status = "idle"
        app.train_status = "idle"
        app._train_proc = None
        app._busy = threading.Lock()
        app._attach(session)
        return app

    def _attach(self, session: Session) -> None:
        """Point the GUI at a (fresh) Session and rebuild view + panel."""
        from .viz.panel import ControlPanel  # circular-import guard

        with self.lock:
            self.session = session
            if self.panel is not None:
                self.panel.cleanup_scene()
            if self.view is not None:
                self.view.remove()
            self.server.gui.reset()
            self.view = RobotView(
                self.server,
                session.robot.model if session.scenario is not None else Kinematics(),
            )
            self._maybe_nl_agent()
            self.recorder = None
            self.panel = ControlPanel(self.server, self)
        name = session.scenario.name if session.scenario else "bare arm"
        self.scene_status = f"scene: {name}"

    def _build(self, scenario_path: str | None) -> None:
        self._attach(Session(backend=self.config.backend, scenario=scenario_path,
                             config=self.config))

    def _maybe_nl_agent(self) -> None:
        import os

        self.nl_agent = None
        if os.environ.get("ANTHROPIC_API_KEY") and not getattr(self.config, "no_nl", False):
            from .nl.agent import NLAgent

            self.nl_agent = NLAgent(self.loop, Kinematics(), model=self.config.nl_model)

    def _teardown(self) -> None:
        if self.recorder is not None:
            try:
                self.recorder.finalize()
            except Exception:
                pass
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass

    def rebuild(self, scenario_path: str | None) -> None:
        """Tear down and rebuild the whole session (worker thread)."""

        def worker():
            if not self._busy.acquire(blocking=False):
                self.scene_status = "busy — another task is running"
                return
            try:
                self.scene_status = "rebuilding scene…"
                self._teardown()
                self._build(scenario_path)
            except Exception as exc:
                self.scene_status = f"error: {exc}"
            finally:
                self._busy.release()

        threading.Thread(target=worker, daemon=True, name="so101-rebuild").start()

    def rebuild_with_scenario(self, scenario: Scenario) -> None:
        """Rebuild from an in-memory Scenario (GUI object editor)."""

        def worker():
            if not self._busy.acquire(blocking=False):
                self.scene_status = "busy — another task is running"
                return
            try:
                self.scene_status = "rebuilding scene…"
                self._teardown()
                self._attach(Session(scenario=scenario, config=self.config))
                self.scene_status = f"scene: {scenario.name} (edited)"
            except Exception as exc:
                self.scene_status = f"error: {exc}"
            finally:
                self._busy.release()

        threading.Thread(target=worker, daemon=True, name="so101-rebuild").start()

    def shutdown(self) -> None:
        self.stop_training()
        self._teardown()

    # -- recording -----------------------------------------------------------------

    def start_recording(self, repo_id: str, root: str, fps: int) -> None:
        if not hasattr(self.backend, "get_camera_frames"):
            self.scene_status = "recording needs cameras: load a scenario (or real robot)"
            return
        if self.recorder is None:
            backend = self.backend
            joint_map = self.config.joint_map
            task = self.scenario.task if self.scenario else ""

            def make():
                from .data.recorder import DatasetRecorder

                frames = backend.get_camera_frames()
                cameras = {name: img.shape[:2] for name, img in frames.items()}
                root_path = root.strip() or None
                resume = bool(root_path and (Path(root_path) / "meta" / "info.json").exists())
                return DatasetRecorder(
                    repo_id=repo_id.strip(), fps=fps, cameras=cameras,
                    joint_map=joint_map, root=root_path, task=task, resume=resume,
                )

            self.recorder = RecorderBridge(make, backend.get_camera_frames, fps)
        self.recorder.start()

    def close_dataset(self) -> None:
        """Finalize the current dataset so a new repo-id/root can be used."""
        if self.recorder is not None:
            try:
                self.recorder.finalize()
            except Exception:
                pass
            self.recorder = None

    # -- scripted demo generation -----------------------------------------------------

    def generate_demos(self, repo_id: str, root: str, episodes: int, fps: int,
                       seed: int) -> None:
        scenario = self.scenario
        if scenario is None or not scenario.objects:
            self.demo_status = "needs a scenario with at least one object"
            return

        def worker():
            if not self._busy.acquire(blocking=False):
                self.demo_status = "busy — another task is running"
                return
            try:
                from .demo.generate import generate_demo_dataset

                root_path = root.strip() or None
                resume = bool(root_path and (Path(root_path) / "meta" / "info.json").exists())

                def progress(attempt, saved, ok):
                    self.demo_status = f"generating… {saved}/{episodes} (attempt {attempt})"

                self.demo_status = "building generation sim…"
                result = generate_demo_dataset(
                    scenario, repo_id.strip(), episodes, root=root_path, fps=fps,
                    seed=seed, resume=resume, joint_map=self.config.joint_map,
                    render_width=self.config.render_width,
                    render_height=self.config.render_height, on_progress=progress,
                )
                self.demo_status = f"done: {result.saved} episodes → {result.dataset_root}"
            except Exception as exc:
                self.demo_status = f"error: {exc}"
            finally:
                self._busy.release()

        threading.Thread(target=worker, daemon=True, name="so101-demogen").start()

    # -- policy ---------------------------------------------------------------------

    def load_policy(self, path: str, task: str) -> None:
        path, task = path.strip(), task.strip()
        if not path:
            self.policy_status = "enter a checkpoint path or hub id"
            return

        def worker():
            if not self._busy.acquire(blocking=False):
                self.policy_status = "busy — another task is running"
                return
            try:
                from .policy.runner import PolicyRunner

                self.policy_status = "loading policy (torch import can take a while)…"
                cfg = dataclasses.replace(
                    self.config, policy_path=path, policy_task=task or None
                )
                runner = PolicyRunner(cfg, self.backend)
                runner.reset()  # load weights now, not inside a control tick
                self.loop.set_policy_runner(runner)
                self.policy_status = f"loaded: {path} — switching to POLICY mode"
                cmd = SetMode(mode=Mode.POLICY)
                self.loop.commands.put(cmd)
                cmd.wait(10.0)
                self.policy_status = (
                    f"RUNNING: {path}" if cmd.ok else f"load ok, mode switch failed: {cmd.error}"
                )
            except Exception as exc:
                self.policy_status = f"error: {exc}"
            finally:
                self._busy.release()

        threading.Thread(target=worker, daemon=True, name="so101-policy-load").start()

    def stop_policy(self) -> None:
        cmd = SetMode(mode=Mode.IDLE)
        self.loop.commands.put(cmd)
        self.policy_status = "stopped (mode: idle)"

    # -- training ----------------------------------------------------------------------

    def start_training(self, dataset: str, policy: str, steps: int, batch_size: int,
                       output_dir: str, device: str, extra: str) -> None:
        if self._train_proc is not None and self._train_proc.poll() is None:
            self.train_status = "a training run is already active"
            return

        def worker():
            try:
                from .training import TrainArgs, _find_train_exe, build_train_command

                args = TrainArgs(
                    dataset=dataset.strip(), policy=policy, steps=steps,
                    batch_size=batch_size, output_dir=output_dir.strip(),
                    device=device, extra=extra.strip(),
                )
                cmd = build_train_command(args)
                exe = _find_train_exe()
                if exe is None:
                    self.train_status = (
                        'error: lerobot-train not found — pip install "so101-tool[policy]" '
                        "(Python >= 3.12)"
                    )
                    return
                self.train_status = "starting…"
                proc = subprocess.Popen(
                    [exe, *cmd[1:]], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True,
                )
                self._train_proc = proc
                tail: deque[str] = deque(maxlen=6)
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        tail.append(line)
                        self.train_status = "```\n" + "\n".join(tail) + "\n```"
                rc = proc.wait()
                ckpt = Path(output_dir) / "checkpoints" / "last" / "pretrained_model"
                self.train_status = (
                    f"✅ finished — checkpoint: `{ckpt}`" if rc == 0
                    else f"❌ exited with code {rc}\n\n" + self.train_status
                )
            except Exception as exc:
                self.train_status = f"error: {exc}"
            finally:
                self._train_proc = None

        threading.Thread(target=worker, daemon=True, name="so101-train").start()

    def stop_training(self) -> None:
        proc = self._train_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            self.train_status = "terminating training run…"

    def emit_colab(self, dataset_hub_repo: str, policy: str, steps: int,
                   batch_size: int, out_path: str) -> None:
        try:
            from .training import TrainArgs, emit_colab

            args = TrainArgs(dataset=dataset_hub_repo.strip(), policy=policy,
                             steps=steps, batch_size=batch_size,
                             dataset_hub_repo=dataset_hub_repo.strip())
            path = emit_colab(args, out_path.strip(), dataset_hub_repo.strip())
            self.train_status = f"Colab notebook written: `{path}` — upload it to colab.research.google.com"
        except Exception as exc:
            self.train_status = f"error: {exc}"

    # -- render loop ------------------------------------------------------------------

    def render_tick(self) -> None:
        with self.lock:
            loop, view, panel, recorder = self.loop, self.view, self.panel, self.recorder
        if loop is None:
            return
        snap = loop.snapshot()
        if snap is None:
            return
        if snap.qpos_full is not None:
            view.sync_qpos(snap.qpos_full)
        else:
            view.sync(snap.q, snap.gripper)
        if panel is not None:
            panel.update(snap)
        if recorder is not None:
            recorder.tick(snap)


def run_app(app: App, render_hz: float) -> None:
    """Main-thread render loop until KeyboardInterrupt."""
    from loop_rate_limiters import RateLimiter

    rate = RateLimiter(frequency=render_hz, warn=False)
    try:
        while True:
            app.render_tick()
            rate.sleep()
    except KeyboardInterrupt:
        print("\nshutting down…")
    finally:
        app.shutdown()
        app.server.stop()
        time.sleep(0.2)
