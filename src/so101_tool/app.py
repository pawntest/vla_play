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

from .config import AppConfig
from .control.commands import Mode, SetMode
from .control.loop import ControlLoop
from .kinematics import Kinematics
from .scenario import Scenario, load_scenario
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
        self.scenario: Scenario | None = None
        self.backend = None
        self.loop: ControlLoop | None = None
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

    # -- construction / rebuild -------------------------------------------------

    def _make_backend(self, scenario: Scenario | None):
        if scenario is not None:
            from .robot.physics_sim import PhysicsBackend

            return PhysicsBackend(
                scenario, self.config.joint_map,
                self.config.render_width, self.config.render_height,
            )
        if self.config.backend == "real":
            from .robot.lerobot_backend import LeRobotBackend

            return LeRobotBackend(self.config)
        from .robot.sim import SimBackend

        return SimBackend(self.config.joint_map)

    def _build(self, scenario_path: str | None) -> None:
        from .viz.panel import ControlPanel  # circular-import guard

        scenario = load_scenario(scenario_path) if scenario_path else None
        backend = self._make_backend(scenario)
        backend.connect()
        loop = ControlLoop(backend, Kinematics(), self.config)
        loop.start()
        with self.lock:
            self.scenario = scenario
            self.backend = backend
            self.loop = loop
            if self.view is not None:
                self.view.remove()
            self.server.gui.reset()
            self.view = RobotView(
                self.server, backend.model if scenario is not None else Kinematics()
            )
            self._maybe_nl_agent()
            self.recorder = None
            self.panel = ControlPanel(self.server, self)
        self.scene_status = f"scene: {scenario.name}" if scenario else "scene: bare arm"

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
        if self.loop is not None:
            self.loop.stop()
            self.loop.join(timeout=2.0)
        if self.backend is not None:
            try:
                self.backend.disconnect()
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
                from .viz.panel import ControlPanel

                backend = self._make_backend(scenario)
                backend.connect()
                loop = ControlLoop(backend, Kinematics(), self.config)
                loop.start()
                with self.lock:
                    self.scenario = scenario
                    self.backend = backend
                    self.loop = loop
                    if self.view is not None:
                        self.view.remove()
                    self.server.gui.reset()
                    self.view = RobotView(self.server, backend.model)
                    self._maybe_nl_agent()
                    self.recorder = None
                    self.panel = ControlPanel(self.server, self)
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
                from .data.recorder import DatasetRecorder
                from .demo.scripted import PickParams, make_backend_and_pick

                self.demo_status = "building generation sim…"
                backend, pick = make_backend_and_pick(
                    scenario, self.config.joint_map,
                    self.config.render_width, self.config.render_height,
                    seed=seed, params=PickParams(object_name=scenario.objects[0].name),
                    sample_hz=fps,
                )
                root_path = root.strip() or None
                resume = bool(root_path and (Path(root_path) / "meta" / "info.json").exists())
                recorder = DatasetRecorder(
                    repo_id=repo_id.strip(), fps=fps,
                    cameras={n: (self.config.render_height, self.config.render_width)
                             for n in backend.camera_names},
                    joint_map=self.config.joint_map, root=root_path,
                    task=scenario.task, resume=resume,
                )

                def sample(state, q_cmd, g_cmd):
                    recorder.add_frame(q=state.q, gripper=state.gripper, q_cmd=q_cmd,
                                       gripper_cmd=g_cmd, images=backend.get_camera_frames())

                saved = attempts = 0
                while saved < episodes and attempts < episodes * 3:
                    attempts += 1
                    backend.reset(randomize=True)
                    recorder.start_episode()
                    ok = pick.run_episode(sample)
                    recorder.end_episode(save=ok)
                    saved += ok
                    self.demo_status = f"generating… {saved}/{episodes} (attempt {attempts})"
                out = recorder.finalize()
                backend.disconnect()
                self.demo_status = f"done: {saved} episodes → {out}"
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
