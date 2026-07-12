"""viser GUI panel: every so101_tool capability, driven through an App session.

Folders: Scenario (pick/edit/save scenes), Status + e-stop, Joints, Cartesian,
Record dataset, Scripted demos, Policy, Train, Natural language. All callbacks
either enqueue Commands on the control loop or call App methods that spawn
worker threads — nothing here blocks a viser callback.

update(snap) is called from the render loop and refreshes readouts and the
status strings of background tasks.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import viser

from ..config import ARM_JOINTS, ARM_LIMITS_HI, ARM_LIMITS_LO
from ..control.commands import Home, JogTool, Mode, MoveJ, MoveL, SetMode, Stop
from ..control.loop import LoopSnapshot
from ..scenario import ObjectSpec, Scenario, save_scenario

_MODES = [m.value for m in Mode]
_SCENARIO_DIRS = ("examples", "scenarios")


def _find_scenarios() -> list[str]:
    found = []
    for d in _SCENARIO_DIRS:
        found += sorted(str(p) for p in Path(d).glob("*.yaml"))
    return found


class ControlPanel:
    def __init__(self, server: viser.ViserServer, app):
        self._app = app
        self._loop = app.loop
        self._config = app.config
        self._tick = 0
        self._nl_busy = False
        gui = server.gui
        scenario = app.scenario

        # ---- Scenario: load / edit / save ------------------------------------
        with gui.add_folder("Scenario", expand_by_default=scenario is None):
            title = scenario.name if scenario else "bare arm (no physics scene)"
            self._scene_md = gui.add_markdown(f"**{title}**\n\ntask: *{scenario.task if scenario else '-'}*")
            options = ["(bare arm)"] + _find_scenarios()
            current = scenario.source_path if scenario and scenario.source_path else "(bare arm)"
            if current not in options:
                options.append(current)
            self._scene_dd = gui.add_dropdown("file", options=options, initial_value=current)
            load_btn = gui.add_button("Load scenario")
            load_btn.on_click(lambda _: self._load_selected_scenario())
            if scenario is not None:
                reset_btn = gui.add_button("Reset scene (randomize)")
                reset_btn.on_click(lambda _: app.backend.reset(randomize=True))
                self._task_input = gui.add_text("task", initial_value=scenario.task)

                with gui.add_folder("Edit objects", expand_by_default=False):
                    names = [o.name for o in scenario.objects] or ["-"]
                    self._obj_dd = gui.add_dropdown("object", options=names,
                                                    initial_value=names[0])
                    rm_btn = gui.add_button("Remove selected")
                    rm_btn.on_click(lambda _: self._remove_object())
                    self._new_name = gui.add_text("new name", initial_value="obj1")
                    self._new_type = gui.add_dropdown(
                        "type", options=["box", "sphere", "cylinder"], initial_value="box"
                    )
                    self._new_size = gui.add_vector3(
                        "size (m)", initial_value=(0.015, 0.015, 0.015), step=0.005
                    )
                    self._new_pos = gui.add_vector3(
                        "position (m)", initial_value=(0.25, 0.0, 0.02), step=0.01
                    )
                    self._new_color = gui.add_rgb("color", initial_value=(220, 30, 30))
                    self._new_mass = gui.add_number("mass (kg)", initial_value=0.03,
                                                    min=0.001, max=2.0, step=0.01)
                    self._new_noise = gui.add_slider(
                        "pos noise ± (cm)", min=0.0, max=10.0, step=0.5, initial_value=4.0
                    )
                    add_btn = gui.add_button("Add object → rebuild scene")
                    add_btn.on_click(lambda _: self._add_object())
                    self._save_path = gui.add_text(
                        "save as", initial_value=scenario.source_path or "scenarios/my_scene.yaml"
                    )
                    save_btn = gui.add_button("Save scenario YAML")
                    save_btn.on_click(lambda _: self._save_scenario())

        # ---- Status / e-stop ---------------------------------------------------
        with gui.add_folder("Status"):
            self._mode_dd = gui.add_dropdown("Mode", options=_MODES, initial_value=Mode.IDLE.value)
            self._status_md = gui.add_markdown("connecting…")
            estop_btn = gui.add_button("EMERGENCY STOP", color="red")
            reset_btn = gui.add_button("Reset e-stop")
            stop_btn = gui.add_button("Stop motion")
        self._mode_dd.on_update(lambda _: self._put(SetMode(mode=Mode(self._mode_dd.value))))
        estop_btn.on_click(lambda _: self._loop.estop())
        reset_btn.on_click(lambda _: self._loop.reset_estop())
        stop_btn.on_click(lambda _: self._put(Stop()))

        # ---- Joints ---------------------------------------------------------------
        with gui.add_folder("Joints (RULE)", expand_by_default=False):
            self._speed = gui.add_slider("speed", min=0.1, max=1.0, step=0.05, initial_value=0.5)
            self._joint_sliders = [
                gui.add_slider(name, min=float(ARM_LIMITS_LO[i]), max=float(ARM_LIMITS_HI[i]),
                               step=0.01, initial_value=0.0)
                for i, name in enumerate(ARM_JOINTS)
            ]
            self._gripper_slider = gui.add_slider("gripper", min=0.0, max=1.0, step=0.01,
                                                  initial_value=0.0)
            movej_btn = gui.add_button("Move to sliders")
            home_btn = gui.add_button("Home")
        movej_btn.on_click(
            lambda _: self._put(
                MoveJ(q=np.array([s.value for s in self._joint_sliders]),
                      gripper=self._gripper_slider.value, speed=self._speed.value)
            )
        )
        home_btn.on_click(lambda _: self._put(Home(speed=self._speed.value)))

        # ---- Cartesian ---------------------------------------------------------------
        with gui.add_folder("Cartesian (RULE)", expand_by_default=False):
            self._gizmo = server.scene.add_transform_controls("/target", scale=0.15)
            snap_btn = gui.add_button("Snap target to TCP")
            go_pos_btn = gui.add_button("Go to target (position)")
            go_pose_btn = gui.add_button("Go to target (pose)")
            self._jog_step = gui.add_slider("jog step (cm)", min=0.5, max=10.0, step=0.5,
                                            initial_value=2.0)
            jog_btns = {}
            for axis in "XYZ":
                jog_btns[f"+{axis}"] = gui.add_button(f"Jog +{axis} (tool)")
                jog_btns[f"-{axis}"] = gui.add_button(f"Jog -{axis} (tool)")
        snap_btn.on_click(lambda _: self._snap_gizmo_to_tcp())
        go_pos_btn.on_click(
            lambda _: self._put(MoveL(position=np.array(self._gizmo.position), wxyz=None,
                                      speed=self._speed.value))
        )
        go_pose_btn.on_click(
            lambda _: self._put(MoveL(position=np.array(self._gizmo.position),
                                      wxyz=np.array(self._gizmo.wxyz), speed=self._speed.value))
        )
        for label, btn in jog_btns.items():
            sign = 1.0 if label[0] == "+" else -1.0
            axis = "XYZ".index(label[1])

            def _jog(_evt, sign=sign, axis=axis):
                dpos = np.zeros(3)
                dpos[axis] = sign * self._jog_step.value / 100.0
                self._put(JogTool(dpos=dpos, frame="tool", speed=self._speed.value))

            btn.on_click(_jog)

        # ---- Record dataset -------------------------------------------------------------
        has_cameras = hasattr(app.backend, "get_camera_frames")
        with gui.add_folder("Record dataset", expand_by_default=False):
            if not has_cameras:
                gui.add_markdown("*needs cameras — load a scenario (or real robot)*")
            default_repo = app.record_repo or (scenario.name if scenario else "my_dataset")
            default_root = app.record_root or f"data/{default_repo}"
            self._rec_repo = gui.add_text("repo id", initial_value=default_repo)
            self._rec_root = gui.add_text("local dir", initial_value=default_root)
            self._rec_fps = gui.add_slider("fps", min=5, max=30, step=5,
                                           initial_value=app.record_fps)
            self._rec_status = gui.add_markdown("idle")
            rec_start = gui.add_button("● Start episode", color="red", disabled=not has_cameras)
            rec_save = gui.add_button("■ Stop & save", disabled=not has_cameras)
            rec_discard = gui.add_button("✕ Stop & discard", disabled=not has_cameras)
            rec_close = gui.add_button("Close dataset (finalize)", disabled=not has_cameras)
        rec_start.on_click(
            lambda _: app.start_recording(self._rec_repo.value, self._rec_root.value,
                                          int(self._rec_fps.value))
        )
        rec_save.on_click(lambda _: app.recorder and app.recorder.stop(save=True))
        rec_discard.on_click(lambda _: app.recorder and app.recorder.stop(save=False))
        rec_close.on_click(lambda _: app.close_dataset())

        # ---- Scripted demos -----------------------------------------------------------
        with gui.add_folder("Scripted demos (auto-generate)", expand_by_default=False):
            if scenario is None or not scenario.objects:
                gui.add_markdown("*needs a scenario with an object*")
            self._demo_eps = gui.add_number("episodes", initial_value=20, min=1, max=500, step=1)
            self._demo_seed = gui.add_number("seed", initial_value=0, min=0, max=9999, step=1)
            self._demo_status = gui.add_markdown("idle")
            demo_btn = gui.add_button(
                "Generate pick demos", disabled=scenario is None or not scenario.objects
            )
        demo_btn.on_click(
            lambda _: app.generate_demos(
                self._rec_repo.value, self._rec_root.value,
                int(self._demo_eps.value), int(self._rec_fps.value), int(self._demo_seed.value),
            )
        )

        # ---- Policy ---------------------------------------------------------------------
        with gui.add_folder("Policy (AI)", expand_by_default=False):
            self._policy_path = gui.add_text(
                "checkpoint", initial_value=self._config.policy_path or
                "outputs/train/checkpoints/last/pretrained_model"
            )
            self._policy_task = gui.add_text(
                "task", initial_value=self._config.policy_task or (scenario.task if scenario else "")
            )
            self._policy_status = gui.add_markdown(app.policy_status)
            pol_load = gui.add_button("Load & run policy")
            pol_stop = gui.add_button("Stop policy (→ idle)")
        pol_load.on_click(
            lambda _: app.load_policy(self._policy_path.value, self._policy_task.value)
        )
        pol_stop.on_click(lambda _: app.stop_policy())

        # ---- Train -----------------------------------------------------------------------
        with gui.add_folder("Train", expand_by_default=False):
            self._train_dataset = gui.add_text("dataset (dir or hub id)",
                                               initial_value=f"data/{default_repo}")
            self._train_policy = gui.add_dropdown("policy", options=["act", "smolvla", "diffusion"],
                                                  initial_value="act")
            self._train_steps = gui.add_number("steps", initial_value=20000, min=1,
                                               max=1_000_000, step=1000)
            self._train_batch = gui.add_number("batch size", initial_value=8, min=1, max=256, step=1)
            self._train_device = gui.add_dropdown("device", options=["auto", "cuda", "mps", "cpu"],
                                                  initial_value="auto")
            self._train_out = gui.add_text("output dir", initial_value="outputs/train")
            self._train_extra = gui.add_text("extra args", initial_value="")
            self._train_status = gui.add_markdown("idle")
            train_btn = gui.add_button("Start training", color="green")
            train_stop = gui.add_button("Stop training")
            self._colab_repo = gui.add_text("HF dataset repo (for Colab)", initial_value="")
            colab_btn = gui.add_button("Export Colab notebook (free GPU)")
        train_btn.on_click(
            lambda _: app.start_training(
                self._train_dataset.value, self._train_policy.value,
                int(self._train_steps.value), int(self._train_batch.value),
                self._train_out.value, self._train_device.value, self._train_extra.value,
            )
        )
        train_stop.on_click(lambda _: app.stop_training())
        colab_btn.on_click(
            lambda _: app.emit_colab(
                self._colab_repo.value, self._train_policy.value,
                int(self._train_steps.value), int(self._train_batch.value), "train_colab.ipynb",
            )
        )

        # ---- Natural language -----------------------------------------------------------
        if app.nl_agent is not None:
            with gui.add_folder("Natural language", expand_by_default=False):
                self._nl_input = gui.add_text("command", initial_value="")
                self._nl_send = gui.add_button("Send")
                self._nl_log = gui.add_markdown("")
                self._nl_send.on_click(lambda _: self._send_nl())

    # -- helpers ------------------------------------------------------------------

    def _put(self, cmd) -> None:
        self._loop.commands.put(cmd)

    def _snap_gizmo_to_tcp(self) -> None:
        snap = self._loop.snapshot()
        if snap is not None:
            self._gizmo.position = snap.tcp_position
            self._gizmo.wxyz = snap.tcp_wxyz

    def _load_selected_scenario(self) -> None:
        sel = self._scene_dd.value
        self._app.rebuild(None if sel == "(bare arm)" else sel)

    def _edited_scenario(self) -> Scenario:
        sc = self._app.scenario
        sc.task = self._task_input.value
        return sc

    def _add_object(self) -> None:
        sc = self._edited_scenario()
        rgb = [c / 255.0 for c in self._new_color.value]
        noise_m = self._new_noise.value / 100.0
        sc.objects.append(
            ObjectSpec(
                name=self._new_name.value.strip() or f"obj{len(sc.objects) + 1}",
                type=self._new_type.value,
                size=list(self._new_size.value),
                pos=list(self._new_pos.value),
                rgba=[*rgb, 1.0],
                mass=float(self._new_mass.value),
                pos_noise=[noise_m, noise_m, 0.0] if noise_m > 0 else None,
            )
        )
        self._app.rebuild_with_scenario(sc)

    def _remove_object(self) -> None:
        sc = self._edited_scenario()
        name = self._obj_dd.value
        sc.objects = [o for o in sc.objects if o.name != name]
        self._app.rebuild_with_scenario(sc)

    def _save_scenario(self) -> None:
        try:
            path = save_scenario(self._edited_scenario(), self._save_path.value)
            self._app.scene_status = f"saved: {path}"
        except Exception as exc:
            self._app.scene_status = f"save error: {exc}"

    def _send_nl(self) -> None:
        text = self._nl_input.value.strip()
        if not text or self._nl_busy:
            return
        self._nl_busy = True
        self._nl_send.disabled = True
        self._nl_input.value = ""
        self._nl_log.content += f"\n\n**you:** {text}"

        def worker():
            try:
                reply = self._app.nl_agent.handle(text)
            except Exception as exc:  # never crash the GUI thread
                reply = f"error: {exc}"
            self._nl_log.content += f"\n\n**robot:** {reply}"
            self._nl_busy = False
            self._nl_send.disabled = False

        threading.Thread(target=worker, daemon=True, name="so101-nl").start()

    # -- render-loop hook --------------------------------------------------------------

    def update(self, snap: LoopSnapshot | None) -> None:
        self._tick += 1
        if snap is None or self._tick % 6:  # ~5 Hz text updates at 30 Hz render
            return
        app = self._app
        if app.recorder is not None:
            marker = "🔴 " if app.recorder.recording else ""
            content = f"{marker}{app.recorder.status} — episodes saved: {app.recorder.episodes}"
        else:
            content = "idle"
        if self._rec_status.content != content:
            self._rec_status.content = content
        for handle, text in (
            (self._demo_status, app.demo_status),
            (self._policy_status, app.policy_status),
            (self._train_status, app.train_status),
        ):
            if handle.content != text:
                handle.content = text

        if self._mode_dd.value != snap.mode.value:
            self._mode_dd.value = snap.mode.value
        joints = " ".join(f"{v:+.2f}" for v in snap.q)
        tcp = np.round(snap.tcp_position * 1000).astype(int)
        lines = [
            f"backend: **{'real' if snap.backend_is_real else 'sim'}** — {app.scene_status}",
            f"q (rad): `{joints}`  grip: `{snap.gripper:.2f}`",
            f"TCP (mm): `{tcp[0]} {tcp[1]} {tcp[2]}`",
            f"active: `{snap.active_command or '-'}`",
        ]
        if snap.estop:
            lines.insert(0, "## 🛑 E-STOP LATCHED")
        if snap.error:
            lines.append(f"last error: `{snap.error}`")
        status = "\n\n".join(lines)
        if self._status_md.content != status:
            self._status_md.content = status
        if snap.mode in (Mode.IDLE, Mode.MIRROR):
            for s, v in zip(self._joint_sliders, snap.q):
                s.value = float(np.clip(v, s.min, s.max))
            self._gripper_slider.value = float(snap.gripper)
