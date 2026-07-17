"""viser GUI panel — redesigned for intuitive use.

Layout:
  - Always-visible header: a large colored STATE BANNER (what the robot is
    doing right now), EMERGENCY STOP, a Reset button that only appears while
    the e-stop is latched, and a pause button.
  - Four tabs replace the old wall of folders:
      🕹 操作   manual control (joints, cartesian gizmo/jog, direct drag, NL)
      🎬 データ  dataset recording, scripted demos, remote teleop
      🧠 学習   policy execution and training
      🌍 シーン  scenario load/edit (objects, cameras, environment), save

Modes switch AUTOMATICALLY: any manual-control action drops the loop into
RULE mode first (`_put_motion`), running a policy switches to POLICY,
enabling teleop switches to TELEOP — the user never manages modes by hand.
All callbacks only enqueue Commands or call App worker methods; nothing here
blocks a viser callback. `update(snap)` is called from the render loop.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import viser

from ..config import ARM_JOINTS, ARM_LIMITS_HI, ARM_LIMITS_LO
from ..control.commands import Home, JogTool, Mode, MoveJ, MoveL, SetMode, Stop
from ..control.loop import LoopSnapshot
from ..scenario import (
    ATTACHABLE_BODIES,
    CameraSpec,
    FloorSpec,
    ObjectSpec,
    Scenario,
    TableSpec,
    save_scenario,
)

_SCENARIO_DIRS = ("examples", "scenarios")

# banner color / title / subtitle per mode
_MODE_BANNER = {
    Mode.IDLE: ("#64748b", "⬜ 待機中",
                "スライダー・ギズモ・ドラッグ操作で自動的に「手動操作」へ切り替わります"),
    Mode.RULE: ("#2563eb", "🕹 手動操作中", ""),
    Mode.POLICY: ("#7c3aed", "🤖 AI ポリシー実行中", ""),
    Mode.TELEOP: ("#059669", "🎮 テレオペ中", ""),
    Mode.MIRROR: ("#d97706", "🪞 実機ミラー中", "実機を手で動かすと3Dが追従します(書き込みなし)"),
}
_CMD_JP = {"MoveJ": "関節移動", "MoveL": "直線移動(IK)", "JogTool": "ジョグ",
           "SetGripper": "グリッパー", "Home": "ホーム復帰"}


def _find_scenarios() -> list[str]:
    found = []
    for d in _SCENARIO_DIRS:
        found += sorted(str(p) for p in Path(d).glob("*.yaml"))
    return found


def _banner(color: str, title: str, sub: str, badge: str = "") -> str:
    badge_html = (
        f'<span style="background:#dc2626;border-radius:999px;padding:1px 8px;'
        f'font-size:0.72em;margin-left:8px;vertical-align:middle;">{badge}</span>'
        if badge else ""
    )
    return (
        f'<div style="border-radius:8px;padding:10px 12px;background:{color};'
        f'color:#fff;line-height:1.4;">'
        f'<div style="font-size:1.12em;font-weight:700;">{title}{badge_html}</div>'
        f'<div style="font-size:0.82em;opacity:0.93;white-space:pre-line;">{sub}</div>'
        f"</div>"
    )


class ControlPanel:
    def __init__(self, server: viser.ViserServer, app):
        self._app = app
        self._server = server
        self._loop = app.loop
        self._config = app.config
        self._tick = 0
        self._nl_busy = False
        self._drag_gizmos: list = []
        self._click_add_registered = False
        gui = server.gui
        scenario = app.scenario

        gui.configure_theme(control_layout="collapsible", control_width="large",
                            brand_color=(37, 99, 235))

        # =====================  header (always visible)  =====================
        self._banner_html = gui.add_html(_banner("#64748b", "⬜ 接続中…", ""))
        estop_btn = gui.add_button("🛑 非常停止 (EMERGENCY STOP)", color="red")
        self._reset_estop_btn = gui.add_button("非常停止を解除する", color="orange",
                                               visible=False)
        stop_btn = gui.add_button("⏸ 停止して待機に戻る")
        estop_btn.on_click(lambda _: self._loop.estop())
        self._reset_estop_btn.on_click(lambda _: self._loop.reset_estop())
        stop_btn.on_click(lambda _: (self._put(Stop()),
                                     self._put(SetMode(mode=Mode.IDLE))))
        if app.backend is not None and app.backend.is_real:
            mirror_btn = gui.add_button("🪞 ミラーモード(実機を手で動かす)")
            mirror_btn.on_click(lambda _: self._put(SetMode(mode=Mode.MIRROR)))

        # =============================  tabs  ================================
        # (handles kept for cleanup_gui: viser's gui.reset() crashes on tab
        #  groups — the group must be dismantled tab-first, group-second)
        self._tab_group = gui.add_tab_group()
        self._tabs = []

        tab = self._tab_group.add_tab("🕹 操作")
        self._tabs.append(tab)
        with tab:
            self._build_control_tab(gui, server)

        tab = self._tab_group.add_tab("🎬 データ")
        self._tabs.append(tab)
        with tab:
            self._build_data_tab(gui, app, scenario)

        tab = self._tab_group.add_tab("🧠 学習")
        self._tabs.append(tab)
        with tab:
            self._build_ai_tab(gui, app, scenario)

        tab = self._tab_group.add_tab("🌍 シーン")
        self._tabs.append(tab)
        with tab:
            self._build_scene_tab(gui, app, scenario)

    # ------------------------------------------------------------------ tabs --

    def _build_control_tab(self, gui, server) -> None:
        gui.add_markdown("*どの操作も自動で「手動操作」モードに切り替わります*")
        self._speed = gui.add_slider("速度", min=0.1, max=1.0, step=0.05, initial_value=0.5)

        with gui.add_folder("関節 (スライダー)", expand_by_default=True):
            self._joint_sliders = [
                gui.add_slider(name, min=float(ARM_LIMITS_LO[i]), max=float(ARM_LIMITS_HI[i]),
                               step=0.01, initial_value=0.0)
                for i, name in enumerate(ARM_JOINTS)
            ]
            self._gripper_slider = gui.add_slider("グリッパー (0閉↔1開)", min=0.0, max=1.0,
                                                  step=0.01, initial_value=0.0)
            movej_btn = gui.add_button("▶ この姿勢へ移動")
            home_btn = gui.add_button("⌂ ホームへ戻る")
        movej_btn.on_click(
            lambda _: self._put_motion(
                MoveJ(q=np.array([s.value for s in self._joint_sliders]),
                      gripper=self._gripper_slider.value, speed=self._speed.value)
            )
        )
        home_btn.on_click(lambda _: self._put_motion(Home(speed=self._speed.value)))

        with gui.add_folder("先端位置 (ギズモ / ジョグ)", expand_by_default=True):
            self._direct_drag_cb = gui.add_checkbox(
                "🖐 3Dモデルを直接ドラッグで操作", initial_value=True,
                hint="物体やグリッパーを掴んでそのまま動かせます",
            )
            self._direct_drag_cb.on_update(lambda _: self._set_direct_drag())
            self._gizmo = server.scene.add_transform_controls("/target", scale=0.15)
            snap_btn = gui.add_button("◎ ターゲットを現在位置に合わせる")
            go_pos_btn = gui.add_button("▶ ターゲットへ移動(位置)")
            go_pose_btn = gui.add_button("▶ ターゲットへ移動(姿勢つき)")
            self._jog_step = gui.add_slider("ジョグ幅 (cm)", min=0.5, max=10.0, step=0.5,
                                            initial_value=2.0)
            jog_btns = {}
            for axis in "XYZ":
                jog_btns[f"+{axis}"] = gui.add_button(f"{axis}+ へジョグ")
                jog_btns[f"-{axis}"] = gui.add_button(f"{axis}− へジョグ")
        snap_btn.on_click(lambda _: self._snap_gizmo_to_tcp())
        go_pos_btn.on_click(
            lambda _: self._put_motion(MoveL(position=np.array(self._gizmo.position),
                                             wxyz=None, speed=self._speed.value))
        )
        go_pose_btn.on_click(
            lambda _: self._put_motion(MoveL(position=np.array(self._gizmo.position),
                                             wxyz=np.array(self._gizmo.wxyz),
                                             speed=self._speed.value))
        )
        for label, btn in jog_btns.items():
            sign = 1.0 if label[0] == "+" else -1.0
            axis = "XYZ".index(label[1])

            def _jog(_evt, sign=sign, axis=axis):
                dpos = np.zeros(3)
                dpos[axis] = sign * self._jog_step.value / 100.0
                self._put_motion(JogTool(dpos=dpos, frame="tool", speed=self._speed.value))

            btn.on_click(_jog)

        if self._app.nl_agent is not None:
            with gui.add_folder("💬 自然言語で操作", expand_by_default=True):
                self._nl_input = gui.add_text("指示", initial_value="",
                                              hint="例: グリッパーを5cm前に動かして")
                self._nl_send = gui.add_button("送信")
                self._nl_log = gui.add_markdown("")
                self._nl_send.on_click(lambda _: self._send_nl())

    def _build_data_tab(self, gui, app, scenario) -> None:
        has_cameras = hasattr(app.backend, "get_camera_frames")
        default_repo = app.record_repo or (scenario.name if scenario else "my_dataset")
        default_root = app.record_root or f"data/{default_repo}"

        with gui.add_folder("🔴 デモ録画 (LeRobotDataset)", expand_by_default=True):
            if not has_cameras:
                gui.add_markdown("*カメラが必要です — シーンを読み込むか実機に接続してください*")
            self._rec_repo = gui.add_text("データセット名", initial_value=default_repo)
            self._rec_root = gui.add_text("保存先フォルダ", initial_value=default_root)
            self._rec_fps = gui.add_slider("fps", min=5, max=30, step=5,
                                           initial_value=app.record_fps)
            self._rec_status = gui.add_markdown("待機中")
            rec_start = gui.add_button("● エピソード開始", color="red", disabled=not has_cameras)
            rec_save = gui.add_button("■ 停止して保存", disabled=not has_cameras)
            rec_discard = gui.add_button("✕ 停止して破棄", disabled=not has_cameras)
            rec_close = gui.add_button("データセットを閉じる(確定)", disabled=not has_cameras)
        rec_start.on_click(
            lambda _: app.start_recording(self._rec_repo.value, self._rec_root.value,
                                          int(self._rec_fps.value))
        )
        rec_save.on_click(lambda _: app.recorder and app.recorder.stop(save=True))
        rec_discard.on_click(lambda _: app.recorder and app.recorder.stop(save=False))
        rec_close.on_click(lambda _: app.close_dataset())

        with gui.add_folder("⚙ 自動デモ生成(スクリプト)", expand_by_default=False):
            if scenario is None or not scenario.objects:
                gui.add_markdown("*物体のあるシーンが必要です*")
            self._demo_eps = gui.add_number("エピソード数", initial_value=20, min=1, max=500,
                                            step=1)
            self._demo_seed = gui.add_number("シード", initial_value=0, min=0, max=9999, step=1)
            self._demo_status = gui.add_markdown("待機中")
            demo_btn = gui.add_button(
                "▶ ピックのデモを自動生成", disabled=scenario is None or not scenario.objects
            )
        demo_btn.on_click(
            lambda _: app.generate_demos(
                self._rec_repo.value, self._rec_root.value,
                int(self._demo_eps.value), int(self._rec_fps.value),
                int(self._demo_seed.value),
            )
        )

        if app.teleop_rx is not None:
            rx = app.teleop_rx
            with gui.add_folder("🎮 リモートテレオペ", expand_by_default=True):
                self._teleop_md = gui.add_markdown("接続待ち…")
                gui.add_markdown(
                    "手元のPCから接続:\n\n"
                    f"```\nssh -L {rx.port}:localhost:{rx.port} <this-host>\n"
                    f"so101-tool teleop-client --connect localhost:{rx.port} \\\n"
                    f"    --token {rx.token} --port /dev/ttyACM0\n```"
                )
                teleop_on = gui.add_button("▶ テレオペ開始", color="green")
                teleop_off = gui.add_button("⏸ テレオペ停止")
            teleop_on.on_click(lambda _: self._put(SetMode(mode=Mode.TELEOP)))
            teleop_off.on_click(lambda _: self._put(SetMode(mode=Mode.IDLE)))
        else:
            self._teleop_md = None
            gui.add_markdown(
                "*リモートテレオペは `--teleop` で起動すると使えます "
                "([docs/ja/teleop_remote.md](https://github.com/pawntest/vla_play))*"
            )

    def _build_ai_tab(self, gui, app, scenario) -> None:
        default_repo = app.record_repo or (scenario.name if scenario else "my_dataset")

        with gui.add_folder("🤖 学習済みポリシーを実行", expand_by_default=True):
            self._policy_path = gui.add_text(
                "チェックポイント", initial_value=self._config.policy_path or
                "outputs/train/checkpoints/last/pretrained_model"
            )
            self._policy_task = gui.add_text(
                "タスク指示文", initial_value=self._config.policy_task
                or (scenario.task if scenario else "")
            )
            self._policy_status = gui.add_markdown(app.policy_status)
            pol_load = gui.add_button("▶ 読み込んで実行", color="violet")
            pol_stop = gui.add_button("⏸ ポリシー停止")
        pol_load.on_click(
            lambda _: app.load_policy(self._policy_path.value, self._policy_task.value)
        )
        pol_stop.on_click(lambda _: app.stop_policy())

        with gui.add_folder("🎓 学習 (lerobot-train)", expand_by_default=False):
            self._train_dataset = gui.add_text("データセット (フォルダ/Hub ID)",
                                               initial_value=f"data/{default_repo}")
            self._train_policy = gui.add_dropdown("ポリシー種別",
                                                  options=["act", "smolvla", "diffusion"],
                                                  initial_value="act")
            self._train_steps = gui.add_number("ステップ数", initial_value=20000, min=1,
                                               max=1_000_000, step=1000)
            self._train_batch = gui.add_number("バッチサイズ", initial_value=8, min=1,
                                               max=256, step=1)
            self._train_device = gui.add_dropdown("デバイス",
                                                  options=["auto", "cuda", "mps", "cpu"],
                                                  initial_value="auto")
            self._train_out = gui.add_text("出力先", initial_value="outputs/train")
            self._train_extra = gui.add_text("追加引数", initial_value="")
            self._train_status = gui.add_markdown("待機中")
            train_btn = gui.add_button("▶ 学習開始", color="green")
            train_stop = gui.add_button("⏹ 学習停止")
            self._colab_repo = gui.add_text("HFデータセットrepo (Colab用)", initial_value="")
            colab_btn = gui.add_button("📓 Colabノートブックを書き出す(無料GPU)")
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
                int(self._train_steps.value), int(self._train_batch.value),
                "train_colab.ipynb",
            )
        )

    def _build_scene_tab(self, gui, app, scenario) -> None:
        title = scenario.name if scenario else "素のアーム(物理シーンなし)"
        self._scene_md = gui.add_markdown(
            f"**{title}**\n\nタスク: *{scenario.task if scenario else '-'}*"
        )
        options = ["(bare arm)"] + _find_scenarios()
        current = scenario.source_path if scenario and scenario.source_path else "(bare arm)"
        if current not in options:
            options.append(current)
        self._scene_dd = gui.add_dropdown("シーンファイル", options=options,
                                          initial_value=current)
        load_btn = gui.add_button("📂 シーンを読み込む")
        load_btn.on_click(lambda _: self._load_selected_scenario())
        if scenario is None:
            return

        reset_btn = gui.add_button("🎲 配置をランダムにリセット")
        reset_btn.on_click(lambda _: app.backend.reset(randomize=True))
        self._task_input = gui.add_text("タスク指示文", initial_value=scenario.task)

        with gui.add_folder("📦 物体の編集", expand_by_default=False):
            names = [o.name for o in scenario.objects] or ["-"]
            self._obj_dd = gui.add_dropdown("物体", options=names, initial_value=names[0])
            rm_btn = gui.add_button("選択した物体を削除")
            rm_btn.on_click(lambda _: self._remove_object())
            self._drag_cb = gui.add_checkbox("🖱 配置ギズモを表示(ドラッグで移動)",
                                             initial_value=False)
            self._drag_cb.on_update(lambda _: self._toggle_drag_gizmos())
            self._click_add_cb = gui.add_checkbox("➕ シーンをクリックした場所に追加",
                                                  initial_value=False)
            self._click_add_cb.on_update(lambda _: self._toggle_click_add())
            self._new_name = gui.add_text("名前", initial_value="obj1")
            self._new_type = gui.add_dropdown(
                "種類", options=["box", "sphere", "cylinder", "cloth"], initial_value="box"
            )
            self._new_size = gui.add_vector3("サイズ (m)",
                                             initial_value=(0.015, 0.015, 0.015), step=0.005)
            self._new_pos = gui.add_vector3("位置 (m)", initial_value=(0.25, 0.0, 0.02),
                                            step=0.01)
            self._new_color = gui.add_rgb("色", initial_value=(220, 30, 30))
            self._new_mass = gui.add_number("質量 (kg)", initial_value=0.03, min=0.001,
                                            max=2.0, step=0.01)
            self._new_noise = gui.add_slider("位置ランダム幅 ± (cm)", min=0.0, max=10.0,
                                             step=0.5, initial_value=4.0)
            add_btn = gui.add_button("➕ 物体を追加(シーン再構築)")
            add_btn.on_click(lambda _: self._add_object())

        with gui.add_folder("📷 カメラの編集", expand_by_default=False):
            cam_names = [c.name for c in scenario.cameras] or ["-"]
            self._cam_dd = gui.add_dropdown("カメラ", options=cam_names,
                                            initial_value=cam_names[0])
            cam_rm = gui.add_button("選択したカメラを削除")
            cam_rm.on_click(lambda _: self._remove_camera())
            self._cam_name = gui.add_text("名前", initial_value="cam1")
            self._cam_attach = gui.add_dropdown(
                "取り付け先", options=["world"] + ATTACHABLE_BODIES, initial_value="world",
                hint="worldは固定カメラ。アーム部位を選ぶと一緒に動きます(座標はその部位基準)",
            )
            self._cam_pos = gui.add_vector3("位置 (m)", initial_value=(0.5, -0.3, 0.35),
                                            step=0.01)
            self._cam_lookat = gui.add_vector3("注視点 (m)", initial_value=(0.25, 0.0, 0.05),
                                               step=0.01)
            self._cam_fovy = gui.add_slider("画角 (deg)", min=20, max=120, step=1,
                                            initial_value=58)
            cam_add = gui.add_button("➕ カメラを追加(シーン再構築)")
            cam_add.on_click(lambda _: self._add_camera())

        with gui.add_folder("🏞 環境の編集", expand_by_default=False):
            env = scenario.environment
            self._env_table = gui.add_checkbox("テーブル", initial_value=env.table is not None)
            tbl = env.table or TableSpec()
            self._env_table_size = gui.add_vector2("テーブルサイズ (m)",
                                                   initial_value=tuple(tbl.size), step=0.05)
            self._env_table_rgb = gui.add_rgb(
                "テーブル色", initial_value=tuple(int(c * 255) for c in tbl.rgba[:3])
            )
            self._env_checker = gui.add_checkbox("チェッカー床", initial_value=env.floor.checker)
            self._env_floor_rgb = gui.add_rgb(
                "床の色", initial_value=tuple(int(c * 255) for c in env.floor.rgba[:3])
            )
            env_btn = gui.add_button("✔ 環境を適用(シーン再構築)")
            env_btn.on_click(lambda _: self._apply_environment())

        self._save_path = gui.add_text(
            "保存先", initial_value=scenario.source_path or "scenarios/my_scene.yaml"
        )
        save_btn = gui.add_button("💾 シーンをYAMLに保存")
        save_btn.on_click(lambda _: self._save_scenario())

    # ------------------------------------------------------------- lifecycle --

    def cleanup_gui(self) -> None:
        """Dismantle the tab group tab-by-tab BEFORE gui.reset() — viser's
        reset removes the group first and then chokes updating tab labels."""
        for tab in getattr(self, "_tabs", []):
            try:
                tab.remove()
            except Exception:
                pass
        try:
            self._tab_group.remove()
        except Exception:
            pass

    def cleanup_scene(self) -> None:
        """Remove scene nodes owned by this panel (called before a rebuild)."""
        for g in self._drag_gizmos:
            try:
                g.remove()
            except Exception:
                pass
        self._drag_gizmos.clear()
        try:
            self._gizmo.remove()
        except Exception:
            pass

    # --------------------------------------------------------------- helpers --

    def _put(self, cmd) -> None:
        self._loop.commands.put(cmd)

    def _put_motion(self, cmd) -> None:
        """Enqueue a motion primitive, auto-switching to RULE mode first."""
        snap = self._loop.snapshot()
        if snap is None or snap.mode is not Mode.RULE:
            self._put(SetMode(mode=Mode.RULE))
        self._put(cmd)

    def _set_direct_drag(self) -> None:
        if self._app.direct_drag is not None:
            self._app.direct_drag.enabled = self._direct_drag_cb.value

    def _snap_gizmo_to_tcp(self) -> None:
        snap = self._loop.snapshot()
        if snap is not None:
            self._gizmo.position = snap.tcp_position
            self._gizmo.wxyz = snap.tcp_wxyz

    def _load_selected_scenario(self) -> None:
        sel = self._scene_dd.value
        self._app.rebuild(None if sel == "(bare arm)" else sel)

    # -- interactive placement -------------------------------------------------

    def _toggle_drag_gizmos(self) -> None:
        for g in self._drag_gizmos:
            try:
                g.remove()
            except Exception:
                pass
        self._drag_gizmos.clear()
        if not self._drag_cb.value:
            return
        backend = self._app.backend
        for obj in self._app.scenario.objects:
            try:
                pos, wxyz = backend.object_pose(obj.name)
            except Exception:
                continue
            gizmo = self._server.scene.add_transform_controls(
                f"/edit/{obj.name}", scale=0.1, position=pos, wxyz=wxyz,
                disable_rotations=obj.is_cloth,
            )

            def _moved(_evt, name=obj.name, g=gizmo):
                self._app.backend.set_object_pose(name, np.array(g.position),
                                                  np.array(g.wxyz))

            gizmo.on_update(_moved)
            self._drag_gizmos.append(gizmo)

    def _toggle_click_add(self) -> None:
        if self._click_add_cb.value and not self._click_add_registered:
            self._click_add_registered = True

            @self._server.scene.on_pointer_event(event_type="click")
            def _(event) -> None:
                if not self._click_add_cb.value:
                    return
                o = np.array(event.ray_origin)
                d = np.array(event.ray_direction)
                if abs(d[2]) < 1e-6:
                    return
                # intersect the click ray with the working surface (z = 0)
                t = -o[2] / d[2]
                if t <= 0:
                    return
                hit = o + t * d
                self._new_pos.value = (round(float(hit[0]), 3), round(float(hit[1]), 3),
                                       max(self._new_size.value[2], 0.01))
                self._add_object()

    def _remove_camera(self) -> None:
        sc = self._edited_scenario()
        name = self._cam_dd.value
        sc.cameras = [c for c in sc.cameras if c.name != name]
        self._app.rebuild_with_scenario(sc)

    def _add_camera(self) -> None:
        sc = self._edited_scenario()
        attach = self._cam_attach.value
        sc.cameras.append(
            CameraSpec(
                name=self._cam_name.value.strip() or f"cam{len(sc.cameras) + 1}",
                pos=list(self._cam_pos.value),
                lookat=list(self._cam_lookat.value),
                fovy=float(self._cam_fovy.value),
                attach_to=None if attach == "world" else attach,
            )
        )
        self._app.rebuild_with_scenario(sc)

    def _apply_environment(self) -> None:
        sc = self._edited_scenario()
        floor_rgb = [c / 255.0 for c in self._env_floor_rgb.value]
        sc.environment.floor = FloorSpec(rgba=[*floor_rgb, 1.0],
                                         checker=self._env_checker.value)
        if self._env_table.value:
            tbl_rgb = [c / 255.0 for c in self._env_table_rgb.value]
            sc.environment.table = TableSpec(size=list(self._env_table_size.value),
                                             rgba=[*tbl_rgb, 1.0])
        else:
            sc.environment.table = None
        self._app.rebuild_with_scenario(sc)

    def _edited_scenario(self) -> Scenario:
        sc = self._app.scenario
        sc.task = self._task_input.value
        return sc

    def _add_object(self) -> None:
        sc = self._edited_scenario()
        rgb = [c / 255.0 for c in self._new_color.value]
        noise_m = self._new_noise.value / 100.0
        base = self._new_name.value.strip() or "obj"
        name = base
        taken = {o.name for o in sc.objects}
        i = 1
        while name in taken:
            i += 1
            name = f"{base}{i}"
        kind = self._new_type.value
        size = list(self._new_size.value)
        sc.objects.append(
            ObjectSpec(
                name=name,
                type=kind,
                size=size[:2] if kind == "cloth" else size,
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
        self._nl_log.content += f"\n\n**あなた:** {text}"

        def worker():
            try:
                reply = self._app.nl_agent.handle(text)
            except Exception as exc:  # never crash the GUI thread
                reply = f"error: {exc}"
            self._nl_log.content += f"\n\n**ロボット:** {reply}"
            self._nl_busy = False
            self._nl_send.disabled = False

        threading.Thread(target=worker, daemon=True, name="so101-nl").start()

    # ------------------------------------------------------- render-loop hook --

    def _banner_state(self, snap: LoopSnapshot) -> str:
        app = self._app
        tcp = np.round(snap.tcp_position * 1000).astype(int)
        info = (f"TCP: {tcp[0]}, {tcp[1]}, {tcp[2]} mm ・ グリッパー {snap.gripper:.2f} ・ "
                f"{'実機' if snap.backend_is_real else 'シム'}")
        badge = "🔴 REC" if (app.recorder is not None and app.recorder.recording) else ""

        if snap.estop:
            return _banner("#dc2626", "🛑 非常停止中",
                           "「非常停止を解除する」を押すまで一切動きません\n" + info, badge)
        color, title, sub = _MODE_BANNER[snap.mode]
        if snap.mode is Mode.RULE:
            act = _CMD_JP.get(snap.active_command or "", None)
            sub = f"実行中: {act}…" if act else "指示待ち(スライダー・ギズモ・ドラッグ・自然言語)"
        elif snap.mode is Mode.POLICY:
            sub = app.policy_status
        elif snap.mode is Mode.TELEOP and app.teleop_rx is not None:
            sub = ("🟢 リーダーアームのストリームに追従中" if app.teleop_rx.fresh
                   else "⚪ ストリーム待ち(接続が切れると保持します)")
        if snap.error:
            sub += f"\n⚠ {snap.error}"
        return _banner(color, title, sub + "\n" + info, badge)

    def update(self, snap: LoopSnapshot | None) -> None:
        self._tick += 1
        if snap is None or self._tick % 6:  # ~5 Hz updates at 30 Hz render
            return
        app = self._app

        html = self._banner_state(snap)
        if self._banner_html.content != html:
            self._banner_html.content = html
        if self._reset_estop_btn.visible != snap.estop:
            self._reset_estop_btn.visible = snap.estop

        if self._teleop_md is not None and app.teleop_rx is not None:
            rx = app.teleop_rx
            live = "🟢 接続中(ストリーム受信)" if rx.fresh else f"⚪ {rx.status}"
            tcontent = f"{live} — 127.0.0.1:{rx.port}(トークン保護・SSHトンネル専用)"
            if self._teleop_md.content != tcontent:
                self._teleop_md.content = tcontent

        if app.recorder is not None:
            marker = "🔴 " if app.recorder.recording else ""
            content = f"{marker}{app.recorder.status} — 保存済みエピソード: {app.recorder.episodes}"
        else:
            content = "待機中"
        if self._rec_status.content != content:
            self._rec_status.content = content
        for handle, text in (
            (self._demo_status, app.demo_status),
            (self._policy_status, app.policy_status),
            (self._train_status, app.train_status),
        ):
            if handle.content != text:
                handle.content = text

        if snap.mode in (Mode.IDLE, Mode.MIRROR):
            for s, v in zip(self._joint_sliders, snap.q):
                s.value = float(np.clip(v, s.min, s.max))
            self._gripper_slider.value = float(snap.gripper)
