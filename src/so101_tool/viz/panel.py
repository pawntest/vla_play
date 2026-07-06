"""viser GUI panel: mode switch, e-stop, joint/cartesian motion, NL chat.

All callbacks only enqueue Commands on the control loop (never block, never
touch the robot). update(snap) is called from the render loop in the main
thread and refreshes readouts.
"""

from __future__ import annotations

import threading

import numpy as np
import viser

from ..config import ARM_JOINTS, ARM_LIMITS_HI, ARM_LIMITS_LO, AppConfig
from ..control.commands import Home, JogTool, Mode, MoveJ, MoveL, SetGripper, SetMode, Stop
from ..control.loop import ControlLoop, LoopSnapshot
from ..kinematics import Kinematics

_MODES = [m.value for m in Mode]


class ControlPanel:
    def __init__(
        self,
        server: viser.ViserServer,
        loop: ControlLoop,
        kinematics: Kinematics,
        config: AppConfig,
        nl_agent=None,
    ):
        self._loop = loop
        self._kin = kinematics
        self._config = config
        self._nl_agent = nl_agent
        self._tick = 0
        self._nl_busy = False
        gui = server.gui

        with gui.add_folder("Status"):
            self._mode_dd = gui.add_dropdown("Mode", options=_MODES, initial_value=Mode.IDLE.value)
            self._status_md = gui.add_markdown("connecting…")
            estop_btn = gui.add_button("EMERGENCY STOP", color="red")
            reset_btn = gui.add_button("Reset e-stop")
            stop_btn = gui.add_button("Stop motion")

        with gui.add_folder("Joints (RULE)"):
            self._speed = gui.add_slider("speed", min=0.1, max=1.0, step=0.05, initial_value=0.5)
            self._joint_sliders = [
                gui.add_slider(
                    name, min=float(ARM_LIMITS_LO[i]), max=float(ARM_LIMITS_HI[i]),
                    step=0.01, initial_value=0.0,
                )
                for i, name in enumerate(ARM_JOINTS)
            ]
            self._gripper_slider = gui.add_slider("gripper", min=0.0, max=1.0, step=0.01,
                                                  initial_value=0.0)
            movej_btn = gui.add_button("Move to sliders")
            home_btn = gui.add_button("Home")

        with gui.add_folder("Cartesian (RULE)"):
            self._gizmo = server.scene.add_transform_controls("/target", scale=0.15)
            snap_btn = gui.add_button("Snap target to TCP")
            go_pos_btn = gui.add_button("Go to target (position)")
            go_pose_btn = gui.add_button("Go to target (pose)")
            self._jog_step = gui.add_slider("jog step (cm)", min=0.5, max=10.0, step=0.5,
                                            initial_value=2.0)
            jog_btns = {}
            for axis, vec in (("X", (1, 0, 0)), ("Y", (0, 1, 0)), ("Z", (0, 0, 1))):
                jog_btns[f"+{axis}"] = gui.add_button(f"Jog +{axis} (tool)")
                jog_btns[f"-{axis}"] = gui.add_button(f"Jog -{axis} (tool)")

        if nl_agent is not None:
            with gui.add_folder("Natural language"):
                self._nl_input = gui.add_text("command", initial_value="")
                self._nl_send = gui.add_button("Send")
                self._nl_log = gui.add_markdown("")
                self._nl_send.on_click(lambda _: self._send_nl())

        # -- callbacks (enqueue only) --------------------------------------

        self._mode_dd.on_update(lambda _: self._put(SetMode(mode=Mode(self._mode_dd.value))))
        estop_btn.on_click(lambda _: loop.estop())
        reset_btn.on_click(lambda _: loop.reset_estop())
        stop_btn.on_click(lambda _: self._put(Stop()))
        home_btn.on_click(lambda _: self._put(Home(speed=self._speed.value)))
        movej_btn.on_click(
            lambda _: self._put(
                MoveJ(
                    q=np.array([s.value for s in self._joint_sliders]),
                    gripper=self._gripper_slider.value,
                    speed=self._speed.value,
                )
            )
        )
        snap_btn.on_click(lambda _: self._snap_gizmo_to_tcp())
        go_pos_btn.on_click(
            lambda _: self._put(
                MoveL(position=np.array(self._gizmo.position), wxyz=None,
                      speed=self._speed.value)
            )
        )
        go_pose_btn.on_click(
            lambda _: self._put(
                MoveL(position=np.array(self._gizmo.position),
                      wxyz=np.array(self._gizmo.wxyz), speed=self._speed.value)
            )
        )
        for label, btn in jog_btns.items():
            sign = 1.0 if label[0] == "+" else -1.0
            axis = "XYZ".index(label[1])

            def _jog(_evt, sign=sign, axis=axis):
                dpos = np.zeros(3)
                dpos[axis] = sign * self._jog_step.value / 100.0
                self._put(JogTool(dpos=dpos, frame="tool", speed=self._speed.value))

            btn.on_click(_jog)

    # -- helpers -------------------------------------------------------------

    def _put(self, cmd) -> None:
        self._loop.commands.put(cmd)

    def _snap_gizmo_to_tcp(self) -> None:
        snap = self._loop.snapshot()
        if snap is not None:
            self._gizmo.position = snap.tcp_position
            self._gizmo.wxyz = snap.tcp_wxyz

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
                reply = self._nl_agent.handle(text)
            except Exception as exc:  # never crash the GUI thread
                reply = f"error: {exc}"
            self._nl_log.content += f"\n\n**robot:** {reply}"
            self._nl_busy = False
            self._nl_send.disabled = False

        threading.Thread(target=worker, daemon=True, name="so101-nl").start()

    # -- render-loop hook ------------------------------------------------------

    def update(self, snap: LoopSnapshot | None) -> None:
        self._tick += 1
        if snap is None or self._tick % 6:  # ~5 Hz text updates at 30 Hz render
            return
        if self._mode_dd.value != snap.mode.value:
            self._mode_dd.value = snap.mode.value
        joints = " ".join(f"{v:+.2f}" for v in snap.q)
        tcp = np.round(snap.tcp_position * 1000).astype(int)
        lines = [
            f"backend: **{'real' if snap.backend_is_real else 'sim'}**",
            f"q (rad): `{joints}`  grip: `{snap.gripper:.2f}`",
            f"TCP (mm): `{tcp[0]} {tcp[1]} {tcp[2]}`",
            f"active: `{snap.active_command or '-'}`",
        ]
        if snap.estop:
            lines.insert(0, "## 🛑 E-STOP LATCHED")
        if snap.error:
            lines.append(f"last error: `{snap.error}`")
        self._status_md.content = "\n\n".join(lines)
        if snap.mode in (Mode.IDLE, Mode.MIRROR):
            for s, v in zip(self._joint_sliders, snap.q):
                s.value = float(np.clip(v, s.min, s.max))
            self._gripper_slider.value = float(snap.gripper)
