"""Full-stack headless smoke test: App (backend + loop + viser view/panel)."""

import socket
import time

import numpy as np
import viser

from so101_tool.app import App
from so101_tool.config import AppConfig
from so101_tool.control.commands import Mode, MoveJ, SetMode


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _drive(app, timeout, predicate):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.render_tick()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_full_stack_boots_moves_and_rebuilds():
    server = viser.ViserServer(port=_free_port(), verbose=False)
    app = App(server, AppConfig(control_hz=100.0))
    try:
        # bare-arm sim: mode switch + MoveJ through the GUI-facing loop
        mode = SetMode(mode=Mode.RULE)
        app.loop.commands.put(mode)
        assert mode.wait(2.0) and mode.ok

        target = np.array([0.3, -0.2, 0.3, 0.1, 0.05])
        mv = MoveJ(q=target, speed=1.0)
        app.loop.commands.put(mv)
        assert _drive(app, 10.0, lambda: mv.wait(0.0)), "MoveJ did not finish"
        assert mv.ok, mv.error
        assert _drive(
            app, 2.0,
            lambda: np.max(np.abs(app.loop.snapshot().q - target)) < 0.03,
        )

        # rebuild into a physics scenario from the running app (as the GUI does)
        old_loop = app.loop
        app.rebuild("examples/pick_cube.yaml")
        assert _drive(app, 30.0, lambda: app.scenario is not None and app.loop is not old_loop)
        assert app.scenario.name == "pick_cube"
        assert hasattr(app.backend, "get_camera_frames")
        assert _drive(app, 5.0, lambda: app.loop.snapshot() is not None)
        assert app.loop.snapshot().qpos_full is not None  # objects in the scene
        assert app.panel is not None

        # rebuild back to the bare arm
        old_loop = app.loop
        app.rebuild(None)
        assert _drive(app, 30.0, lambda: app.scenario is None and app.loop is not old_loop)
    finally:
        app.shutdown()
        server.stop()
