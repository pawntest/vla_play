"""Headless tests for direct mouse-drag manipulation (no browser needed).

We build the real App stack on a free viser port, then invoke the drag handlers
that DirectDrag registered with viser directly, feeding them fake
SceneNodeDragEvent objects (SimpleNamespace with the fields viser provides).
"""

import socket
import time
import types

import numpy as np
import viser

from so101_tool.app import App
from so101_tool.config import AppConfig
from so101_tool.control.commands import Mode


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _drive(app, timeout, predicate) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.render_tick()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _event(phase, start_position, end_position):
    return types.SimpleNamespace(
        phase=phase,
        start_position=np.asarray(start_position, dtype=float),
        end_position=np.asarray(end_position, dtype=float),
        button="left",
        modifier=None,
    )


def _make_app():
    server = viser.ViserServer(port=_free_port(), verbose=False)
    app = App(server, AppConfig(control_hz=200.0), scenario_path="examples/pick_cube.yaml")
    return server, app


def test_object_drag_follows_pointer():
    server, app = _make_app()
    try:
        handler = app.direct_drag._object_handlers["cube"]
        pos0 = app.backend.object_pose("cube")[0]
        # Grab exactly at the object -> zero offset for a clean expected target.
        handler(_event("start", pos0, pos0))
        end = np.array([0.30, 0.10, 0.12])
        handler(_event("update", pos0, end))
        moved = app.backend.object_pose("cube")[0]
        assert np.allclose(moved, end, atol=5e-3), (moved, end)
    finally:
        app.shutdown()
        server.stop()


def test_object_drag_z_clamped_to_floor():
    server, app = _make_app()
    try:
        handler = app.direct_drag._object_handlers["cube"]
        pos0 = app.backend.object_pose("cube")[0]
        handler(_event("start", pos0, pos0))
        handler(_event("update", pos0, [0.30, 0.0, -0.5]))  # below the floor
        moved = app.backend.object_pose("cube")[0]
        assert abs(moved[2] - 0.005) < 1e-3, moved
    finally:
        app.shutdown()
        server.stop()


def test_robot_drag_commands_ik_chase():
    server, app = _make_app()
    try:
        assert _drive(app, 5.0, lambda: app.loop.snapshot() is not None)
        handler = app.direct_drag._robot_handler
        assert handler is not None, "no robot mesh nodes found"

        delta = np.array([0.03, 0.0, 0.0])
        # start: not RULE yet -> handler enqueues SetMode(RULE) (does not wait)
        handler(_event("start", np.zeros(3), delta))
        assert _drive(app, 3.0, lambda: app.loop.snapshot().mode is Mode.RULE)

        tcp0 = np.asarray(app.loop.snapshot().tcp_position, dtype=float)
        target = tcp0 + delta
        target[2] = max(target[2], 0.01)
        norm = float(np.linalg.norm(target))
        if norm > 0.45:
            target *= 0.45 / norm
        # two updates: first is throttled, the second enqueues one MoveL
        handler(_event("update", np.zeros(3), delta))
        handler(_event("update", np.zeros(3), delta))

        reached = _drive(
            app, 12.0,
            lambda: np.linalg.norm(app.loop.snapshot().tcp_position - target) < 0.01,
        )
        final = app.loop.snapshot().tcp_position
        assert reached, (final, target)
    finally:
        app.shutdown()
        server.stop()


def test_disabled_handlers_noop():
    server, app = _make_app()
    try:
        app.direct_drag.enabled = False
        handler = app.direct_drag._object_handlers["cube"]
        pos0 = app.backend.object_pose("cube")[0]
        handler(_event("start", pos0, pos0))
        handler(_event("update", pos0, [0.30, 0.10, 0.12]))
        after = app.backend.object_pose("cube")[0]
        assert np.allclose(after, pos0, atol=1e-4), (after, pos0)
    finally:
        app.shutdown()
        server.stop()
