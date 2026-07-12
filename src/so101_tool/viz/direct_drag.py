"""Direct mouse-drag manipulation of 3D meshes in the viser browser view.

Two kinds of grab-and-drag, wired per mesh node from ``RobotView.mesh_nodes``:

* **Scene objects** (bodies named ``obj_<name>``): the object teleports to follow
  the pointer. On grab we capture ``offset = object_pos - grab_point`` so the
  object keeps its relative position under the cursor, then on every update we
  set the object pose to ``pointer + offset``.
* **The robot gripper** (bodies ``gripper`` / ``moving_jaw_so101_v1`` / ``wrist``):
  dragging commands an IK "chase" of the pointer. We take the pointer motion as a
  world-space delta and enqueue a ``MoveL`` toward ``tcp + delta``; the control
  loop's preemption absorbs the ~20 Hz churn.

Drag-plane math contract (viser ``SceneNodeDragEvent``):
  * ``start_position`` — LIVE world coords of the grabbed point; it tracks the
    node as it moves, so the object/robot delta is always measured from where the
    grab currently is, not where it began.
  * ``end_position`` — the current pointer, projected onto a camera-aligned plane
    through the grab point, in world coords. Object drag places the object there
    (plus offset); robot drag uses ``end - start`` as the incremental target delta.
  * ``phase`` — "start", "update" (~20 Hz), "end" (always fires).

Callbacks run on viser's websocket thread: they never block or ``wait()`` and are
fully wrapped in try/except so a drag landing mid scene-rebuild can never crash it.
"""

from __future__ import annotations

import numpy as np

from ..control.commands import Mode, MoveL, SetMode

_ROBOT_BODIES = {"gripper", "moving_jaw_so101_v1", "wrist"}
_FLOOR_Z = 0.005  # objects: keep above the working surface
_TCP_MIN_Z = 0.01  # robot: don't drive the TCP into the floor
_WORKSPACE_R = 0.45  # robot: clamp target into a reachable ball around the base


class DirectDrag:
    """Grab-and-drag manipulation: scene objects follow the pointer, and
    dragging the gripper commands an IK chase of the pointer."""

    enabled: bool = True

    def __init__(self, app):
        self._app = app
        # Handlers kept referenceable so tests can invoke them without a browser.
        self._object_handlers: dict[str, callable] = {}
        self._robot_handler: callable | None = None
        self._robot_tick = 0
        for body_name, handle in app.view.mesh_nodes:
            if body_name.startswith("obj_"):
                name = body_name[len("obj_"):]
                fn = self._make_object_handler(name)
                # multiple geoms share a body name; one handler per object suffices
                self._object_handlers.setdefault(name, fn)
                handle.on_drag(fn)
            elif body_name in _ROBOT_BODIES:
                if self._robot_handler is None:
                    self._robot_handler = self._make_robot_handler()
                handle.on_drag(self._robot_handler)

    # -- object drag ---------------------------------------------------------

    def _make_object_handler(self, name: str):
        state = {"offset": np.zeros(3)}

        def _handler(event) -> None:
            if not self.enabled:
                return
            try:
                if event.phase == "start":
                    pos = self._app.backend.object_pose(name)[0]
                    state["offset"] = np.asarray(pos) - np.asarray(event.start_position)
                    return
                pos = np.asarray(event.end_position, dtype=float) + state["offset"]
                pos[2] = max(pos[2], _FLOOR_Z)
                self._app.backend.set_object_pose(name, pos)
            except Exception:
                # scene rebuild / stale backend — drop the event, never crash viser
                return

        return _handler

    # -- robot drag ----------------------------------------------------------

    def _make_robot_handler(self):
        def _handler(event) -> None:
            if not self.enabled:
                return
            try:
                loop = self._app.loop
                if event.phase == "start":
                    snap = loop.snapshot()
                    if snap is not None and snap.mode is not Mode.RULE:
                        loop.commands.put(SetMode(mode=Mode.RULE))  # do NOT wait
                    return
                if event.phase == "update":
                    self._robot_tick += 1
                    if self._robot_tick % 2:  # throttle: every 2nd update
                        return
                snap = loop.snapshot()
                if snap is None:
                    return
                delta = np.asarray(event.end_position, dtype=float) - np.asarray(
                    event.start_position, dtype=float
                )
                target = np.asarray(snap.tcp_position, dtype=float) + delta
                target[2] = max(target[2], _TCP_MIN_Z)
                norm = float(np.linalg.norm(target))
                if norm > _WORKSPACE_R:
                    target *= _WORKSPACE_R / norm
                loop.commands.put(MoveL(position=target, wxyz=None, speed=0.9))
            except Exception:
                return

        return _handler


def install_direct_drag(app) -> DirectDrag:
    """Walk ``app.view.mesh_nodes`` and bind on_drag callbacks. Works for both
    bare-arm sessions (robot drag only) and scenario sessions (objects + robot)."""
    return DirectDrag(app)
