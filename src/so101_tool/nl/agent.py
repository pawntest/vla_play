"""Natural-language robot control via the Claude API.

The model gets a small set of tools that map 1:1 onto the rule-based motion
primitives; every motion still flows through the control loop and its safety
filter. handle() is blocking and is meant to run in a worker thread (the viser
panel does this); it never raises — errors come back as the reply string.
"""

from __future__ import annotations

import os

import numpy as np

from ..control.commands import Home, JogTool, Mode, MoveL, SetGripper, SetMode, Stop

_CMD_TIMEOUT = 25.0  # s per motion tool call
_MAX_TOOL_ROUNDS = 10
_MAX_HISTORY = 30  # messages kept across handle() calls

_SYSTEM = """You control a LeRobot SO-101 robot arm (5 joints + gripper, reach ~0.45 m).
Base frame: +X forward, +Y left, +Z up; origin at the arm base. The TCP is the
point between the gripper jaws. Tool distances are millimeters.

Rules:
- ALWAYS call get_state first to know where the arm is before moving it.
- Prefer small, conservative motions (<= 100 mm per step). If a request is
  ambiguous or looks unsafe, ask instead of guessing.
- The workspace is roughly a hemisphere of radius 450 mm around the base;
  keep the TCP above z = 20 mm unless explicitly asked.
- Reply briefly with what you did and the resulting TCP position."""

_TOOLS = [
    {
        "name": "get_state",
        "description": "Current joint angles (deg), gripper opening (%), TCP position (mm).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "move_tool_relative",
        "description": "Move the TCP by a relative offset in mm. frame='tool' uses the "
        "gripper axes, frame='base' the robot base axes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dx_mm": {"type": "number"},
                "dy_mm": {"type": "number"},
                "dz_mm": {"type": "number"},
                "frame": {"type": "string", "enum": ["tool", "base"], "default": "tool"},
            },
            "required": ["dx_mm", "dy_mm", "dz_mm"],
        },
    },
    {
        "name": "move_to_point",
        "description": "Move the TCP to an absolute point in the base frame (mm).",
        "input_schema": {
            "type": "object",
            "properties": {
                "x_mm": {"type": "number"},
                "y_mm": {"type": "number"},
                "z_mm": {"type": "number"},
            },
            "required": ["x_mm", "y_mm", "z_mm"],
        },
    },
    {
        "name": "set_gripper",
        "description": "Open/close the gripper. 0 = fully closed, 100 = fully open.",
        "input_schema": {
            "type": "object",
            "properties": {"percent": {"type": "number", "minimum": 0, "maximum": 100}},
            "required": ["percent"],
        },
    },
    {
        "name": "go_home",
        "description": "Move all joints to the home (zero) pose.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "stop",
        "description": "Abort the current motion and hold position.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


class NLAgent:
    def __init__(self, loop, kinematics, model: str = "claude-opus-4-8", client=None):
        self._loop = loop
        self._kin = kinematics
        self._model = model
        self._client = client
        self._history: list[dict] = []

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    'anthropic SDK not installed: pip install "so101-tool[nl]"'
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic()
        return self._client

    # -- tool implementations ---------------------------------------------

    def _ensure_rule_mode(self) -> str | None:
        snap = self._loop.snapshot()
        if snap is not None and snap.mode is Mode.RULE:
            return None
        cmd = SetMode(mode=Mode.RULE)
        self._loop.commands.put(cmd)
        if not cmd.wait(5.0):
            return "could not switch to RULE mode (timeout)"
        return cmd.error

    def _run_motion(self, cmd) -> str:
        err = self._ensure_rule_mode()
        if err:
            return f"error: {err}"
        self._loop.commands.put(cmd)
        if not cmd.wait(_CMD_TIMEOUT):
            self._loop.commands.put(Stop())
            return "error: motion timed out and was stopped"
        if not cmd.ok:
            return f"error: {cmd.error}"
        return f"ok. {self._state_text()}"

    def _state_text(self) -> str:
        snap = self._loop.snapshot()
        if snap is None:
            return "state unavailable (loop starting)"
        deg = ", ".join(f"{name}={np.degrees(v):.0f}" for name, v in
                        zip(("pan", "lift", "elbow", "wrist", "roll"), snap.q))
        tcp = (snap.tcp_position * 1000).round(0).astype(int)
        return (
            f"joints(deg): {deg}; gripper: {snap.gripper * 100:.0f}%; "
            f"TCP(mm): x={tcp[0]} y={tcp[1]} z={tcp[2]}; mode={snap.mode.value}"
            + ("; E-STOP LATCHED" if snap.estop else "")
        )

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "get_state":
            return self._state_text()
        if name == "move_tool_relative":
            dpos = np.array([args["dx_mm"], args["dy_mm"], args["dz_mm"]]) / 1000.0
            if np.linalg.norm(dpos) > 0.25:
                return "error: refused, relative step > 250 mm"
            return self._run_motion(JogTool(dpos=dpos, frame=args.get("frame", "tool")))
        if name == "move_to_point":
            pos = np.array([args["x_mm"], args["y_mm"], args["z_mm"]]) / 1000.0
            return self._run_motion(MoveL(position=pos, wxyz=None))
        if name == "set_gripper":
            return self._run_motion(SetGripper(fraction=float(args["percent"]) / 100.0))
        if name == "go_home":
            return self._run_motion(Home())
        if name == "stop":
            cmd = Stop()
            self._loop.commands.put(cmd)
            cmd.wait(5.0)
            return "stopped"
        return f"error: unknown tool {name}"

    # -- public API ---------------------------------------------------------

    def handle(self, text: str) -> str:
        """Process one user utterance; returns the assistant's final text."""
        try:
            client = self._get_client()
        except RuntimeError as exc:
            return str(exc)
        rollback = len(self._history)
        self._history.append({"role": "user", "content": text})
        try:
            for _ in range(_MAX_TOOL_ROUNDS):
                response = client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=_SYSTEM,
                    messages=self._history,
                    tools=_TOOLS,
                )
                content = [block.model_dump() for block in response.content]
                self._history.append({"role": "assistant", "content": content})
                if response.stop_reason != "tool_use":
                    break
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": self._dispatch(block.name, dict(block.input)),
                            }
                        )
                self._history.append({"role": "user", "content": results})
            else:
                return "stopped: too many tool calls in one request"
        except Exception as exc:
            # Drop the partial exchange: a dangling tool_use without its
            # tool_result would poison every later request.
            del self._history[rollback:]
            return f"error talking to Claude: {exc}"
        finally:
            # Trim old turns, but only cut at a plain user message so we never
            # orphan a tool_use/tool_result pair.
            while len(self._history) > _MAX_HISTORY:
                cut = next(
                    (
                        i
                        for i, m in enumerate(self._history[1:], start=1)
                        if m["role"] == "user" and isinstance(m["content"], str)
                    ),
                    None,
                )
                if cut is None:
                    break
                del self._history[:cut]
        texts = [b["text"] for b in self._history[-1]["content"] if b.get("type") == "text"]
        return "\n".join(texts) or "(no reply)"
