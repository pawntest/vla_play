"""NLAgent tests with a scripted fake Anthropic client and a fake control loop."""

import types

import numpy as np

from so101_tool.control.commands import JogTool, Mode, MoveL, SetGripper, SetMode
from so101_tool.nl.agent import NLAgent


class FakeLoop:
    """Records commands and finishes them immediately."""

    def __init__(self, fail_with: str | None = None):
        self.received = []
        self.fail_with = fail_with
        self.commands = self  # quacks like a queue

    def put(self, cmd):
        self.received.append(cmd)
        if isinstance(cmd, SetMode):
            cmd.finish()
        else:
            cmd.finish(error=self.fail_with)

    def snapshot(self):
        return types.SimpleNamespace(
            mode=Mode.RULE,
            q=np.zeros(5),
            gripper=0.5,
            tcp_position=np.array([0.391, 0.0, 0.246]),
            tcp_wxyz=np.array([1.0, 0, 0, 0]),
            estop=False,
        )


def _block(**kw):
    return types.SimpleNamespace(**kw, model_dump=lambda **_: dict(kw))


class FakeClient:
    """Yields scripted responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


def _tool_use_response(name, args, tool_id="tu_1"):
    return types.SimpleNamespace(
        stop_reason="tool_use",
        content=[_block(type="tool_use", id=tool_id, name=name, input=args)],
    )


def _text_response(text):
    return types.SimpleNamespace(
        stop_reason="end_turn", content=[_block(type="text", text=text)]
    )


def test_move_forward_5cm():
    loop = FakeLoop()
    client = FakeClient(
        [
            _tool_use_response("get_state", {}),
            _tool_use_response("move_tool_relative", {"dx_mm": 50, "dy_mm": 0, "dz_mm": 0}),
            _text_response("Moved 5 cm forward."),
        ]
    )
    agent = NLAgent(loop, kinematics=None, client=client)
    reply = agent.handle("move the gripper 5cm forward")
    assert reply == "Moved 5 cm forward."
    jogs = [c for c in loop.received if isinstance(c, JogTool)]
    assert len(jogs) == 1
    assert np.allclose(jogs[0].dpos, [0.05, 0, 0])
    assert jogs[0].frame == "tool"
    # tool_result fed back for both tool calls
    tool_results = [
        m for m in client.requests[-1]["messages"] if isinstance(m["content"], list)
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_results) == 2


def test_move_to_point_and_gripper():
    loop = FakeLoop()
    client = FakeClient(
        [
            _tool_use_response("move_to_point", {"x_mm": 300, "y_mm": 0, "z_mm": 150}),
            _tool_use_response("set_gripper", {"percent": 0}, tool_id="tu_2"),
            _text_response("Done."),
        ]
    )
    agent = NLAgent(loop, kinematics=None, client=client)
    assert agent.handle("go to x=300 z=150 and close the gripper") == "Done."
    moves = [c for c in loop.received if isinstance(c, MoveL)]
    grips = [c for c in loop.received if isinstance(c, SetGripper)]
    assert np.allclose(moves[0].position, [0.3, 0.0, 0.15]) and moves[0].wxyz is None
    assert grips[0].fraction == 0.0


def test_failed_motion_reported_to_model():
    loop = FakeLoop(fail_with="timeout")
    client = FakeClient(
        [
            _tool_use_response("move_to_point", {"x_mm": 300, "y_mm": 0, "z_mm": 150}),
            _text_response("The motion failed."),
        ]
    )
    agent = NLAgent(loop, kinematics=None, client=client)
    agent.handle("go somewhere")
    last_messages = client.requests[-1]["messages"]
    result_blocks = [
        b for m in last_messages if isinstance(m["content"], list) for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    assert any("timeout" in b["content"] for b in result_blocks)


def test_oversized_jog_refused():
    loop = FakeLoop()
    client = FakeClient(
        [
            _tool_use_response("move_tool_relative", {"dx_mm": 500, "dy_mm": 0, "dz_mm": 0}),
            _text_response("Refused."),
        ]
    )
    agent = NLAgent(loop, kinematics=None, client=client)
    agent.handle("lunge forward half a meter")
    assert not [c for c in loop.received if isinstance(c, JogTool)]


def test_api_error_rolls_back_history():
    loop = FakeLoop()

    class ExplodingClient(FakeClient):
        def create(self, **kwargs):
            raise ConnectionError("api down")

    agent = NLAgent(loop, kinematics=None, client=ExplodingClient([]))
    reply = agent.handle("hello")
    assert "api down" in reply
    assert agent._history == []
