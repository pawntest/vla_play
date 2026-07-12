"""Remote-teleop loopback tests: receiver + client protocol + TELEOP mode."""

import json
import socket
import time

import numpy as np
import pytest

from so101_tool.config import AppConfig
from so101_tool.control.commands import Mode, SetMode
from so101_tool.control.loop import ControlLoop
from so101_tool.kinematics import Kinematics
from so101_tool.robot.sim import SimBackend
from so101_tool.teleop.receiver import TeleopReceiver


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _connect(rx: TeleopReceiver, token: str | None = None) -> socket.socket:
    sock = socket.create_connection(("127.0.0.1", rx.port), timeout=5.0)
    sock.sendall((json.dumps({"token": token or rx.token}) + "\n").encode())
    reply = json.loads(sock.makefile("rb").readline(4096))
    return sock, reply


def test_token_required():
    rx = TeleopReceiver(port=_free_port())
    try:
        sock, reply = _connect(rx, token="wrong")
        assert reply == {"ok": False, "error": "bad token"}
        sock.close()
        assert rx.step(None) is None  # nothing accepted
    finally:
        rx.close()


def test_stream_drives_teleop_mode():
    rx = TeleopReceiver(port=_free_port())
    robot = SimBackend()
    robot.connect()
    loop = ControlLoop(robot, Kinematics(), AppConfig(control_hz=200.0))
    loop.set_teleop_source(rx)
    loop.start()
    try:
        mode = SetMode(mode=Mode.TELEOP)
        loop.commands.put(mode)
        assert mode.wait(2.0) and mode.ok

        sock, reply = _connect(rx)
        assert reply == {"ok": True}
        target = [0.3, -0.2, 0.4, 0.1, 0.0]
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            sock.sendall((json.dumps({"q": target, "gripper": 0.6}) + "\n").encode())
            time.sleep(0.02)
            snap = loop.snapshot()
            if (snap is not None and np.max(np.abs(snap.q - target)) < 0.03
                    and abs(snap.gripper - 0.6) < 0.05):
                break
        snap = loop.snapshot()
        assert np.max(np.abs(snap.q - target)) < 0.03, snap.q
        assert abs(snap.gripper - 0.6) < 0.05
        assert rx.fresh

        # stale stream -> hold (no new writes), robot stays put
        sock.close()
        time.sleep(rx.stale_timeout + 0.2)
        q_hold = loop.snapshot().q.copy()
        time.sleep(0.2)
        assert np.allclose(loop.snapshot().q, q_hold)
        assert not rx.fresh
    finally:
        loop.stop()
        loop.join(timeout=2.0)
        robot.disconnect()
        rx.close()


def test_malformed_and_out_of_range_frames_are_safe():
    rx = TeleopReceiver(port=_free_port())
    try:
        sock, reply = _connect(rx)
        assert reply == {"ok": True}
        sock.sendall(b"not json\n")
        sock.sendall(json.dumps({"q": [1, 2]}).encode() + b"\n")  # wrong shape
        sock.sendall(json.dumps({"q": [float("nan")] * 5, "gripper": 0}).encode() + b"\n")
        sock.sendall(json.dumps({"q": [99, -99, 99, -99, 99], "gripper": 7}).encode() + b"\n")
        time.sleep(0.3)
        result = rx.step(None)  # only the last (clampable) frame is accepted
        assert result is not None
        q, g = result
        from so101_tool.config import ARM_LIMITS_HI, ARM_LIMITS_LO

        assert np.all(q <= ARM_LIMITS_HI + 1e-9) and np.all(q >= ARM_LIMITS_LO - 1e-9)
        assert g == 1.0
        sock.close()
    finally:
        rx.close()


def test_never_binds_public_interfaces():
    with pytest.raises(ValueError, match="127.0.0.1"):
        TeleopReceiver(port=_free_port(), bind="0.0.0.0")


def test_mode_refused_without_receiver():
    robot = SimBackend()
    robot.connect()
    loop = ControlLoop(robot, Kinematics(), AppConfig(control_hz=200.0))
    loop.start()
    try:
        cmd = SetMode(mode=Mode.TELEOP)
        loop.commands.put(cmd)
        assert cmd.wait(2.0)
        assert "teleop receiver" in (cmd.error or "")
    finally:
        loop.stop()
        loop.join(timeout=2.0)
        robot.disconnect()
