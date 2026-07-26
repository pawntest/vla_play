"""LinkedBackend (実機↔Mujoco link) mode semantics + RemoteArmBackend duplex."""

import json
import socket
import time

import numpy as np
import pytest

from so101_tool.config import JointMap
from so101_tool.robot.linked import LINK_MODES, LinkedBackend
from so101_tool.robot.sim import SimBackend
from so101_tool.teleop.receiver import RemoteArmBackend, TeleopReceiver


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _pair(link: str) -> tuple[LinkedBackend, SimBackend, SimBackend, Clock]:
    """A LinkedBackend over two SimBackends fast enough to converge in one dt."""
    clock = Clock()
    jm = JointMap(vmax_rad_s=np.full(5, 1000.0))
    sim = SimBackend(jm, clock=clock)
    real = SimBackend(jm, clock=clock)  # stands in for the real arm
    linked = LinkedBackend(sim, real, link=link)
    linked.connect()  # real side connects in a background thread
    assert linked.wait_real(5.0), linked.real_status
    linked.read_state()  # first read marks the real side live
    return linked, sim, real, clock


Q = np.array([0.3, -0.2, 0.4, 0.1, -0.5])


def test_link_mode_validation():
    clock = Clock()
    sim, real = SimBackend(clock=clock), SimBackend(clock=clock)
    with pytest.raises(ValueError):
        LinkedBackend(sim, real, link="sideways")
    linked = LinkedBackend(sim, real, link="both")
    with pytest.raises(ValueError):
        linked.set_link("nope")
    for mode in LINK_MODES:
        linked.set_link(mode)
        assert linked.link == mode
    assert linked.is_real


def test_both_commands_reach_real_and_sim_follows():
    linked, sim, real, clock = _pair("both")
    linked.write_targets(Q, 0.7)
    # several ticks: the gripper rate limit (2.0/s) needs ~0.35 s to reach 0.7,
    # and the sim chases the real arm one read behind
    for _ in range(6):
        clock.t += 0.1
        state = linked.read_state()  # reads real, sim chases real
    assert np.allclose(state.q, Q, atol=1e-6)  # reported state = real arm
    assert abs(state.gripper - 0.7) < 1e-6
    sim_state = sim.read_state()  # sim inherited the pose by following real
    assert np.allclose(sim_state.q, Q, atol=1e-6)


def test_to_sim_drops_writes_and_mirrors_real():
    linked, sim, real, clock = _pair("to_sim")
    linked.write_targets(Q, 0.9)  # must be dropped: real arm is the source
    clock.t += 0.05
    assert np.allclose(linked.read_state().q, 0.0)
    # the operator moves the real arm (externally / by hand)
    real.write_targets(Q, 0.4)
    clock.t += 0.05
    state = linked.read_state()
    clock.t += 0.05
    state = linked.read_state()
    assert np.allclose(state.q, Q, atol=1e-6)
    assert np.allclose(sim.read_state().q, Q, atol=1e-6)  # sim mirrored it


def test_to_real_reads_sim_and_shadows_real():
    linked, sim, real, clock = _pair("to_real")
    linked.write_targets(Q, 0.3)
    clock.t += 0.05
    state = linked.read_state()
    assert np.allclose(state.q, Q, atol=1e-6)  # state comes from the sim
    clock.t += 0.05
    assert np.allclose(real.read_state().q, Q, atol=1e-6)  # real shadowed it


def test_real_write_failure_degrades_but_never_raises():
    class Broken(SimBackend):
        def write_targets(self, q, gripper):
            raise OSError("serial port gone")

    clock = Clock()
    jm = JointMap(vmax_rad_s=np.full(5, 1000.0))
    linked = LinkedBackend(SimBackend(jm, clock=clock), Broken(jm, clock=clock),
                           link="to_real")
    linked.connect()
    assert linked.wait_real(5.0)
    linked.write_targets(Q, 0.5)  # must not raise: sim keeps working
    clock.t += 0.05
    assert np.allclose(linked.read_state().q, Q, atol=1e-6)
    assert not linked.real_ok and linked.real_status.startswith("error")

    # both: same failure degrades to sim-only instead of crashing the loop
    linked2 = LinkedBackend(SimBackend(jm, clock=clock), Broken(jm, clock=clock),
                            link="both")
    linked2.connect()
    assert linked2.wait_real(5.0)
    linked2.read_state()  # marks the real side live
    linked2.write_targets(Q, 0.5)  # real write fails -> sim still gets the target
    clock.t += 0.05
    assert np.allclose(linked2.read_state().q, Q, atol=1e-6)
    assert linked2.real_status.startswith("error")


def test_real_connect_failure_keeps_sim_alive():
    """The reported bug: a real arm that fails to connect must not take the
    app down — the sim keeps working and the status says what went wrong."""

    class NoPort(SimBackend):
        def connect(self):
            raise RuntimeError("could not open /dev/ttyACM0")

    clock = Clock()
    jm = JointMap(vmax_rad_s=np.full(5, 1000.0))
    linked = LinkedBackend(SimBackend(jm, clock=clock), NoPort(jm, clock=clock),
                           link="both")
    linked.connect()  # must not raise
    assert not linked.wait_real(5.0)
    assert "ttyACM0" in linked.real_status
    linked.write_targets(Q, 0.5)  # degraded: behaves like the plain sim
    clock.t += 0.05
    state = linked.read_state()
    assert np.allclose(state.q, Q, atol=1e-6)


def test_reconnect_real_recovers_after_failure():
    """Fix the power/port, press reconnect — no app restart needed."""

    class FlakyReal(SimBackend):
        attempts = 0

        def connect(self):
            FlakyReal.attempts += 1
            if FlakyReal.attempts == 1:
                raise RuntimeError("motor check failed: found {}")
            super().connect()

    clock = Clock()
    jm = JointMap(vmax_rad_s=np.full(5, 1000.0))
    linked = LinkedBackend(SimBackend(jm, clock=clock), FlakyReal(jm, clock=clock),
                           link="both")
    linked.connect()
    assert not linked.wait_real(5.0)
    assert "motor check failed" in linked.real_status
    linked.reconnect_real()  # user fixed the hardware and pressed the button
    assert linked.wait_real(5.0), linked.real_status
    linked.read_state()
    linked.write_targets(Q, 0.5)
    clock.t += 0.05
    for _ in range(2):
        state = linked.read_state()
        clock.t += 0.05
    assert np.allclose(state.q, Q, atol=1e-6)


def test_real_error_hint_covers_common_failures():
    from so101_tool.viz.panel import _real_error_hint

    motor_err = ("FeetechMotorsBus motor check failed on port '/dev/ttyACM0': "
                 "Missing motor IDs ... Full found motor list (id: model_number): {}")
    assert "電源" in _real_error_hint(motor_err)
    assert "権限" in _real_error_hint("Permission denied: '/dev/ttyACM0'")
    assert "--port" in _real_error_hint("[Errno 2] No such file or directory")
    assert _real_error_hint("some novel failure") == ""


def test_camera_overlay_server_serves_page_and_frames():
    from urllib.request import urlopen

    from so101_tool.viz.overlay_server import CameraOverlayServer

    srv = CameraOverlayServer(viser_port=8080, port=0, bind="127.0.0.1")
    try:
        base = f"http://127.0.0.1:{srv.port}"
        page = urlopen(f"{base}/", timeout=5).read().decode()
        assert "iframe" in page and "8080" in page  # wrapper embeds viser
        assert json.loads(urlopen(f"{base}/cams", timeout=5).read()) == []

        srv.set_frames({"front": np.zeros((60, 80, 3), dtype=np.uint8),
                        "top": np.full((60, 80, 3), 128, dtype=np.uint8)})
        assert json.loads(urlopen(f"{base}/cams", timeout=5).read()) == ["front", "top"]
        jpeg = urlopen(f"{base}/cam/front", timeout=5).read()
        assert jpeg[:2] == b"\xff\xd8"  # JPEG magic

        srv.enabled = False  # 📷 toggle / mode off hides all tiles
        assert json.loads(urlopen(f"{base}/cams", timeout=5).read()) == []
    finally:
        srv.close()


def test_placeholder_real_state_does_not_drag_sim_home():
    """A remote arm reports connected=False until its stream starts; the sim
    must stay the source (not get pulled to the placeholder home pose)."""

    class NotStreaming(SimBackend):
        def read_state(self):
            state = super().read_state()
            state.connected = False
            return state

    clock = Clock()
    jm = JointMap(vmax_rad_s=np.full(5, 1000.0))
    sim = SimBackend(jm, clock=clock)
    linked = LinkedBackend(sim, NotStreaming(jm, clock=clock), link="both")
    linked.connect()
    assert linked.wait_real(5.0)
    linked.write_targets(Q, 0.5)  # no live stream -> the sim takes the command
    clock.t += 0.05
    assert np.allclose(linked.read_state().q, Q, atol=1e-6)
    assert not linked.real_live


def test_session_builds_linked_backend_with_remote_real_side():
    from so101_tool.api import Session

    session = Session(link="both", teleop_port=_free_port())
    try:
        assert isinstance(session.robot, LinkedBackend)
        assert isinstance(session.robot.real, RemoteArmBackend)
        assert session.teleop_rx is not None  # printed/closed by the app
        session.robot.read_state()  # no client yet: sim mirrors home, no raise
    finally:
        session.close()


# -- RemoteArmBackend: the operator's arm across the tunnel, full duplex --------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_remote_arm_backend_duplex():
    rx = TeleopReceiver(port=_free_port())
    arm = RemoteArmBackend(rx)
    try:
        # before any client: home pose, marked disconnected
        state = arm.read_state()
        assert not state.connected and np.allclose(state.q, 0.0)

        with socket.create_connection(("127.0.0.1", rx.port), timeout=5.0) as sock:
            sock.sendall((json.dumps({"token": rx.token}) + "\n").encode())
            rfile = sock.makefile("rb")
            assert json.loads(rfile.readline(4096)) == {"ok": True}

            # client -> server: measured joints appear as read_state()
            sock.sendall((json.dumps({"q": list(Q), "gripper": 0.6}) + "\n").encode())
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not rx.fresh:
                time.sleep(0.02)
            state = arm.read_state()
            assert state.connected
            assert np.allclose(state.q, Q, atol=1e-5) and abs(state.gripper - 0.6) < 1e-5

            # server -> client: write_targets arrives as a target_q frame
            arm.write_targets(Q * 0.5, 0.25)
            msg = json.loads(rfile.readline(4096))
            assert np.allclose(msg["target_q"], Q * 0.5, atol=1e-4)
            assert abs(msg["gripper"] - 0.25) < 1e-4

        # stream gone -> stale: last pose held but marked disconnected
        time.sleep(rx.stale_timeout + 0.2)
        state = arm.read_state()
        assert not state.connected and np.allclose(state.q, Q, atol=1e-5)
        # once the server notices the disconnect (EOF or EPIPE on a later
        # send), send_targets reports False — and never raises
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and rx.send_targets(Q, 0.5):
            time.sleep(0.05)
        assert rx.send_targets(Q, 0.5) is False
    finally:
        rx.close()
