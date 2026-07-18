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
    linked.connect()
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


def test_to_real_tolerates_real_failure_but_both_does_not():
    class Broken(SimBackend):
        def write_targets(self, q, gripper):
            raise OSError("serial port gone")

    clock = Clock()
    jm = JointMap(vmax_rad_s=np.full(5, 1000.0))
    linked = LinkedBackend(SimBackend(jm, clock=clock), Broken(jm, clock=clock),
                           link="to_real")
    linked.connect()
    linked.write_targets(Q, 0.5)  # must not raise: sim keeps working
    clock.t += 0.05
    assert np.allclose(linked.read_state().q, Q, atol=1e-6)
    linked.set_link("both")
    with pytest.raises(OSError):  # in both, the real arm IS the robot
        linked.write_targets(Q, 0.5)


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
