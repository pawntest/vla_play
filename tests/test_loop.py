import time

import numpy as np
import pytest

from so101_tool.config import AppConfig
from so101_tool.control.commands import Home, JogTool, Mode, MoveJ, MoveL, SetMode, Stop
from so101_tool.control.loop import ControlLoop
from so101_tool.kinematics import Kinematics
from so101_tool.robot.base import RobotInterface, RobotState


class FakeRobot(RobotInterface):
    """Instant-tracking robot: reads back the last written target."""

    def __init__(self):
        self.q = np.zeros(5)
        self.gripper = 0.0
        self.writes = 0

    def connect(self):
        pass

    def disconnect(self):
        pass

    def read_state(self):
        return RobotState(q=self.q.copy(), gripper=self.gripper, t=time.monotonic())

    def write_targets(self, q, gripper):
        self.q = np.asarray(q, dtype=float).copy()
        self.gripper = float(gripper)
        self.writes += 1

    @property
    def is_real(self):
        return False


@pytest.fixture
def kin():
    """Loop-owned instance. Kinematics is not thread-safe: tests must NOT call
    this instance while the loop runs — use check_kin for assertions."""
    return Kinematics()


@pytest.fixture
def check_kin():
    return Kinematics()


@pytest.fixture
def stack(kin):
    robot = FakeRobot()
    loop = ControlLoop(robot, kin, AppConfig(control_hz=200.0))
    loop.start()
    cmd = SetMode(mode=Mode.RULE)
    loop.commands.put(cmd)
    assert cmd.wait(2.0) and cmd.ok
    yield robot, loop
    loop.stop()
    loop.join(timeout=2.0)


def test_movej_completes(stack):
    robot, loop = stack
    target = np.array([0.4, -0.3, 0.5, 0.2, 0.1])
    cmd = MoveJ(q=target, gripper=0.6, speed=1.0)
    loop.commands.put(cmd)
    assert cmd.wait(10.0), "MoveJ did not finish"
    assert cmd.ok, cmd.error
    assert np.max(np.abs(robot.q - target)) < 0.03
    assert abs(robot.gripper - 0.6) < 0.03


def test_movel_reaches_position(stack, check_kin):
    robot, loop = stack
    target_pos = check_kin.fk(np.array([0.3, -0.4, 0.6, 0.1, 0.0])).position
    cmd = MoveL(position=target_pos, wxyz=None, speed=1.0)
    loop.commands.put(cmd)
    assert cmd.wait(15.0), "MoveL did not finish"
    assert cmd.ok, cmd.error
    assert np.linalg.norm(check_kin.fk(robot.q).position - target_pos) < 5e-3


def test_jog_tool_moves_tcp(stack, check_kin):
    robot, loop = stack
    p0 = check_kin.fk(robot.q).position.copy()
    cmd = JogTool(dpos=np.array([0.0, 0.0, 0.03]), frame="base", speed=1.0)
    loop.commands.put(cmd)
    assert cmd.wait(15.0) and cmd.ok, cmd.error
    p1 = check_kin.fk(robot.q).position
    assert np.linalg.norm(p1 - (p0 + [0, 0, 0.03])) < 6e-3


def test_preemption(stack):
    robot, loop = stack
    slow = MoveJ(q=np.array([1.5, 0, 0, 0, 0]), speed=0.1)
    loop.commands.put(slow)
    time.sleep(0.1)
    fast = MoveJ(q=np.zeros(5), speed=1.0)
    loop.commands.put(fast)
    assert slow.wait(2.0)
    assert slow.error == "preempted"
    assert fast.wait(5.0) and fast.ok


def test_stop_aborts(stack):
    robot, loop = stack
    mv = MoveJ(q=np.array([1.5, 0, 0, 0, 0]), speed=0.1)
    loop.commands.put(mv)
    time.sleep(0.1)
    loop.commands.put(Stop())
    assert mv.wait(2.0)
    assert mv.error == "stopped"


def test_estop_blocks_writes(stack):
    robot, loop = stack
    mv = MoveJ(q=np.array([1.0, 0, 0, 0, 0]), speed=0.3)
    loop.commands.put(mv)
    time.sleep(0.1)
    loop.estop()
    assert mv.wait(2.0)
    assert mv.error == "estop"
    writes_at_estop = robot.writes
    q_at_estop = robot.q.copy()
    time.sleep(0.2)
    assert robot.writes == writes_at_estop
    assert np.array_equal(robot.q, q_at_estop)
    # motion refused while latched
    mv2 = MoveJ(q=np.zeros(5))
    loop.commands.put(mv2)
    assert mv2.wait(2.0)
    assert mv2.error == "estop"
    loop.reset_estop()


def test_motion_refused_outside_rule(kin):
    robot, loop = FakeRobot(), None
    loop = ControlLoop(robot, kin, AppConfig(control_hz=200.0))
    loop.start()
    try:
        mv = MoveJ(q=np.zeros(5))
        loop.commands.put(mv)
        assert mv.wait(2.0)
        assert "not in RULE mode" in (mv.error or "")
        assert robot.writes == 0
    finally:
        loop.stop()
        loop.join(timeout=2.0)


def test_shutdown_finishes_queue(kin):
    robot = FakeRobot()
    loop = ControlLoop(robot, kin, AppConfig(control_hz=200.0))
    loop.start()
    loop.stop()
    loop.join(timeout=2.0)
    cmd = Home()
    loop.commands.put(cmd)
    # queued after shutdown: drained by the shutdown path already run, so put
    # before stop in a second scenario instead
    loop2 = ControlLoop(FakeRobot(), kin, AppConfig(control_hz=200.0))
    loop2.start()
    hanging = SetMode(mode=Mode.RULE)
    loop2.commands.put(hanging)
    loop2.stop()
    loop2.join(timeout=2.0)
    assert hanging.wait(0.5)


def test_policy_mode_requires_runner(stack):
    robot, loop = stack
    cmd = SetMode(mode=Mode.POLICY)
    loop.commands.put(cmd)
    assert cmd.wait(2.0)
    assert "no policy" in (cmd.error or "")
