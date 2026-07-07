"""Full-stack headless smoke test: SimBackend + ControlLoop + viser view/panel."""

import socket
import time

import numpy as np
import viser

from so101_tool.config import AppConfig
from so101_tool.control.commands import Mode, MoveJ, SetMode
from so101_tool.control.loop import ControlLoop
from so101_tool.kinematics import Kinematics
from so101_tool.robot.sim import SimBackend
from so101_tool.viz.panel import ControlPanel
from so101_tool.viz.robot_view import RobotView


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_full_stack_boots_and_moves():
    config = AppConfig(control_hz=100.0)
    robot = SimBackend(config.joint_map)
    robot.connect()
    loop = ControlLoop(robot, Kinematics(), config)
    loop.start()
    server = viser.ViserServer(port=_free_port(), verbose=False)
    try:
        kin_viz = Kinematics()
        view = RobotView(server, kin_viz)
        panel = ControlPanel(server, loop, kin_viz, config, nl_agent=None)

        mode = SetMode(mode=Mode.RULE)
        loop.commands.put(mode)
        assert mode.wait(2.0) and mode.ok

        target = np.array([0.3, -0.2, 0.3, 0.1, 0.05])
        mv = MoveJ(q=target, speed=1.0)
        loop.commands.put(mv)

        deadline = time.monotonic() + 10.0
        last_snap = None
        while time.monotonic() < deadline and not mv.wait(0.05):
            last_snap = loop.snapshot()
            if last_snap is not None:
                view.sync(last_snap.q, last_snap.gripper)
                panel.update(last_snap)
        assert mv.ok, mv.error
        # The snapshot can lag the finish by a tick, and the sim keeps
        # tracking the final target: poll briefly for full convergence.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            snap = loop.snapshot()
            if snap is not None and np.max(np.abs(snap.q - target)) < 0.03:
                break
            time.sleep(0.02)
        assert snap is not None
        assert np.max(np.abs(snap.q - target)) < 0.03
        assert snap.tcp_position[2] > 0  # arm above the floor
    finally:
        loop.stop()
        loop.join(timeout=2.0)
        robot.disconnect()
        server.stop()
