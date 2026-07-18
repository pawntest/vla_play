"""Teleop client: runs on the OPERATOR'S machine (where the leader arm is
plugged in) and streams joint targets to a (remote) so101-tool app.

Typical remote setup (GitHub Codespaces / NVIDIA Brev / any SSH box):

    remote$ so101-tool run --scenario examples/pick_cube.yaml --teleop
            # prints the 127.0.0.1 port and the session token

    laptop$ ssh -L 8765:localhost:8765 <remote>          # or the Codespaces
            # port forward UI — both keep the port private to you
    laptop$ so101-tool teleop-client --connect localhost:8765 \
            --token <printed token> --port /dev/ttyACM0 --robot-id my_leader

The stream then drives the remote app's TELEOP mode: sim preview, dataset
recording and (if the remote box has a follower) the real arm — all through
the normal safety filter. `--source sine` streams a synthetic trajectory for
testing the link without hardware.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time

import numpy as np

from ..config import ARM_JOINTS


class _SineSource:
    """Hardware-free test source: small synchronized sinusoids."""

    def __init__(self):
        self._t0 = time.monotonic()

    def read(self) -> tuple[list[float], float]:
        t = time.monotonic() - self._t0
        q = [0.4 * math.sin(0.5 * t + i) for i in range(5)]
        return q, 0.5 * (1 + math.sin(0.8 * t))


class _FollowerDevice:
    """Local SO-101 follower: streams its measured joints up AND applies
    target frames received from the server (Mujoco→実機 / 双方向 link)."""

    def __init__(self, port: str, robot_id: str):
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise RuntimeError(
                'follower teleop requires lerobot: pip install "so101-tool[real]" '
                "(Python >= 3.12)"
            ) from exc
        self._robot = SO101Follower(SO101FollowerConfig(
            port=port, id=robot_id, use_degrees=True,
            disable_torque_on_disconnect=True))
        self._robot.connect()

    def read(self) -> tuple[list[float], float]:
        obs = self._robot.get_observation()
        q = [math.radians(float(obs[f"{j}.pos"])) for j in ARM_JOINTS]
        return q, float(np.clip(obs["gripper.pos"] / 100.0, 0.0, 1.0))

    def apply(self, q: list[float], gripper: float) -> None:
        action = {f"{j}.pos": math.degrees(q[i]) for i, j in enumerate(ARM_JOINTS)}
        action["gripper.pos"] = float(np.clip(gripper, 0.0, 1.0) * 100.0)
        self._robot.send_action(action)

    def close(self) -> None:
        self._robot.disconnect()


class _LeaderSource:
    """Real SO-101 leader arm via lerobot (requires Python >= 3.12)."""

    def __init__(self, port: str, robot_id: str):
        try:
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        except ImportError as exc:
            raise RuntimeError(
                'leader teleop requires lerobot: pip install "so101-tool[real]" '
                "(Python >= 3.12)"
            ) from exc
        self._leader = SO101Leader(SO101LeaderConfig(port=port, id=robot_id, use_degrees=True))
        self._leader.connect()

    def read(self) -> tuple[list[float], float]:
        action = self._leader.get_action()
        q = [math.radians(float(action[f"{j}.pos"])) for j in ARM_JOINTS]
        gripper = float(np.clip(action["gripper.pos"] / 100.0, 0.0, 1.0))
        return q, gripper

    def close(self) -> None:
        self._leader.disconnect()


def run_teleop_client(connect: str, token: str, source: str = "leader",
                      serial_port: str = "/dev/ttyACM0", robot_id: str = "so101_leader",
                      hz: float = 50.0) -> None:
    """Stream leader joints to `connect` ("host:port") until Ctrl-C."""
    host, _, port = connect.partition(":")
    if source == "sine":
        reader = _SineSource()
    elif source == "follower":
        reader = _FollowerDevice(serial_port, robot_id)
    else:
        reader = _LeaderSource(serial_port, robot_id)
    applied = [0]

    def _apply_targets(f):
        """Reader thread: server -> client target frames (full duplex)."""
        while True:
            try:
                line = f.readline(4096)
            except TimeoutError:
                continue  # no targets lately (実機→シム link): keep listening
            except (OSError, ValueError):
                return  # socket closed — the outer loop reconnects
            if not line:
                return
            try:
                msg = json.loads(line)
                tq = msg.get("target_q")
                if tq is None or len(tq) != 5:
                    continue
                applied[0] += 1
                if isinstance(reader, _FollowerDevice):
                    reader.apply([float(v) for v in tq], float(msg.get("gripper", 0.0)))
            except Exception:
                continue
    print(f"teleop source: {source} → {connect} at {hz:.0f} Hz (Ctrl-C to stop)")
    try:
        while True:  # outer reconnect loop
            try:
                with socket.create_connection((host, int(port)), timeout=5.0) as sock:
                    sock.sendall((json.dumps({"token": token}) + "\n").encode())
                    rfile = sock.makefile("rb")
                    reply = json.loads(rfile.readline(4096))
                    if not reply.get("ok"):
                        raise SystemExit(f"server refused the connection: {reply}")
                    print("connected — streaming (full duplex)")
                    threading.Thread(target=_apply_targets, args=(rfile,),
                                     daemon=True).start()
                    period = 1.0 / hz
                    while True:
                        q, gripper = reader.read()
                        frame = {"q": [round(v, 5) for v in q], "gripper": round(gripper, 4)}
                        sock.sendall((json.dumps(frame) + "\n").encode())
                        time.sleep(period)
            except (ConnectionError, TimeoutError, OSError) as exc:
                print(f"link lost ({exc}); retrying in 2 s…")
                time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nteleop client stopped")
    finally:
        if hasattr(reader, "close"):
            reader.close()
