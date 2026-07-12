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
    reader = _SineSource() if source == "sine" else _LeaderSource(serial_port, robot_id)
    print(f"teleop source: {source} → {connect} at {hz:.0f} Hz (Ctrl-C to stop)")
    try:
        while True:  # outer reconnect loop
            try:
                with socket.create_connection((host, int(port)), timeout=5.0) as sock:
                    sock.sendall((json.dumps({"token": token}) + "\n").encode())
                    reply = json.loads(sock.makefile("rb").readline(4096))
                    if not reply.get("ok"):
                        raise SystemExit(f"server refused the connection: {reply}")
                    print("connected — streaming")
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
