"""Remote-teleop receiver: a leader arm on the OPERATOR'S machine drives this
(possibly remote) app through an SSH tunnel.

Threat model / security posture:
  - Binds to 127.0.0.1 ONLY. The socket is never reachable from the network;
    the operator brings it to their laptop with standard SSH port forwarding
    (``ssh -L``), so transport encryption and authentication are SSH's.
  - A per-session random token is required in the handshake. Even another
    local user on a shared container cannot inject motion without it.
  - Fixed newline-delimited JSON schema, 4 KiB line cap, numeric validation,
    joint-limit clamping. No dynamic evaluation of any kind.
  - Targets flow through the normal control-loop safety filter (velocity
    clamp, limits) and stop within `stale_timeout` when the stream dies;
    the e-stop always wins.

Protocol (newline-delimited JSON, full duplex on one socket):
    client -> server  {"token": "<token>"}                 # handshake -> {"ok": true}
    client -> server  {"q": [j1..j5 rad], "gripper": 0..1} # measured state, 10-100 Hz
    server -> client  {"target_q": [...], "gripper": g}    # targets for a local follower
                                                            # (Mujoco→実機 / 双方向リンク)
"""

from __future__ import annotations

import json
import secrets
import socket
import threading
import time

import numpy as np

from ..config import ARM_LIMITS_HI, ARM_LIMITS_LO
from ..robot.base import RobotInterface, RobotState

_MAX_LINE = 4096


class TeleopReceiver:
    """Accepts one authenticated teleop stream and exposes it as a target
    source for the control loop (``step()`` mirrors PolicyRunner's contract)."""

    def __init__(self, port: int = 8765, token: str | None = None,
                 bind: str = "127.0.0.1", stale_timeout: float = 0.5):
        if bind != "127.0.0.1":
            raise ValueError(
                "TeleopReceiver only binds to 127.0.0.1 by design — reach it "
                "through an SSH tunnel (ssh -L), never by exposing the port."
            )
        self.port = port
        self.token = token or secrets.token_urlsafe(16)
        self.stale_timeout = stale_timeout
        self._latest: tuple[np.ndarray, float, float] | None = None  # (q, gripper, t)
        self._lock = threading.Lock()
        self._conn: socket.socket | None = None  # active authenticated connection
        self._server = socket.create_server((bind, port))
        self._server.settimeout(0.5)
        self._run = True
        self.status = "waiting for connection"
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="so101-teleop-rx")
        self._thread.start()

    # -- target source (control loop, TELEOP mode) ---------------------------

    def step(self, state) -> tuple[np.ndarray, float] | None:
        """Latest leader target, or None (hold) when no fresh data."""
        with self._lock:
            if self._latest is None:
                return None
            q, g, t = self._latest
        if time.monotonic() - t > self.stale_timeout:
            return None
        return q.copy(), g

    def reset(self) -> None:  # target-source contract
        pass

    def close(self) -> None:
        self._run = False
        try:
            self._server.close()
        except OSError:
            pass

    # -- server internals ---------------------------------------------------------

    def _serve(self) -> None:
        while self._run:
            try:
                conn, addr = self._server.accept()
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            try:
                self._handle(conn)
            except Exception:
                pass
            finally:
                with self._lock:
                    if self._conn is conn:
                        self._conn = None
                try:
                    conn.close()
                except OSError:
                    pass
                self.status = "waiting for connection"

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(5.0)
        reader = conn.makefile("rb")
        hello = reader.readline(_MAX_LINE)
        try:
            handshake = json.loads(hello)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not secrets.compare_digest(str(handshake.get("token", "")), self.token):
            conn.sendall(b'{"ok": false, "error": "bad token"}\n')
            self.status = "rejected connection (bad token)"
            return
        conn.sendall(b'{"ok": true}\n')
        with self._lock:
            self._conn = conn
        self.status = "leader connected"
        conn.settimeout(2.0)
        while self._run:
            line = reader.readline(_MAX_LINE)
            if not line:
                return
            try:
                msg = json.loads(line)
                q = np.asarray(msg["q"], dtype=float)
                g = float(msg.get("gripper", 0.0))
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError,
                    TypeError, ValueError):
                continue  # malformed frame: ignore, never raise
            if q.shape != (5,) or not np.all(np.isfinite(q)) or not np.isfinite(g):
                continue
            q = np.clip(q, ARM_LIMITS_LO, ARM_LIMITS_HI)
            with self._lock:
                self._latest = (q, float(np.clip(g, 0.0, 1.0)), time.monotonic())

    def send_targets(self, q: np.ndarray, gripper: float) -> bool:
        """Push a target frame to the connected operator client (full duplex).
        Returns False when no client is connected. Never raises."""
        with self._lock:
            conn = self._conn
        if conn is None:
            return False
        frame = {"target_q": [round(float(v), 5) for v in q],
                 "gripper": round(float(gripper), 4)}
        try:
            conn.sendall((json.dumps(frame) + "\n").encode())
            return True
        except OSError:
            return False

    @property
    def fresh(self) -> bool:
        with self._lock:
            latest = self._latest
        return latest is not None and time.monotonic() - latest[2] <= self.stale_timeout


class RemoteArmBackend(RobotInterface):
    """The operator's arm across the SSH tunnel, as a RobotInterface.

    Reads = the client's measured-joint stream; writes = target frames pushed
    back down the same socket (the client applies them to a local follower).
    Used as the "real" side of a LinkedBackend for remote 実機↔Mujoco links.
    """

    def __init__(self, receiver: TeleopReceiver):
        self.rx = receiver
        self._last: tuple[np.ndarray, float] | None = None

    def connect(self) -> None:
        pass  # the receiver accepts the client whenever it dials in

    def disconnect(self) -> None:
        pass  # receiver lifecycle is owned by the session

    def read_state(self) -> RobotState:
        result = self.rx.step(None)
        if result is not None:
            self._last = result
        if self._last is None:
            # nothing received yet: report home so the sim stays put
            return RobotState(q=np.zeros(5), gripper=0.0, t=time.monotonic(),
                              connected=False)
        q, g = self._last
        return RobotState(q=q.copy(), gripper=g, t=time.monotonic(),
                          connected=self.rx.fresh)

    def write_targets(self, q, gripper: float) -> None:
        self.rx.send_targets(q, gripper)

    @property
    def is_real(self) -> bool:
        return True
