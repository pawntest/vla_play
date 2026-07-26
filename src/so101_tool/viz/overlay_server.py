"""Zoom/Teams-style camera overlay: a real fixed DOM component on screen.

viser cannot host screen-anchored DOM, so this serves a tiny WRAPPER page that
embeds the viser UI in a full-viewport iframe and stacks the live camera tiles
in a fixed vertical column on top (top-left), with a 📷 toggle button. Open
THIS page instead of the raw viser URL.

Endpoints (stdlib http.server, daemon thread):
    GET /            wrapper page (iframe -> viser on the same host)
    GET /cams        JSON list of camera names ([] while disabled)
    GET /cam/<name>  latest JPEG frame for that camera

The render loop pushes frames with set_frames() (~2 Hz); the page polls
/cam/<name> every 500 ms. Read-only, images only — no control surface.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import numpy as np

_log = logging.getLogger("so101_tool.overlay")

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>so101-tool</title><style>
html,body{margin:0;height:100%;overflow:hidden;background:#111}
#viser{position:absolute;inset:0;border:0;width:100%;height:100%}
#cams{position:fixed;top:52px;left:12px;z-index:10;display:flex;
  flex-direction:column;gap:8px;width:236px;max-height:calc(100% - 64px);
  overflow-y:auto}
.tile{background:rgba(17,20,28,.78);border-radius:10px;padding:5px;
  backdrop-filter:blur(3px);box-shadow:0 2px 8px rgba(0,0,0,.35)}
.tile img{width:100%;display:block;border-radius:7px}
.tile .name{color:#fff;font:11px/1.6 sans-serif;padding:0 3px;opacity:.85}
#toggle{position:fixed;top:12px;left:12px;z-index:11;border:0;cursor:pointer;
  border-radius:9px;padding:6px 11px;font-size:15px;color:#fff;
  background:rgba(17,20,28,.78);backdrop-filter:blur(3px)}
</style></head><body>
<iframe id="viser"></iframe>
<button id="toggle" title="カメラ表示/非表示">📷</button>
<div id="cams"></div>
<script>
const VISER_PORT = __VISER_PORT__;
document.getElementById("viser").src =
  location.protocol + "//" + location.hostname + ":" + VISER_PORT;
const cams = document.getElementById("cams");
let names = [], shown = true;
document.getElementById("toggle").onclick = () => {
  shown = !shown; cams.style.display = shown ? "flex" : "none";
};
async function refreshList() {
  try {
    const list = await (await fetch("/cams")).json();
    if (JSON.stringify(list) === JSON.stringify(names)) return;
    names = list;
    cams.innerHTML = "";
    for (const n of names) {
      const tile = document.createElement("div"); tile.className = "tile";
      const img = document.createElement("img"); img.dataset.name = n;
      const label = document.createElement("div");
      label.className = "name"; label.textContent = "📷 " + n;
      tile.appendChild(img); tile.appendChild(label); cams.appendChild(tile);
    }
  } catch (e) {}
}
setInterval(refreshList, 2000); refreshList();
setInterval(() => {
  if (!shown) return;
  for (const img of cams.querySelectorAll("img"))
    img.src = "/cam/" + encodeURIComponent(img.dataset.name) + "?t=" + Date.now();
}, 500);
</script></body></html>
"""


class CameraOverlayServer:
    """Serves the wrapper page + latest camera JPEGs. Thread-safe setters."""

    def __init__(self, viser_port: int, port: int | None = None,
                 bind: str = "0.0.0.0"):
        self._lock = threading.Lock()
        self._jpegs: dict[str, bytes] = {}
        self.enabled = True
        page = _PAGE.replace("__VISER_PORT__", str(viser_port)).encode()

        overlay = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the console clean
                pass

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 (http.server API)
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", page)
                elif path == "/cams":
                    with overlay._lock:
                        names = sorted(overlay._jpegs) if overlay.enabled else []
                    self._send(200, "application/json", json.dumps(names).encode())
                elif path.startswith("/cam/"):
                    name = unquote(path[len("/cam/"):])
                    with overlay._lock:
                        jpeg = overlay._jpegs.get(name)
                    if jpeg is None:
                        self._send(404, "text/plain", b"no frame")
                    else:
                        self._send(200, "image/jpeg", jpeg)
                else:
                    self._send(404, "text/plain", b"not found")

        want = port if port is not None else viser_port + 1
        try:
            self._server = ThreadingHTTPServer((bind, want), Handler)
        except OSError:  # port taken (tests, parallel apps): pick any free one
            self._server = ThreadingHTTPServer((bind, 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True, name="so101-cam-overlay")
        self._thread.start()
        _log.info("camera overlay page on port %d", self.port)

    def set_frames(self, frames: dict[str, np.ndarray]) -> None:
        """Encode + publish the latest frames (called from the render loop)."""
        import imageio.v3 as iio

        encoded = {}
        for name, img in frames.items():
            if not isinstance(img, np.ndarray) or img.ndim != 3:
                continue
            try:
                encoded[name] = iio.imwrite("<bytes>", img.astype(np.uint8),
                                            extension=".jpg")
            except Exception:
                continue
        with self._lock:
            self._jpegs = encoded

    def close(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
