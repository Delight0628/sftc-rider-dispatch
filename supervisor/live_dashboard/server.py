"""server — Live 调度大屏 HTTP 服务

用法：
    python supervisor/live_dashboard/server.py [--port 8788] [--seed 20261020]
    # 打开 http://127.0.0.1:8788

接口：
    GET  /          大屏 HTML（注入当次推演数据，单文件可另存离线看）
    GET  /api/pack  推演数据 JSON（frames/gantt/events/summary/meta）
    POST /api/regen 换 seed 重新推演（返回新 pack）

设计：首次请求触发一次完整推演并缓存；"重新推演"按钮 POST /api/regen。
输出语义 = 每骑手实时订单 list（紧迫 x 偏好 外部加权融合，project.md §7.5）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from supervisor.live_dashboard.sim_stream import capture_episode  # noqa: E402

STATIC_HTML = os.path.join(os.path.dirname(__file__), "static", "index.html")

_LOCK = threading.Lock()
_PACK = {"data": None, "seed": None}


def _regen(seed: int):
    data = capture_episode(seed=seed)
    with _LOCK:
        _PACK["data"] = data
        _PACK["seed"] = seed
    return data


def _ensure():
    with _LOCK:
        if _PACK["data"] is not None:
            return _PACK["data"]
    return _regen(_PACK["seed"] or 20261020)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path.startswith("/api/pack"):
                data = _ensure()
                self._send(200, json.dumps(data, ensure_ascii=False).encode("utf-8"))
                return
            # 大屏 HTML：注入数据 → 单文件（可另存离线复看）
            data = _ensure()
            with open(STATIC_HTML, encoding="utf-8") as f:
                html = f.read()
            payload = json.dumps(data, ensure_ascii=False)
            payload = payload.replace("</", "<\\/")  # 防 </script> 截断
            html = html.replace("/*__DATA__*/null", payload)
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        try:
            if self.path.startswith("/api/regen"):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                seed = int(body.get("seed", 20261020))
                data = _regen(seed)
                self._send(200, json.dumps({"ok": True, "seed": seed,
                                            "meta": data["meta"]},
                                           ensure_ascii=False).encode("utf-8"))
                return
            self._send(404, b'{"error":"not found"}')
        except Exception as e:  # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    _PACK["seed"] = args.seed
    print(f"🚦 Live 调度大屏 http://127.0.0.1:{args.port}  (seed={args.seed})")
    print("   首次加载推演约 15-40s（400 步仿真+16 维边效用网络），随后秒开")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
