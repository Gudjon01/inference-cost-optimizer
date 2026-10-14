"""A gateway admin API and one engine on real local ports.

For the tests that cross a process boundary: the replay command run whole,
and the watchdog, which is a process of its own. The engine can be stopped
and started over HTTP, so a command launcher can drive it with one-liners.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = "test-admin-token"


class FakeInstall:
    def __init__(self, *, replicas=("r1", "r2"), original_seqs: int = 2, service_s: float = 0.08) -> None:
        self.drained = {name: False for name in replicas}
        self.admin_calls: list[str] = []
        self.engine_flags: str | None = f"--max-num-seqs {original_seqs}"  # None while the engine is down
        self.starts: list[str] = []
        self.broken_seqs: set[int] = set()
        self.service_s = service_s
        self._slots = threading.Semaphore(original_seqs)
        self._lock = threading.Lock()

        self._admin = _serve(self._admin_handler())
        self._engine = _serve(self._engine_handler())
        self.admin_url = f"http://127.0.0.1:{self._admin.server_port}"
        self.engine_url = f"http://127.0.0.1:{self._engine.server_port}"

    def close(self) -> None:
        for server in (self._admin, self._engine):
            server.shutdown()
            server.server_close()

    def launcher(self) -> dict:
        """A command launcher config whose commands start and stop this engine."""
        python = f'"{sys.executable}"'
        start = (
            "import sys, urllib.parse, urllib.request; "
            f"urllib.request.urlopen('{self.engine_url}/_start?flags=' + urllib.parse.quote(' '.join(sys.argv[1:])))"
        )
        stop = f"import urllib.request; urllib.request.urlopen('{self.engine_url}/_stop')"
        return {"kind": "command", "start": f'{python} -c "{start}" {{flags}}', "stop": f'{python} -c "{stop}"'}

    def _start(self, flags: str) -> None:
        parts = flags.split()
        seqs = int(parts[parts.index("--max-num-seqs") + 1]) if "--max-num-seqs" in parts else 1
        with self._lock:
            self.starts.append(flags)
            if seqs in self.broken_seqs:
                self.engine_flags = None  # it crashes on the way up
                return
            self._slots = threading.Semaphore(seqs)
            self.engine_flags = flags

    def _admin_handler(self):
        install = self

        class Handler(_Quiet):
            def _answer(self):
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    return self._json(401, {"error": "missing or wrong bearer token"})
                install.admin_calls.append(f"{self.command} {self.path}")
                parts = self.path.strip("/").split("/")
                if self.command == "GET" and parts == ["admin", "replicas"]:
                    return self._json(200, {"replicas": [{"name": n, "drained": d, "inflight": 0} for n, d in install.drained.items()]})
                if self.command == "POST" and len(parts) == 4 and parts[1] == "replicas" and parts[2] in install.drained:
                    was = install.drained[parts[2]]
                    install.drained[parts[2]] = parts[3] == "drain"
                    return self._json(200, {"name": parts[2], "drained": install.drained[parts[2]], "was_drained": was})
                return self._json(404, {"error": "not found"})

            do_GET = do_POST = _answer

        return Handler

    def _engine_handler(self):
        install = self

        class Handler(_Quiet):
            def do_GET(self):
                url = urllib.parse.urlparse(self.path)
                if url.path == "/_stop":
                    install.engine_flags = None
                    return self._json(200, {})
                if url.path == "/_start":
                    install._start(urllib.parse.parse_qs(url.query).get("flags", [""])[0])
                    return self._json(200, {})
                if install.engine_flags is None:
                    return self._json(503, {"error": "engine is down"})
                if url.path == "/health":
                    return self._json(200, {})
                if url.path == "/v1/models":
                    return self._json(200, {"object": "list", "data": [{"id": "fake-model"}]})
                return self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if install.engine_flags is None:
                    return self._json(503, {"error": "engine is down"})
                if self.path == "/tokenize":
                    words = body["prompt"].split()
                    return self._json(200, {"count": len(words), "tokens": [1000 + sum(map(ord, w)) % 5000 for w in words]})
                if self.path != "/v1/completions":
                    return self._json(404, {})

                slots = install._slots
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    with slots:  # the rest wait their turn, as on an engine at its limit
                        time.sleep(install.service_s / 2)
                        for i in range(body["max_tokens"]):
                            if i:
                                time.sleep(install.service_s / 2 / max(1, body["max_tokens"] - 1))
                            self._event({"choices": [{"index": 0, "text": " tok", "finish_reason": None}]})
                    self._event({"choices": [], "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": body["max_tokens"]}})
                    self.wfile.write(b"data: [DONE]\n\n")
                except OSError:
                    pass  # the client cut the request off

            def _event(self, payload):
                self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
                self.wfile.flush()

        return Handler


class _Quiet(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(handler) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
