#!/usr/bin/env python3
"""Accept a GitHub push webhook and rebuild MediKo on Fox.

The process listens on localhost. Tailscale Funnel publishes one HTTPS path
to it. The existing Funnel route for other apps is left alone.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 9000
REPO = Path(os.environ.get("MEDIKO_REPO_DIR", "/home/aibel/mediko"))
SCRIPT = Path(
    os.environ.get(
        "MEDIKO_REBUILD_SCRIPT",
        "/home/aibel/.local/lib/mediko/rebuild.sh",
    )
)
LOG = Path(
    os.environ.get(
        "MEDIKO_DEPLOY_LOG",
        "/home/aibel/.local/share/mediko/deploy.log",
    )
)

_state = threading.Lock()
_running = False
_again = False


def load_secret() -> bytes:
    file_name = os.environ.get("MEDIKO_WEBHOOK_SECRET_FILE")
    if file_name:
        secret = Path(file_name).read_text(encoding="utf-8").strip()
    else:
        secret = os.environ.get("MEDIKO_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise SystemExit("Webhook secret is not configured.")
    return secret.encode()


SECRET = load_secret()


def signature_ok(header: str | None, body: bytes) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    given = header.removeprefix("sha256=")
    expected = hmac.new(SECRET, body, hashlib.sha256).hexdigest()
    if len(given) != len(expected):
        return False
    return hmac.compare_digest(given, expected)


def start_rebuild() -> None:
    global _running
    with _state:
        global _again
        if _running:
            _again = True
            return
        _running = True
    threading.Thread(target=_rebuild_loop, daemon=True).start()


def _rebuild_loop() -> None:
    global _running, _again
    while True:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("ab") as handle:
            subprocess.run(
                [str(SCRIPT)],
                cwd=REPO,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        with _state:
            if not _again:
                _running = False
                return
            _again = False


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._reply(200, b"mediko deploy hook\n")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length)
        if not signature_ok(self.headers.get("X-Hub-Signature-256"), body):
            self._reply(401, b"invalid signature\n")
            return
        event = self.headers.get("X-GitHub-Event", "")
        if event == "ping":
            self._reply(200, b"pong\n")
            return
        if event != "push":
            self._reply(200, b"ignored\n")
            return
        try:
            payload = json.loads(body.decode())
        except json.JSONDecodeError:
            self._reply(400, b"invalid json\n")
            return
        if payload.get("ref") != "refs/heads/main":
            self._reply(200, b"ignored ref\n")
            return
        start_rebuild()
        self._reply(202, b"rebuild started\n")

    def _reply(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> None:
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
