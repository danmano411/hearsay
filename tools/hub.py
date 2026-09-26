"""LAN hub for the two-node setup (plans/07_two_node_protocol.md): messages, artifacts, heartbeats.

    python tools/hub.py [--host 192.168.137.1] [--port 8770]
Stdlib only. Every request needs header X-Hearsay-Token == contents of data/hub/token (created on first run).
State lives in data/hub/: messages.jsonl (append-only log), heartbeats.json, hub.log.
"""
import argparse
import hashlib
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(os.environ.get("HEARSAY_ROOT", Path(__file__).resolve().parents[1]))
HUB = ROOT / "data" / "hub"
# artifacts may only be read/written under these (relative to ROOT)
ALLOWED = ("data/scores/", "data/features/", "data/models/", "data/processed/sim/",
           "data/processed/manifest.parquet", "data/processed/manifests/")
TYPES = {"ASSIGN", "CLAIM", "PROGRESS", "ARTIFACT", "DONE", "REVIEW_REQUEST", "REVIEW", "QUESTION", "ANSWER",
         "BLOCKED", "SYNC", "STOP", "NOTE"}
CHUNK = 1 << 20

lock = threading.Lock()


def safe_path(rel):
    rel = unquote(rel).replace("\\", "/").lstrip("/")
    p = (ROOT / rel).resolve()
    ok = any(rel == a or rel.startswith(a) for a in ALLOWED) and ROOT.resolve() in p.parents
    return (p, rel) if ok and ".." not in rel.split("/") else (None, rel)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def read_msgs():
    f = HUB / "messages.jsonl"
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line] if f.exists() else []


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        if self.headers.get("X-Hearsay-Token") != TOKEN:
            self._json(401, {"error": "bad token"})
            return False
        return True

    def _body_json(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def log_message(self, fmt, *args):
        with open(HUB / "hub.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {self.client_address[0]} {fmt % args}\n")

    def do_GET(self):
        if not self._auth():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query, keep_blank_values=True).items()}
        if u.path == "/msg":
            after, to = int(q.get("after", 0)), q.get("to")
            msgs = [m for m in read_msgs() if m["id"] > after and (to is None or m["to"] in (to, "all"))]
            return self._json(200, msgs)
        if u.path == "/status":
            hb = json.loads((HUB / "heartbeats.json").read_text()) if (HUB / "heartbeats.json").exists() else {}
            for v in hb.values():
                v["age_s"] = round(time.time() - v["t"])
            return self._json(200, {"heartbeats": hb, "messages": len(read_msgs())})
        if u.path.startswith("/files/"):
            p, rel = safe_path(u.path[len("/files/"):])
            if p is None:
                return self._json(403, {"error": f"not allowed: {rel}"})
            if "list" in q:
                if not p.is_dir():
                    return self._json(404, {"error": "no such dir"})
                items = [{"path": x.relative_to(ROOT).as_posix(), "bytes": x.stat().st_size}
                         for x in sorted(p.rglob("*")) if x.is_file()]
                return self._json(200, items)
            if not p.is_file():
                return self._json(404, {"error": "no such file"})
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(p.stat().st_size))
            self.send_header("X-Sha256", sha256(p))
            self.end_headers()
            with open(p, "rb") as f:
                while b := f.read(CHUNK):
                    self.wfile.write(b)
            return
        self._json(404, {"error": "unknown endpoint"})

    def do_POST(self):
        if not self._auth():
            return
        path = urlparse(self.path).path
        if path == "/msg":
            m = self._body_json()
            missing = {"from", "to", "type", "body"} - set(m)
            if missing or m["type"] not in TYPES:
                return self._json(400, {"error": f"missing {sorted(missing)} or type not in {sorted(TYPES)}"})
            with lock:
                m.update(id=len(read_msgs()) + 1, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
                with open(HUB / "messages.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(m) + "\n")
            return self._json(200, {"id": m["id"]})
        if path == "/heartbeat":
            b = self._body_json()
            with lock:
                f = HUB / "heartbeats.json"
                hb = json.loads(f.read_text()) if f.exists() else {}
                hb[b.get("node", "?")] = {**b, "t": time.time(), "when": time.strftime("%H:%M:%S")}
                f.write_text(json.dumps(hb, indent=1))
            return self._json(200, {"ok": True})
        self._json(404, {"error": "unknown endpoint"})

    def _drain(self):
        """Read and discard a rejected upload so the client gets our error instead of a reset connection."""
        left = int(self.headers.get("Content-Length", 0))
        while left > 0 and (b := self.rfile.read(min(CHUNK, left))):
            left -= len(b)

    def do_PUT(self):
        u = urlparse(self.path)
        p, rel = safe_path(u.path[len("/files/"):]) if u.path.startswith("/files/") else (None, u.path)
        if self.headers.get("X-Hearsay-Token") != TOKEN or p is None:
            self._drain()
        if not self._auth():
            return
        if not u.path.startswith("/files/"):
            return self._json(404, {"error": "unknown endpoint"})
        if p is None:
            return self._json(403, {"error": f"not allowed: {rel}"})
        n, h = int(self.headers["Content-Length"]), hashlib.sha256()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".part{secrets.token_hex(4)}")
        with open(tmp, "wb") as f:
            left = n
            while left:
                b = self.rfile.read(min(CHUNK, left))
                if not b:
                    break
                f.write(b)
                h.update(b)
                left -= len(b)
        want = self.headers.get("X-Sha256")
        if left or (want and want != h.hexdigest()):
            tmp.unlink(missing_ok=True)
            return self._json(400, {"error": "incomplete upload or sha256 mismatch"})
        os.replace(tmp, p)  # atomic: readers never see a half file
        self._json(200, {"path": rel, "bytes": n, "sha256": h.hexdigest()})


def main():
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.137.1", help="hotspot address only (not 0.0.0.0: eduroam is public)")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    HUB.mkdir(parents=True, exist_ok=True)
    tf = HUB / "token"
    if not tf.exists():
        tf.write_text(secrets.token_urlsafe(16))
    TOKEN = tf.read_text().strip()
    print(f"hub on http://{a.host}:{a.port}  (token in {tf})", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


TOKEN = None
if __name__ == "__main__":
    main()
