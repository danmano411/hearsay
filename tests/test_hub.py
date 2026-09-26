import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def hub(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "HEARSAY_ROOT": str(tmp_path), "HEARSAY_HUB": f"http://127.0.0.1:{port}"}
    env.pop("HEARSAY_HUB_TOKEN", None)  # a node's real token (CLAUDE.md) would override the test hub's token file
    proc = subprocess.Popen([sys.executable, str(TOOLS / "hub.py"), "--host", "127.0.0.1", "--port", str(port)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for _ in range(100):
        if (tmp_path / "data/hub/token").exists():
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                pass
        time.sleep(0.1)
    yield tmp_path, env, port
    proc.kill()


def ctl(env, *args):
    r = subprocess.run([sys.executable, str(TOOLS / "hubctl.py"), *args], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_messages_files_and_guards(hub):
    root, env, port = hub
    ctl(env, "send", "--me", "cpu", "--to", "gpu", "--type", "ASSIGN", "--task", "7", "--body", "extract xlsr")
    out = ctl(env, "inbox", "--me", "gpu")
    assert "cpu->gpu ASSIGN #7: extract xlsr" in out
    assert ctl(env, "inbox", "--me", "cpu") == ""  # not addressed to cpu

    src = root / "local.bin"
    src.write_bytes(os.urandom(3_000_000))
    ctl(env, "put", str(src), "data/scores/x.parquet")
    assert (root / "data/scores/x.parquet").read_bytes() == src.read_bytes()
    ctl(env, "get", "data/scores/x.parquet", str(root / "back.bin"))
    assert (root / "back.bin").read_bytes() == src.read_bytes()
    assert "data/scores/x.parquet" in ctl(env, "ls", "data/scores")
    # a second machine: its own root, token via env -> pull fetches only what it lacks
    other = root / "other"
    env2 = {**env, "HEARSAY_ROOT": str(other), "HEARSAY_HUB_TOKEN": (root / "data/hub/token").read_text().strip()}
    assert "1 of 1 files to fetch" in ctl(env2, "pull", "data/scores")
    assert (other / "data/scores/x.parquet").read_bytes() == src.read_bytes()
    assert "0 of 1 files to fetch" in ctl(env2, "pull", "data/scores")
    (other / "data/scores/x.parquet").write_bytes(os.urandom(3_000_000))  # same size, different bytes
    assert "0 of 1 files to fetch" in ctl(env2, "pull", "data/scores")  # size-only check misses it
    assert "1 of 1 files to fetch" in ctl(env2, "pull", "data/scores", "--sha")  # --sha catches it
    (root / "data/scores/y.parquet.partdeadbeef").write_bytes(b"in flight")
    assert "y.parquet.partdeadbeef" not in ctl(env, "ls", "data/scores")  # in-flight uploads are not listed
    (root / "data/scores/xlsr.part1.pt").write_bytes(b"real artifact")
    assert "xlsr.part1.pt" in ctl(env, "ls", "data/scores")  # only exact upload-temp names are hidden

    r = subprocess.run([sys.executable, str(TOOLS / "hubctl.py"), "put", str(src), "src/evil.py"], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "403" in r.stderr  # outside the artifact allowlist
    req = urllib.request.Request(f"http://127.0.0.1:{port}/status", headers={"X-Hearsay-Token": "wrong"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 401


def raw(port, token, method, path, body=b"", headers=None):
    """-> HTTP status of one request (the hub must always answer, never drop the connection)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body if method != "GET" else None,
                                 method=method, headers={"X-Hearsay-Token": token, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_bad_requests_get_a_status_not_a_dropped_connection(hub):
    root, env, port = hub
    tok = (root / "data/hub/token").read_text().strip()
    good_sha = "0" * 64
    assert raw(port, tok, "GET", "/msg?to=gpu&after=x") == 400
    assert raw(port, tok, "POST", "/msg", b"[1, 2]", {"Content-Type": "application/json"}) == 400
    assert raw(port, tok, "POST", "/heartbeat", b"[]", {"Content-Type": "application/json"}) == 400
    # PUT must carry a sha256 the hub verifies (plan 07 s3), not just trust the bytes
    assert raw(port, tok, "PUT", "/files/data/scores/a.bin", b"abc") == 400
    assert raw(port, tok, "PUT", "/files/data/scores/a.bin", b"abc", {"X-Sha256": good_sha}) == 400  # mismatch
    assert not list((root / "data/scores").glob("*"))  # no committed file, no leaked .part temp
    # allowlist: the single-file entry is not a prefix; NTFS alternate data streams / drive letters are refused
    sha = __import__("hashlib").sha256(b"abc").hexdigest()
    for bad in ("data/processed/manifest.parquet_evil/x", "data/scores/a.bin:evil", "C:/x/data/scores/a.bin"):
        assert raw(port, tok, "PUT", f"/files/{bad}", b"abc", {"X-Sha256": sha}) == 403, bad
    assert raw(port, tok, "PUT", "/files/data/scores/a.bin", b"abc", {"X-Sha256": sha}) == 200


def test_listen_survives_a_malformed_response(tmp_path):
    """A hub restarting mid-response (or any non-JSON 200) must not kill the listener the Monitor depends on."""
    import http.server
    import threading

    class Bad(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"not json")

        do_POST = do_GET

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Bad)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {**os.environ, "HEARSAY_ROOT": str(tmp_path), "HEARSAY_HUB": f"http://127.0.0.1:{srv.server_port}",
           "HEARSAY_HUB_TOKEN": "t" * 22}
    proc = subprocess.Popen([sys.executable, "-u", str(TOOLS / "hubctl.py"), "listen", "--me", "gpu"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        line = proc.stdout.readline()
        time.sleep(1)
        assert "[hub] unreachable" in line and proc.poll() is None  # logged and still running
    finally:
        proc.kill()
        srv.shutdown()
