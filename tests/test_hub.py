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

    r = subprocess.run([sys.executable, str(TOOLS / "hubctl.py"), "put", str(src), "src/evil.py"], env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "403" in r.stderr  # outside the artifact allowlist
    req = urllib.request.Request(f"http://127.0.0.1:{port}/status", headers={"X-Hearsay-Token": "wrong"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 401
