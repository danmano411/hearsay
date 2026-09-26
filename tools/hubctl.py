"""Client for tools/hub.py (stdlib only). Config: HEARSAY_HUB (default http://192.168.137.1:8770) and
HEARSAY_HUB_TOKEN (or the file data/hub/token).

    python tools/hubctl.py send --me gpu --to cpu --type DONE --task 12 --body "PR #20, R4 xlsr val_testlike 0.12"
    python tools/hubctl.py listen --me gpu            # for the Monitor tool: one line per new message, + heartbeats
    python tools/hubctl.py job --me gpu "XLS-R extraction 40%, eta 25 min"   # what the heartbeat reports
    python tools/hubctl.py put data/scores/R4_xlsr_mlp.parquet
    python tools/hubctl.py get data/scores/R4_xlsr_mlp.parquet
    python tools/hubctl.py ls data/scores
    python tools/hubctl.py status | inbox --me gpu [--all]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("HEARSAY_ROOT", Path(__file__).resolve().parents[1]))
HUB_DIR = ROOT / "data" / "hub"
URL = os.environ.get("HEARSAY_HUB", "http://192.168.137.1:8770").rstrip("/")
PEER = {"cpu": "gpu", "gpu": "cpu"}


def token():
    t = os.environ.get("HEARSAY_HUB_TOKEN")
    if not t and (HUB_DIR / "token").exists():
        t = (HUB_DIR / "token").read_text().strip()
    if not t:
        sys.exit("no hub token: set HEARSAY_HUB_TOKEN or write it to data/hub/token")
    return t


def call(method, path, data=None, headers=None, timeout=30):
    req = urllib.request.Request(URL + path, data=data, method=method,
                                 headers={"X-Hearsay-Token": token(), **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def jcall(method, path, obj=None):
    data = json.dumps(obj).encode() if obj is not None else None
    with call(method, path, data, {"Content-Type": "application/json"} if data else None) as r:
        return json.loads(r.read())


def fmt(m):
    task = f" #{m['task']}" if m.get("task") else ""
    return f"[hub {m['id']}] {m['ts']} {m['from']}->{m['to']} {m['type']}{task}: {m['body']}"


def state_file(me):
    HUB_DIR.mkdir(parents=True, exist_ok=True)
    return HUB_DIR / f"last_seen_{me}.txt"


def cmd_send(a):
    r = jcall("POST", "/msg", {"from": a.me, "to": a.to, "type": a.type, "task": a.task, "body": a.body})
    print(f"sent id {r['id']}")


def cmd_inbox(a):
    after = 0 if a.all else int(state_file(a.me).read_text() or 0) if state_file(a.me).exists() else 0
    for m in jcall("GET", f"/msg?to={a.me}&after={after}"):
        print(fmt(m))


def cmd_listen(a):
    """Poll forever; print each new message to me (and 'all'); heartbeat every 5 min; flag a silent peer once."""
    sf, last_hb, warned = state_file(a.me), 0.0, False
    after = int(sf.read_text() or 0) if sf.exists() else 0
    while True:
        try:
            if time.time() - last_hb > 300:
                job = (HUB_DIR / f"job_{a.me}.txt").read_text().strip() if (HUB_DIR / f"job_{a.me}.txt").exists() else ""
                jcall("POST", "/heartbeat", {"node": a.me, "job": job})
                last_hb = time.time()
                peer = jcall("GET", "/status")["heartbeats"].get(PEER[a.me])
                silent = peer is None or peer["age_s"] > 900
                if silent and not warned:
                    print(f"[hub] ALERT: {PEER[a.me]} has not sent a heartbeat for >15 min", flush=True)
                warned = silent
            for m in jcall("GET", f"/msg?to={a.me}&after={after}"):
                if m["from"] != a.me:
                    print(fmt(m), flush=True)
                after = m["id"]
                sf.write_text(str(after))
        except (urllib.error.URLError, OSError) as e:
            print(f"[hub] unreachable: {e}", flush=True)
            time.sleep(60)  # ponytail: fixed back-off; the Monitor stays armed, the owner sees the line
        time.sleep(a.interval)


def cmd_job(a):
    HUB_DIR.mkdir(parents=True, exist_ok=True)
    (HUB_DIR / f"job_{a.me}.txt").write_text(a.text)
    jcall("POST", "/heartbeat", {"node": a.me, "job": a.text})


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def cmd_put(a):
    src = Path(a.local)
    rel = (a.remote or src.resolve().relative_to(ROOT.resolve()).as_posix())
    with open(src, "rb") as f:
        with call("PUT", f"/files/{rel}", f, {"Content-Length": str(src.stat().st_size), "X-Sha256": sha(src)},
                  timeout=600) as r:
            print(json.loads(r.read()))


def cmd_get(a):
    dst = Path(a.local) if a.local else ROOT / a.remote
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with call("GET", f"/files/{a.remote}", timeout=600) as r, open(tmp, "wb") as f:
        want = r.headers["X-Sha256"]
        while b := r.read(1 << 20):
            f.write(b)
    if sha(tmp) != want:
        tmp.unlink()
        sys.exit("sha256 mismatch, file discarded")
    os.replace(tmp, dst)
    print(f"got {dst} ({dst.stat().st_size} bytes, sha256 ok)")


def cmd_ls(a):
    for it in jcall("GET", f"/files/{a.dir.rstrip('/')}/?list"):
        print(f"{it['bytes']:>14,}  {it['path']}")


def cmd_status(a):
    print(json.dumps(jcall("GET", "/status"), indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("send"); s.add_argument("--me", required=True); s.add_argument("--to", required=True)
    s.add_argument("--type", required=True); s.add_argument("--task", default=None); s.add_argument("--body", required=True)
    s = sp.add_parser("inbox"); s.add_argument("--me", required=True); s.add_argument("--all", action="store_true")
    s = sp.add_parser("listen"); s.add_argument("--me", required=True); s.add_argument("--interval", type=int, default=20)
    s = sp.add_parser("job"); s.add_argument("--me", required=True); s.add_argument("text")
    s = sp.add_parser("put"); s.add_argument("local"); s.add_argument("remote", nargs="?")
    s = sp.add_parser("get"); s.add_argument("remote"); s.add_argument("local", nargs="?")
    s = sp.add_parser("ls"); s.add_argument("dir")
    sp.add_parser("status")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
