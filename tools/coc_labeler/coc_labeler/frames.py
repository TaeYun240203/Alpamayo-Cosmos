"""Front-camera frames for keyframes: native JPEG bytes nearest to requested bag times (2 Hz, t = -2..+6 s).
LocalBags reads <ep>_<i>.bag from a folder; Winpc extracts inside the Noetic chroot on winpc (bags stay on D:)."""
import io, json, os, shutil, subprocess, tarfile, time, uuid
from pathlib import Path

import numpy as np

TOPIC = "/image/front/compressed_10hz"
CACHE = Path(os.path.expanduser("~/.cache/coc_labeler/frames"))
REMOTE_DIR = "/home/taeyun_kim/coc_labeler_remote"
SSH = ["ssh", "-o", "ConnectTimeout=10", "winpc-wsl"]

EXTRACT_PY = r'''
import glob, json, os, sys, genpy, rosbag
req = json.load(open(sys.argv[1])); out = sys.argv[2]
for ep, keys in req.items():
    chunks = sorted(glob.glob("/mnt/2026_MORAI_collect_20261018/*/%s_*.bag" % ep)) or glob.glob("/mnt/2026_MORAI_collect_20261018/*/%s.bag" % ep)
    bags, spans = [], []
    for c in chunks:                       # a trailing chunk can be empty (collector stopped right after a split)
        try:
            b = rosbag.Bag(c); spans.append((b.get_start_time(), b.get_end_time())); bags.append(b)
        except Exception:
            pass
    for key, nss in keys.items():
        d = os.path.join(out, ep, key)
        if not os.path.isdir(d): os.makedirs(d)
        for j, ns in enumerate(nss):
            t = ns / 1e9; best = None
            for b, (s0, s1) in zip(bags, spans):
                if not (s0 - 1 <= t <= s1 + 1): continue
                for w in (0.15, 0.6):
                    for _, msg, tt in b.read_messages(topics=["/image/front/compressed_10hz"], start_time=genpy.Time.from_sec(t - w), end_time=genpy.Time.from_sec(t + w)):
                        dt = abs(tt.to_sec() - t)
                        if best is None or dt < best[0]: best = (dt, msg.data)
                    if best is not None: break
            if best is not None:
                open(os.path.join(d, "%02d.jpg" % j), "wb").write(best[1])
    for b in bags: b.close()
'''

RUN_SH = r'''#!/usr/bin/env bash
R=/home/taeyun_kim/.codex/morai-mpc-runsh-wsl-20261010-a1/noetic-rootfs
W=$(cd "$(dirname "$0")" && pwd)
exec unshare -Urm --fork bash -c "mount --bind /mnt/d/share $R/mnt && mount --bind $W $R/tmp && chroot $R /bin/bash -c \"source /opt/ros/noetic/setup.bash; cd /tmp; python3 $*\""
'''


def key_dir(ep, key):
    return CACHE / ep / key


class LocalBags:
    """bags already on this machine: <root>/<ep>_<i>.bag"""
    def __init__(self, root):
        self.root = Path(root)

    def fetch(self, ep, requests):
        from rosbags.highlevel import AnyReader
        paths = sorted(self.root.glob(f"{ep}_*.bag")) or sorted(self.root.glob(f"{ep}.bag"))
        if not paths: raise FileNotFoundError(f"no bags for {ep} in {self.root}")
        want = sorted({ns for nss in requests.values() for ns in nss})
        best = {}
        with AnyReader(paths) as r:
            con = [c for c in r.connections if c.topic == TOPIC]
            for c, t, raw in r.messages(connections=con):
                i = np.searchsorted(want, t)
                for j in (i - 1, i):
                    if 0 <= j < len(want) and abs(t - want[j]) < 6e8 and (want[j] not in best or abs(t - want[j]) < best[want[j]][0]):
                        best[want[j]] = (abs(t - want[j]), raw, c)
            for key, nss in requests.items():
                d = key_dir(ep, key); d.mkdir(parents=True, exist_ok=True)
                for j, ns in enumerate(nss):
                    if ns in best:
                        _, raw, c = best[ns]; (d / f"{j:02d}.jpg").write_bytes(bytes(r.deserialize(raw, c.msgtype).data))
        return {k: key_dir(ep, k) for k in requests}


class Winpc:
    """extract on winpc (rosbag in the Noetic chroot) and pull only the JPEGs over ssh."""
    def __init__(self):
        self.ready = False

    def _setup(self):
        if self.ready: return
        subprocess.run(SSH + [f"mkdir -p {REMOTE_DIR}"], check=True)
        for name, body in (("extract.py", EXTRACT_PY), ("run.sh", RUN_SH)):
            subprocess.run(SSH + [f"cat > {REMOTE_DIR}/{name}"], input=body.encode(), check=True)
        subprocess.run(SSH + [f"chmod +x {REMOTE_DIR}/run.sh"], check=True)
        self.ready = True

    def fetch(self, ep, requests):
        for attempt in range(4):                     # ssh drops under load on winpc; the request is idempotent
            try:
                return self._fetch(ep, requests)
            except subprocess.CalledProcessError:
                if attempt == 3: raise
                time.sleep(20 * (attempt + 1))

    def _fetch(self, ep, requests):
        self._setup()
        job = uuid.uuid4().hex[:10]
        subprocess.run(SSH + [f"cat > {REMOTE_DIR}/req_{job}.json"], input=json.dumps({ep: requests}).encode(), check=True)
        subprocess.run(SSH + [f"cd {REMOTE_DIR} && ./run.sh extract.py req_{job}.json out_{job}"], check=True, capture_output=True)
        tar = subprocess.run(SSH + [f"tar cf - -C {REMOTE_DIR}/out_{job} {ep}"], check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
            for m in tf.getmembers():
                if m.isfile():
                    dst = CACHE / m.name; dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(tf.extractfile(m).read())
        clean = f"import shutil,os; shutil.rmtree('{REMOTE_DIR}/out_{job}', ignore_errors=True); os.remove('{REMOTE_DIR}/req_{job}.json')"
        subprocess.run(SSH + [f"python3 -c \"{clean}\""], check=False)
        return {k: key_dir(ep, k) for k in requests}


def drop(ep, key=None):
    shutil.rmtree(key_dir(ep, key) if key else CACHE / ep, ignore_errors=True)
