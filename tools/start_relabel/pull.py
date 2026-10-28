"""Inside the Noetic chroot: front JPEG nearest each requested tau, and the BEV height channel of the nearest scan.
In: /tmp/pull_req.json [{ep, cams: [[tag, tau]], bevs: [[tag, tau]]}]. Out: /tmp/pull_out/<ep>_<tag>.jpg / .npy."""
import json, os, zlib
import numpy as np

ROOT = "/mnt/2026_MORAI_step0/"
os.makedirs("/tmp/pull_out", exist_ok=True)
for r in json.load(open("/tmp/pull_req.json")):
    d = ROOT + r["ep"]
    z = np.load(d + "/cams_index.npz"); f = np.where(z["cam"] == 0)[0]
    with open(d + "/cams.bin", "rb") as fh:
        for tag, tau in r.get("cams", []):
            j = f[np.argmin(np.abs(z["tau"][f] - tau))]
            fh.seek(int(z["offset"][j])); open("/tmp/pull_out/%s_%s.jpg" % (r["ep"], tag), "wb").write(fh.read(int(z["length"][j])))
    if r.get("bevs"):
        li = np.load(d + "/lidar_index.npz")
        with open(d + "/lidar.bin", "rb") as fh:
            for tag, tau in r["bevs"]:
                j = int(np.argmin(np.abs(li["tau"] - tau)))
                fh.seek(int(li["offset"][j])); b = zlib.decompress(fh.read(int(li["length"][j])))
                np.save("/tmp/pull_out/%s_%s.npy" % (r["ep"], tag), np.frombuffer(b, np.uint8).reshape(2, 320, 160)[0])
print("done")
