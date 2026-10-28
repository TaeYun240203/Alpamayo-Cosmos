"""Step-0 progress on winpc (plain python3, no numpy): ssh winpc-wsl python3 - < check.py"""
import glob, json, os
OUT = "/mnt/d/share/2026_MORAI_step0"
eps = sorted(os.path.basename(p)[:-4] for p in glob.glob(OUT + "/_ego/*.npz"))
done, frames, scans, byts, secs = 0, 0, 0, 0, 0.0
for e in eps:
    m = os.path.join(OUT, e, "meta.json")
    if os.path.exists(m):
        j = json.load(open(m)); done += 1; frames += sum(j["frames"].values()); scans += j["scans"]; byts += j["cam_bytes"]; secs += j["seconds"]
status = {}
for f in glob.glob(OUT + "/_logs/worker_*.log"):
    for line in open(f):
        p = line.split(" ", 2)
        if len(p) == 3 and not p[2].startswith(("ok", "skip")): status[p[1]] = p[2].strip()
print(json.dumps({"episodes": len(eps), "done": done, "frames": frames, "scans": scans, "cam_GB": round(byts / 1e9, 1),
                  "worker_hours": round(secs / 3600, 1), "not_ok": status}, indent=1))
