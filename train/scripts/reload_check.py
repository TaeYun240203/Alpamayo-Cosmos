"""Strict reload of a saved teacher checkpoint + which parts moved vs the release (max |delta| and mean |delta| per part).
Usage (cwd train/): python scripts/reload_check.py <checkpoint dir> <release dir>
Weights only, on CPU: safetensors are compared tensor by tensor, so no GPU is used."""
import glob, json, sys
from pathlib import Path

import torch
from safetensors import safe_open

from morai_vla.model import PARTS, TeacherVLA

ck, rel = sys.argv[1], sys.argv[2]
TeacherVLA.build(ck, attn_implementation="sdpa", device_map={"": "cpu"})          # raises unless every parameter loads
ref = {}
for f in glob.glob(f"{rel}/*.safetensors"):
    with safe_open(f, "pt") as h: ref.update({k: (f, k) for k in h.keys()})
out = {}
for f in glob.glob(f"{ck}/*.safetensors"):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            part = next((p for p, paths in PARTS.items() if any(k.startswith(x + ".") for x in paths)), "other")
            src = k.replace(".lidar_visual.", ".visual.") if k not in ref else k     # LiDAR ViT starts as the camera ViT copy
            if src not in ref: continue
            with safe_open(ref[src][0], "pt") as r: d = (h.get_tensor(k).float() - r.get_tensor(src).float()).abs()
            m = out.setdefault(part, [0.0, 0.0, 0])
            m[0] = max(m[0], float(d.max())); m[1] += float(d.sum()); m[2] += d.numel()
print(json.dumps({p: {"max_abs": round(a, 6), "mean_abs": float(f"{s / n:.3e}")} for p, (a, s, n) in sorted(out.items())}, indent=1))
