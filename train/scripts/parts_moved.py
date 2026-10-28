"""Which parts a student run changed: rebuild the run's starting weights from its saved config.yaml (a new student is
deterministic: seeded random VLM or the 4B weights + T*'s transplanted expert; stage 2 starts from its init checkpoint),
then max / mean |delta| per part against <run>/final (or --ckpt). The expert of a stage-1 run must not move.
Usage (cwd train/): python scripts/parts_moved.py <run dir> [--ckpt <dir>]"""
import argparse, json
from pathlib import Path

import hydra.utils as hyu
from alpamayo.common import misc
from omegaconf import OmegaConf
from safetensors import safe_open

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--ckpt", default=None)
a = ap.parse_args()
cfg = OmegaConf.load(Path(a.run) / "config.yaml")
cfg.model.distill = False                                                             # the frozen T* is not needed here
misc.seed_everything(42)                                                              # as morai_vla.train (new token rows)
start = hyu.instantiate(cfg.model, _convert_="partial").state_dict()
parts = hyu.get_class(cfg.model._target_.rsplit(".", 1)[0]).PARTS
out = {}
for f in sorted(Path(a.ckpt or Path(a.run) / "final").glob("*.safetensors")):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            part = next((p for p, paths in parts.items() if any(k.startswith(x + ".") for x in paths)), "other")
            d = (h.get_tensor(k).float() - start[k].float()).abs()
            m = out.setdefault(part, [0.0, 0.0, 0])
            m[0] = max(m[0], float(d.max())); m[1] += float(d.sum()); m[2] += d.numel()
print(json.dumps({p: {"max_abs": float(f"{x:.3e}"), "mean_abs": float(f"{s / n:.3e}")} for p, (x, s, n) in sorted(out.items())}, indent=1))
