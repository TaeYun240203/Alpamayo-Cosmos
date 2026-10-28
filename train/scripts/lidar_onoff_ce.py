"""Teacher-forced trajectory CE on the same val samples with LiDAR present vs forced dropped (lidar_dropout=1), per cell.
Usage (cwd train/): python scripts/lidar_onoff_ce.py <checkpoint dir> <step-0 root> [--n 1000] [--out logs/lidar_onoff.json]"""
import argparse, json, time

import numpy as np, torch
from torch.utils.data import DataLoader

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("root")
ap.add_argument("--n", type=int, default=1000); ap.add_argument("--out", default="logs/lidar_onoff.json")
a = ap.parse_args()
m = TeacherVLA.build(a.ckpt, attn_implementation="sdpa", loss_weights={"traj_ce": 1.0, "lm": 0.0, "fm": 0.0}, device_map={"": 0}).eval()
kw = dict(model_config=m.config, vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn"}, frame_stride=10,
          max_samples=a.n, augment=False, seed=1)
on, off = (MoraiDataset(a.root, ["val", "val_asset"], lidar_dropout=p, **kw) for p in (0.0, 1.0))
assert (on.rows[["episode", "i"]].to_numpy() == off.rows[["episode", "i"]].to_numpy()).all(), "on/off must see the same samples"
dev = lambda x: {k: dev(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
load = lambda ds: DataLoader(ds, batch_size=None, num_workers=6, collate_fn=lambda s: s)   # decoding ~2 s/sample: in workers
rows, t = [], time.time()
for k, pair in enumerate(zip(load(on), load(off))):
    r = {"key": f"{on.rows.episode.iloc[k]}/{on.rows.i.iloc[k]}", "cell": str(on.rows.cell.iloc[k])}
    for name, s in zip(("on", "off"), pair):
        m.loss_sums.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m(**dev(collate_fn([s], model_config=m.config)))
        r[name] = m.loss_sums["traj_ce"][0]
    rows.append(r)
    if k % 100 == 0: print(k, round(time.time() - t), flush=True)
mean = lambda rs: {"on": round(float(np.mean([r["on"] for r in rs])), 4), "off": round(float(np.mean([r["off"] for r in rs])), 4),
                   "off_minus_on": round(float(np.mean([r["off"] - r["on"] for r in rs])), 4),
                   "frac_on_better": round(float(np.mean([r["on"] < r["off"] for r in rs])), 3), "n": len(rs)}
res = mean(rows) | {"s": round(time.time() - t), "by_cell": {c: mean([r for r in rows if r["cell"] == c]) for c in sorted({r["cell"] for r in rows})}}
json.dump({"summary": res, "rows": rows}, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
