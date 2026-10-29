"""Nav-text probe (10/29): dynbal did not stop for pop-out pedestrians in closed loop, and 81.7 % of its pop-out training rows
had nav_text exactly "Follow the road" (no ". Speed limit 60 km/h" suffix, a string no other data has), so the student may
read that phrase as "stop". Per row of --keep (optionally only its --kind rows) the same generation prompt twice: nav_text =
--bug, and nav_text = the row's own from --index; K expert samples from the same seed (torch.manual_seed(0) per prompt, as
early_stop.expert_ade), so only the text differs. Per row and variant: v0 and pred_drop (early_stop.brake_drops over 1.0 s),
brake = pred_drop >= 1.5 (finetune.brake_rates), planned distance at 3.0 s and the mean planned speed over 0-3 s (K mean).
Usage (cwd train/): python scripts/navprobe.py --ckpt <student dir> --root /data/step0_popout \
    --index _dyn_relabel/dyn_index_nav.parquet --keep <popout_val_onset.parquet> [--kind onset] --out <dir> [--k 6] [--n N]
Writes <out>/keep.parquet (the rows asked for), rows.csv and summary.json (diff = bug - fixed, paired per row)."""
import argparse, json, logging, sys
from pathlib import Path

import pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morai_vla.data import DT, MoraiDataset                     # noqa: E402
from morai_vla.early_stop import brake_drops                     # noqa: E402
from morai_vla.processor import collate_fn, get_preprocess_fn    # noqa: E402
from morai_vla.student import StudentVLA                         # noqa: E402

T3, DROP, M = 30, 1.5, ("pred_drop", "speed3", "dist3")         # predicted points 0.1 .. 6.4 s: the first 30 reach 3.0 s
log = logging.getLogger("navprobe")


def batch(fn, s, text, config):
    d = dict(s) | {"nav_text": text, "cot": None}; d["tokenized_data"] = fn(data=d)
    return collate_fn([{k: v for k, v in d.items() if k not in ("camera_frames", "bev_frames")}], model_config=config)


def plan(m, b, k, device):
    dev = lambda x: {kk: dev(v) if isinstance(v, dict) else v.to(device) if torch.is_tensor(v) else v for kk, v in x.items()}
    b = dev(b); torch.manual_seed(0)
    xyz, _ = m.sample_trajectories(b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"], num_samples=k)
    p = xyz[0, ..., :2].float()                                                   # (K, 64, 2)
    r = brake_drops(b["ego_history_xyz"][0, 0, :, :2].float(), p, b["ego_future_xyz"][0, 0, :, :2].float(), 1.0)
    dist = float(torch.diff(p[:, :T3], dim=1, prepend=torch.zeros_like(p[:, :1])).norm(dim=-1).sum(1).mean())
    return dict(v0=r["v0"], pred_drop=r["pred_drop"], brake=r["pred_drop"] >= DROP, speed3=dist / (T3 * DT), dist3=dist)


def main(argv=None):
    ap = argparse.ArgumentParser()
    for x in ("--ckpt", "--root", "--keep", "--out"): ap.add_argument(x, required=True)
    ap.add_argument("--index", default="_index/index.parquet"); ap.add_argument("--kind"); ap.add_argument("--bug", default="Follow the road")
    ap.add_argument("--split", nargs="+", default=["val", "val_asset"]); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n", type=int); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args(argv); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keep = pd.read_parquet(a.keep).astype({"episode": str})
    if a.kind: keep = keep[keep.kind.astype(str) == a.kind]
    keep.to_parquet(out / "keep.parquet", index=False)
    kinds = dict(zip(keep.episode + "/" + keep.i.astype(str), keep.kind.astype(str))) if "kind" in keep else {}
    m = StudentVLA.build(init=a.ckpt, vlm_name_or_path="Qwen/Qwen3-VL-4B-Instruct", attn_implementation=a.attn,
                         dtype="bfloat16" if "cuda" in a.device else "float32", device_map={"": a.device}).eval()
    ds = MoraiDataset(a.root, a.split, index=a.index, keep=str(out / "keep.parquet"), max_samples=a.n, augment=False, seed=1)
    fn, rows, skipped = get_preprocess_fn(m.config, generation_mode=True), [], 0
    for n in range(len(ds.rows)):
        try:
            s = ds.sample(n); bs = {v: batch(fn, s, t, m.config) for v, t in (("bug", a.bug), ("fixed", s["nav_text"]))}
        except Exception as e:                                                    # a row whose files fail to read: skip, count
            skipped += 1; log.warning("skip row %d: %r", n, e); continue
        r = dict(key=s["key"], kind=kinds.get(s["key"], ""), nav_text=s["nav_text"])
        for v, b in bs.items(): r |= {f"{v}_{x}": y for x, y in plan(m, b, a.k, a.device).items()}
        rows.append(r)
        if n % 20 == 0: print(n, json.dumps(r), flush=True)
    if not rows: sys.exit(f"no rows ({skipped} skipped)")
    df = pd.DataFrame(rows); df.to_csv(out / "rows.csv", index=False)
    f = lambda x: round(float(x), 4)
    s = dict(n=len(df), skipped=skipped, k=a.k, bug_text=a.bug, ckpt=a.ckpt, index=a.index, kind=a.kind, v0=f(df.bug_v0.mean()),
             same_text=f((df.nav_text == a.bug).mean()),                          # > 0: --index holds the buggy text itself
             variants={v: {"brake_rate": f(df[f"{v}_brake"].mean())} | {x: f(df[f"{v}_{x}"].mean()) for x in M} for v in ("bug", "fixed")},
             diff={x: {"mean": f(d.mean()), "median": f(d.median())} for x in M for d in [df[f"bug_{x}"] - df[f"fixed_{x}"]]},
             only_bug_brakes=f((df.bug_brake & ~df.fixed_brake).mean()), only_fixed_brakes=f((df.fixed_brake & ~df.bug_brake).mean()))
    json.dump(s, open(out / "summary.json", "w"), indent=1); print(json.dumps(s, indent=1))
    return s


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
