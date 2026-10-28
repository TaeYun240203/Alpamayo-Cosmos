"""GPU probe of the R2 fine-tune's validation regression (user 10/28 "원인 분석 둘 다 해", GPU part). For chosen val rows:
s2 final, then ft step N (s2 final + FtSelect's trainable-parameter snapshot, as FtSelect.finalize loads it), each with
the row's real history tokens, zero history (the fine-tune's dropout) and the lane-shaped synthetic one
(scripts/ft_synth_hist.py, precomputed per row); the action space always integrates from the real history (ft_measure.BF16).
Per row / model / history / seed: ADE, |lateral| and signed longitudinal error at 3 s and 6.4 s (+ = ahead of GT, i.e.
too fast / too far), path length vs GT; the first seed's predicted path for overlays.
Usage (cwd train/): python scripts/ft_probe.py --root /data/step0 --keep keep.parquet --synth synth.parquet --s2 <s2/final>
    --snapshot <ft_snapshots/step-1000.safetensors> --out <dir> [--k 2]"""
import argparse, json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morai_vla.data import MoraiDataset, hist_tensors          # noqa: E402
from scripts.ft_measure import BF16, gt_of                       # noqa: E402
from scripts.lateral_error import split_error                    # noqa: E402

T3, T64 = 29, 63


def run(model, tag, ds, synth, k, f):
    for n in range(len(ds.rows)):
        s = ds.sample(n); gt, yaw = gt_of(s); key = s["key"]
        hists = {"real": (s["ego_history_xyz"], s["ego_history_rot"]),
                 "zero": (torch.zeros_like(s["ego_history_xyz"]), torch.eye(3).expand_as(s["ego_history_rot"]).clone())}
        if key in synth: hists["synth"] = hist_tensors(*synth[key])
        L = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum())
        for h, hist in hists.items():
            for j in range(k):
                pred = model(s, hist, 1000 * n + j)
                lat, lon = split_error(pred[None], gt[None], yaw[None])
                r = dict(model=tag, key=key, hist=h, seed=j, ade=float(np.linalg.norm(pred - gt, axis=-1).mean()),
                         lat3=float(abs(lat[0, T3])), lat64=float(abs(lat[0, T64])), lon3=float(lon[0, T3]), lon64=float(lon[0, T64]),
                         len_pred=L(pred), len_gt=L(gt))
                if j == 0: r |= dict(pred=pred.round(3).tolist(), gt=gt.round(3).tolist())
                f.write(json.dumps(r) + "\n")
        f.flush()
        if n % 20 == 0: print(tag, n, key, flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--root", "--keep", "--synth", "--s2", "--snapshot", "--out"): ap.add_argument(k, required=True)
    ap.add_argument("--k", type=int, default=2); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sy = pd.read_parquet(a.synth); synth = {r.key: (r.hist_x, r.hist_y, r.hist_yaw) for r in sy.itertuples()}
    model = BF16(a.s2, a.device)
    ds = MoraiDataset(a.root, ["val", "val_asset"], frame_stride=1, augment=False, seed=1, keep=a.keep)
    print("rows", len(ds.rows), flush=True)
    with open(out / "probe_rows.jsonl", "w") as f:
        run(model, "s2", ds, synth, a.k, f)
        sd = load_file(a.snapshot, device="cpu")                                 # trainable params at step N (FtSelect snapshot)
        missing, unexpected = model.m.load_state_dict(sd, strict=False)
        assert not unexpected and not set(sd) & set(missing), (unexpected[:5], list(set(sd) & set(missing))[:5])
        print("snapshot", a.snapshot, len(sd), "tensors", flush=True); del sd
        run(model, "ft", ds, synth, a.k, f)
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
