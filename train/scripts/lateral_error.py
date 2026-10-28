"""Lateral (cross-track) vs longitudinal (along-track) error of the discrete-token greedy trajectory, relative to the GT
heading at each waypoint, on check_release's 1,000 fixed val + val_asset samples; constant velocity for reference.
Saves per-sample trajectories (<out>.npz) and the tables (<out>.json): |lateral| mean / p90 / max at 0.5, 1, 2, 3, 6.4 s,
share of samples with max |lateral| over the first 2 s above 0.5 / 1.0 m, |longitudinal| mean, by cell and turn/straight
(turn = GT heading change over 6.4 s above --turn-deg).
Usage (cwd train/): python scripts/lateral_error.py <checkpoint> <step-0 root> --out logs/lateral [--n 1000] [--batch 16]"""
import argparse, functools, json

import numpy as np, torch

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn
from morai_vla.stage1_eval import const_vel, discrete_trajectory

H = {"0.5s": 4, "1s": 9, "2s": 19, "3s": 29, "6.4s": 63}


def split_error(pred, gt, yaw):
    """(N, T, 2) positions, (N, T) GT heading -> lateral, longitudinal (N, T): the error in the GT path frame."""
    e = pred - gt; c, s = np.cos(yaw), np.sin(yaw)
    return -s * e[..., 0] + c * e[..., 1], c * e[..., 0] + s * e[..., 1]


def table(lat, lon):
    a = np.abs(lat); first2 = a[:, :H["2s"] + 1].max(1)
    out = {h: {"lat_mean": float(a[:, i].mean()), "lat_p90": float(np.percentile(a[:, i], 90)), "lat_max": float(a[:, i].max()),
               "lon_mean": float(np.abs(lon[:, i]).mean())} for h, i in H.items()}
    return out | {"share_lat_gt_0.5m_first2s": float((first2 > 0.5).mean()), "share_lat_gt_1.0m_first2s": float((first2 > 1.0).mean()), "n": len(lat)}


def report(d, turn_deg):
    lat, lon = split_error(d["pred"], d["gt"], d["gt_yaw"]); clat, clon = split_error(d["cv"], d["gt"], d["gt_yaw"])
    turn = np.abs(d["gt_yaw"][:, -1]) > np.radians(turn_deg)
    groups = {"all": np.ones(len(lat), bool), "turn": turn, "straight": ~turn} | {f"cell_{c}": d["cell"] == c for c in sorted(set(d["cell"]))}
    return {g: {"model": table(lat[m], lon[m]), "const_vel": table(clat[m], clon[m])} for g, m in groups.items() if m.any()} | {"turn_deg": turn_deg}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("root"); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--turn-deg", type=float, default=15.0)
    a = ap.parse_args()
    m = TeacherVLA.build(a.ckpt, attn_implementation="sdpa", device_map={"": 0}).eval()
    ds = MoraiDataset(a.root, ["val", "val_asset"], model_config=m.config, frame_stride=10, max_samples=1000, augment=False, seed=1,
                      vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn", "generation_mode": True})
    ds.rows = ds.rows.iloc[:a.n].reset_index(drop=True)                     # check_release / eval_stage1 order: same samples
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=8, collate_fn=functools.partial(collate_fn, model_config=m.config))
    dev = lambda x: {k: dev(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
    keep = {k: [] for k in ("pred", "gt", "gt_yaw", "cv")}
    for b in dl:
        b = dev(b)
        keep["pred"].append(discrete_trajectory(m, b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"])[..., :2].numpy())
        keep["gt"].append(b["ego_future_xyz"][:, 0, :, :2].float().cpu().numpy())
        r = b["ego_future_rot"][:, 0].float().cpu().numpy(); keep["gt_yaw"].append(np.arctan2(r[..., 1, 0], r[..., 0, 0]))
        keep["cv"].append(np.stack([const_vel(h).numpy() for h in b["ego_history_xyz"][:, 0, :, :].float().cpu()]))
        print(sum(len(x) for x in keep["pred"]), flush=True)
    d = {k: np.concatenate(v) for k, v in keep.items()} | {"cell": ds.rows.cell.astype(str).to_numpy()[:sum(len(x) for x in keep["pred"])],
                                                           "key": (ds.rows.episode.astype(str) + "/" + ds.rows.i.astype(str)).to_numpy()}
    np.savez(a.out + ".npz", **d)
    res = report(d, a.turn_deg); json.dump(res, open(a.out + ".json", "w"), indent=1)
    print(json.dumps({g: {"lat_mean_2s": v["model"]["2s"]["lat_mean"], "lat_mean_6.4s": v["model"]["6.4s"]["lat_mean"],
                          "share>0.5m_2s": v["model"]["share_lat_gt_0.5m_first2s"], "n": v["model"]["n"]} for g, v in res.items() if g != "turn_deg"}, indent=1))
