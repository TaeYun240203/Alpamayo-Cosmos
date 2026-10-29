"""CPU check of every source and validation set of a mixture config (CLAUDE.md 1.2, the LwF launch gate): composes the config
with the given overrides and builds its MixtureDataset without tokenisation, so every source's MoraiDataset opens its index,
keep list, synthetic histories and each episode's step-0 indexes, and kd sources read their _ego signals. Then it samples n
items per source (camera frames, BEVs, history, future; synthetic-history rows too) and opens every ft_select set's rows.
The virtual stop episodes' futures are checked by scripts/dyn_load_check.py (run it on both roots as well).
Prints one line per source / set and ends with `rc=0` (rent_gate.sh --loader-log) or raises.
Usage (cwd train/): python scripts/lwf_load_check.py --config-name student_lwf [--n 3] data.root=<step0> paths.r2_root=<step0_r2> \
  paths.popout_root=<step0_popout> paths.s2=<any> paths.output_dir=<any>
"""
import argparse, json, time
from pathlib import Path

import numpy as np, torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from morai_vla.data import MoraiDataset
from morai_vla.finetune import MixtureDataset


def item_ok(it):
    assert len(it["camera_frames"]) == 4 and all(f.shape[0] == 4 for f in it["camera_frames"]), "camera frames"
    assert "bev_frames" in it or it["lidar_dropped"], "BEV"
    for k in ("ego_history_xyz", "ego_future_xyz", "hist_xyz"):
        assert torch.isfinite(it[k]).all(), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="student_lwf"); ap.add_argument("--n", type=int, default=3); ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(); t0 = time.time()
    with initialize_config_dir(str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        c = OmegaConf.to_container(compose(a.config_name, overrides=a.overrides + ["trainer.deepspeed=null"]), resolve=True)
    td = {k: v for k, v in c["data"]["train_dataset"].items() if k not in ("_target_", "_recursive_", "vla_preprocess_args")}
    total = sum(float(s["share"]) for s in td["sources"].values())
    assert abs(total - 1) < 1e-4, f"shares sum to {total}"
    m = MixtureDataset(**td)
    print(f"mixture: {len(td['sources'])} sources, shares {total:.6f}, schedule {len(m)}, built in {time.time() - t0:.0f} s", flush=True)
    for j, s in enumerate(m.names):
        pos = np.flatnonzero(m.schedule[:, 0] == j)
        assert len(m.parts[j].rows) and len(pos), f"{s}: no rows"
        hs = 0
        for k in pos[:a.n]:
            it = m[int(k)]; item_ok(it); hs += bool(it["hist_synth"])
        kd = m.kd[j] if getattr(m, "kd", None) else None
        print(f"source {s}: rows {len(m.parts[j].rows)}, samples {len(pos)}, sampled {min(a.n, len(pos))} ok (synth hist {hs})"
              + (f", kd rows {int(kd.sum())}" if kd is not None else ""), flush=True)
    for name, parts in c["callbacks"]["ft_select"]["sets"].items():
        for p in parts:
            p = dict(p); n = p.pop("n")
            ds = MoraiDataset(**p, max_samples=n, augment=False, seed=1)
            assert len(ds.rows), f"set {name}: no rows ({p.get('keep')})"
            ds.sample(0)
            print(f"set {name}: {min(n, len(ds.rows))} rows ({p.get('keep', p['root'])}), first row ok", flush=True)
    ex = c["callbacks"]["ft_select"].get("stop_exclude")
    assert not ex or Path(ex).exists(), f"stop_exclude {ex} missing"
    print(json.dumps({k: m.summary[k] for k in ("total", "new_rows_4hz", "gt_share", "hist_drop_share", "hist_synth_share", "kd_share") if k in m.summary}))
    print(f"lwf_load_check done in {time.time() - t0:.0f} s rc=0")


if __name__ == "__main__":
    main()
