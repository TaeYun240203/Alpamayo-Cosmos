"""dagger.to_step0: a fake spool episode -> step-0 rows that MoraiDataset reads by config only, with the request's frames,
scans and history, and the label as the future; the label's (a, kappa) survives the trip (future() = the action space's
action_to_traj). CPU; the spool test needs the eval repo's engine code ($E2E_SERVER, as test_dagger.py)."""
import os
from pathlib import Path

import numpy as np, pandas as pd, pytest, torch

from dagger import to_step0
from morai_vla.data import MoraiDataset, ego_trajectory
from test_data_math import AS

SERVER = os.environ.get("E2E_SERVER", str(Path(__file__).resolve().parents[3] / "MORAI-eval-sync/src/e2e_driver/server"))
need_server = pytest.mark.skipif(not (Path(SERVER) / "assemble.py").exists(), reason=f"no engine code at {SERVER}")


def test_future_is_action_to_traj():
    from alpamayo_r1.action_space import UnicycleAccelCurvatureActionSpace
    sp = UnicycleAccelCurvatureActionSpace(**AS)
    t = np.arange(64) * 0.1
    ak = np.stack([0.8 * np.sin(t / 2), 0.03 * np.cos(t / 3)], -1)
    v, yaw = to_step0.future(ak, 5.0, 0.4)
    act = torch.from_numpy(np.stack([(ak[:, 0] - AS["accel_mean"]) / AS["accel_std"], (ak[:, 1] - AS["curvature_mean"]) / AS["curvature_std"]], -1))[None]
    ref, _ = sp.action_to_traj(act, torch.zeros(1, 16, 3, dtype=torch.float64), torch.eye(3, dtype=torch.float64).expand(1, 16, 3, 3),
                               t0_states={"v": torch.tensor([5.0], dtype=torch.float64)})
    fx = ego_trajectory(np.r_[np.full(16, 5.0), v], np.r_[np.full(16, 0.4), yaw], 15)[2]
    assert np.abs(fx[0, :, :2].numpy() - ref[0, :, :2].numpy()).max() < 0.02          # same path as the engine decodes it


@need_server
def test_fake_spool_to_rows(tmp_path):
    from dagger.fake import episode
    from rl.episodes import e2e
    from morai_vla.data import N_HIST
    wire = e2e(SERVER)[0]
    d = tmp_path / "spool" / "a0" / "ep00000"; episode(d, SERVER, n=40, seed=2)
    z = dict(np.load(d / "dagger_labels.npz")); z["valid"][:] = True; z["valid"][3] = False   # one partly valid row: left out
    np.savez(d / "dagger_labels.npz", **z)
    rows, _ = to_step0.episode_rows(d, wire)
    root = tmp_path / "root"
    st = to_step0.convert(d, root, wire, cell="dagger_ring", val=0.0)
    full = [r for r in rows if r["valid"].all()]
    assert st["kept"] == len(full) == len(rows) - (z["step"][3] in [r["j"] for r in rows]) and st["kept"] > 5
    parts = sorted((root / "_dagger/rows").glob("*.parquet"))
    pd.concat([pd.read_parquet(f) for f in parts]).to_parquet(root / "_dagger/dagger_index.parquet")
    assert to_step0.stage(root) == st["kept"] and to_step0.stage(root) == st["kept"]                 # idempotent
    ds = MoraiDataset(root, split="train", index="_dagger/dagger_index.parquet", augment=False)
    assert len(ds) == st["kept"] and set(ds.rows.cell) == {"dagger_ring"}
    k = int(np.flatnonzero(ds.rows.episode == f"{to_step0.name_of(d)}_dg{full[1]['j']:05d}")[0]); s = ds.sample(k)
    req = wire.unpack((d / f"{full[1]['j']:05d}.req.bin").read_bytes())
    assert np.allclose(s["camera_times"], req["camera_times"], atol=1e-6)                  # the request's own 4 frames
    assert len(s["camera_frames"]) == 4 and all(f.shape[0] == 4 for f in s["camera_frames"])
    assert s["bev_frames"].shape[0] == 2 and s["nav_text"] == req["nav_text"]
    v, yaw = to_step0.future(full[1]["ak"], req["ego_v"][-1], req["ego_yaw"][-1])
    hx, _, fx, _ = ego_trajectory(np.r_[req["ego_v"], v], np.r_[req["ego_yaw"], yaw], N_HIST - 1)
    assert torch.allclose(s["ego_future_xyz"], fx) and torch.allclose(s["ego_history_xyz"], hx)   # label future, real history
    root2 = tmp_path / "root2"                                                              # route s window: fake steps have no s
    assert to_step0.convert(d, root2, wire, s_range=(850, 935))["kept"] == 0
