"""Braking-onset metrics (reports/lwf_dynstop_recipe_20261029.md section 6), CPU, no model: early_stop.brake_drops on
ego_trajectory paths through expert_ade (fake model + collate), and FtSelect's per-set brake / false_brake / brake_hit
(expert_ade mocked as in test_finetune.py), ft_val_rows.jsonl and scripts/ft_metrics.py with the extra keys."""
import json, math
from types import SimpleNamespace

import numpy as np, torch

from morai_vla import early_stop, finetune
from morai_vla.data import ego_trajectory


def path(v):
    """(hist (1, 16, 3), hist rot, fut (1, 64, 3), fut rot) of a straight drive with grid speeds v (row 15 = t0)."""
    return ego_trajectory(np.asarray(v, float), np.zeros(len(v)), 15)


def test_brake_drops_through_expert_ade(monkeypatch):
    t = np.arange(80) * 0.1 - 1.5                                                   # grid time, t0 = row 15
    cruise, stop = path(np.full(80, 10.0)), path(np.clip(10 - 3 * np.clip(t, 0, None), 0, None))   # 3 m/s^2 from t0
    late = path(np.clip(10 - 3 * np.clip(t - 1.5, 0, None), 0, None))                       # brakes only after 1.5 s
    items = [dict(key=k, ego_history_xyz=p[0], ego_history_rot=p[1], ego_future_xyz=p[2], ego_future_rot=p[3], tokenized_data={})
             for k, p in (("cruise", cruise), ("stop", stop), ("late", late))]
    preds = {"cruise": cruise[2], "stop": stop[2], "late": late[2]}

    class Fake(torch.nn.Linear):                                                   # the expert predicts each row's GT
        config, action_space = None, SimpleNamespace(get_action_space_dims=lambda: (64, 2))   # batch > 1 draws x0
        def sample_trajectories(self, tok, hx, hr, num_samples=1, noise=None):
            return torch.stack([p.expand(num_samples, 64, 3) for p in self.batch]), None
    m = Fake(1, 1)
    def collate(it, model_config=None):
        m.batch = [preds[x["key"]] for x in it]
        return {k: torch.stack([x[k] for x in it]) for k in it[0] if k not in ("key", "tokenized_data")} | {"tokenized_data": {}}
    monkeypatch.setattr(early_stop, "collate_fn", collate)
    rows = dict(early_stop.expert_ade(m, items, num_samples=3, num_workers=0, detail=True, batch=2))
    assert all(abs(r["v0"] - 10) < 1e-4 for r in rows.values()), rows                 # history steps are 0.1 s apart
    assert abs(rows["cruise"]["pred_drop"]) < 1e-4 and abs(rows["cruise"]["gt_drop"]) < 1e-4
    assert abs(rows["stop"]["gt_drop"] - 3 * 0.95) < 1e-3 and abs(rows["stop"]["pred_drop"] - rows["stop"]["gt_drop"]) < 1e-5   # ~3 in 1 s
    assert abs(rows["late"]["gt_drop"]) < 1e-4                                       # outside the 1.0 s window
    assert rows["stop"]["ade"] < 1e-5 and set(rows["stop"]) == {"ade", "fde", "lat", "lon", "v0", "pred_drop", "gt_drop"}
    assert [k for k, _ in early_stop.expert_ade(m, items[:1], 3, 0)] == ["cruise"]          # detail False: (key, ADE) as before
    # K samples: the mean of the per-sample drops (half the samples brake -> half the drop)
    two = torch.cat([stop[2][0, None].expand(2, 64, 3), cruise[2][0, None].expand(2, 64, 3)])[..., :2]
    d = early_stop.brake_drops(cruise[0][0, :, :2], two, cruise[2][0, :, :2])
    assert abs(d["pred_drop"] - 3 * 0.95 / 2) < 1e-3 and abs(d["gt_drop"]) < 1e-4
    assert abs(early_stop.brake_drops(cruise[0][0, :, :2], late[2][0, None, :, :2], late[2][0, :, :2], window=2.0)["gt_drop"] - 3 * 0.45) < 1e-3


GEN = [("e/1", 10.0, 0.0, 2.0), ("e/2", 10.0, 0.0, 0.1), ("e/3", 1.0, 0.0, 1.0), ("e/4", 10.0, 3.0, 2.5)]   # key, v0, gt, pred
ONSET = [("p/1", 8.0, 3.0, 2.0), ("p/2", 8.0, 3.0, 0.5), ("p/3", 8.0, 3.0, 1.5)]


def test_ftselect_brake_rates(tmp_path, monkeypatch):
    from scripts import ft_metrics
    seen = []
    def fake(model, samples, *a, **k):
        seen.append(k.get("brake_s"))
        return [(key, {"ade": 1.0, "fde": 2.0, "lat": 0.1, "lon": 0.2, "v0": v, "gt_drop": g, "pred_drop": p}) for key, v, g, p in samples]
    monkeypatch.setattr(finetune, "expert_ade", fake)
    cb = finetune.FtSelect({"general": [], "popout_onset": []}, every_steps=1, rel_worse=None, brake_s=1.2)
    cb.samples = {"general": GEN, "popout_onset": ONSET}
    args, model = SimpleNamespace(output_dir=str(tmp_path)), torch.nn.Linear(1, 1)
    cb.on_train_begin(args, SimpleNamespace(global_step=0, max_steps=10, is_world_process_zero=True), None, model=model)
    met = cb.st["history"][0]["metrics"]
    assert seen == [1.2, 1.2]
    assert met["general"] == {"ade": 1.0, "fde": 2.0, "lat": 0.1, "lon": 0.2, "brake": 0.5, "false_brake": 0.5, "brake_hit": 1.0}
    o = met["popout_onset"]                                                          # 1.5 counts as braking
    assert o["brake"] == o["brake_hit"] == round(2 / 3, 5) and math.isnan(o["false_brake"])
    assert finetune.brake_rates([(k, dict(v0=v, gt_drop=g, pred_drop=p)) for k, v, g, p in GEN], drop=2.1)["brake"] == 0.25
    assert finetune.brake_rates([("x", {"ade": 1.0})]) == {}                          # rows without the keys: nothing added
    rows = [json.loads(l) for l in open(tmp_path / "ft_val_rows.jsonl")]
    assert len(rows) == 7 and rows[0]["pred_drop"] == 2.0 and rows[0]["set"] == "general"
    val, _ = ft_metrics.val_rows(tmp_path)                                           # val.csv rows carry the new keys
    assert {r["set"]: r["brake"] for r in val} == {"general": 0.5, "popout_onset": round(2 / 3, 5)}
