"""DAgger spool -> training rows (train/dagger/README.md). One episode dir per actor episode, as the eval repo's
src/e2e_driver/tools/rl_episode.py writes it (header there), plus the shadow labeler's file:

  <spool>/<actor>/epK/steps.jsonl        one line per engine decision (10 Hz): t_sim (simulator s), req, out (file names)
  <spool>/<actor>/epK/K.req.bin, K.out.bin   the engine's wire frames (request = the full observation; response carries
                                         lora_version)
  <spool>/<actor>/epK/done.json          written last by the actor
  <spool>/<actor>/epK/dagger_labels.npz  the labeler's (written after done.json):
      step    int32 (N,)        line number in steps.jsonl
      ak      float32 (N,64,2)  (a m/s^2, kappa 1/m), 0.1 s apart, 6.4 s
      valid   bool (N,64)
      source  int8 (N,)         0 planner, 1 official route lane (large lane departures)
      signal  int8 (N,)         optional: the light the row follows at t0, _ego code (0 none / unknown, 1 red, 2 yellow, 3 green)
      stop_d  float32 (N,)      optional: front axle -> next judged stop line ahead (m, inf if none)
Rows (user 10/28): of the 10 Hz decisions, the one nearest each point of a 0.25 s grid of simulator time (4 Hz), every
such row trained on. A 4 Hz row without a label, or whose label has no valid point, cannot be trained: counted, kept out.
Row tags (configs/dagger.yaml dagger.tags, users 10/28) give the row's loss weight (dagger.weights; both tags: the larger):
  signal  label signal red / yellow and stop_d <= signal_stop_d, or red and v < stop_v (critic_obs v: standing at the light)
  lane    label source 1, or |critic_obs lane_offset| >= lane_offset (left +, from the lane centreline)
critic_obs is rl_critic.FIELDS order (fixed, new fields appended): v = 0, lane_offset = 7.
The spool comes down from B2 (b2 sync writes *.b2.sync.tmp and renames): an episode is read once done.json, the labels
and every req file of its 4 Hz rows exist.
"""
import json
from pathlib import Path

import numpy as np, torch

GRID = 0.25
SOURCES = ("planner", "route")                                  # dagger_labels.npz source 0, 1
OBS_V, OBS_LANE = 0, 7                                          # rl_critic.FIELDS: v, lane_offset
TAGS = dict(signal_stop_d=60.0, stop_v=0.5, lane_offset=0.5)


def rows_4hz(t_sim, grid=GRID):
    """indices of the decisions nearest each grid point of simulator time (one per grid point, the grid point being each
    decision's nearest one); a grid point no decision rounds to (actor late > 0.25 s) has none."""
    t = np.asarray(t_sim, float)
    g = np.round(t / grid)
    best = {}
    for i, (gi, d) in enumerate(zip(g, np.abs(t - g * grid))):
        if gi not in best or d < best[gi][1]: best[gi] = (i, d)
    return np.array(sorted(i for i, _ in best.values()), int)


def ready(d):
    d = Path(d)
    return (d / "done.json").exists() and (d / "dagger_labels.npz").exists() and (d / "steps.jsonl").exists()


def episodes(spool):
    """ready episode dirs, oldest labels first (b2 sync keeps the source mtimes)."""
    out = [d for d in (f.parent for f in Path(spool).glob("*/*/dagger_labels.npz")) if ready(d)]
    return sorted(out, key=lambda d: ((d / "dagger_labels.npz").stat().st_mtime, str(d)))


def read_labels(f, n_steps):
    """dagger_labels.npz, checked (it comes from another session's labeler, through B2): shapes, dtypes, step range;
    a non-finite (a, kappa) point is made invalid."""
    z = np.load(f)
    step, ak, valid, src = (z[k] for k in ("step", "ak", "valid", "source"))
    n = len(step)
    assert step.shape == (n,) and ak.shape == (n, 64, 2) and valid.shape == (n, 64) and src.shape == (n,), (f, step.shape, ak.shape, valid.shape, src.shape)
    assert np.issubdtype(step.dtype, np.integer) and valid.dtype == bool and set(np.unique(src)) <= {0, 1}, f
    assert ((step >= 0) & (step < n_steps)).all() and len(np.unique(step)) == n, f"{f}: step out of range or repeated"
    valid = valid & np.isfinite(ak).all(-1)
    sig = z["signal"].astype(int) if "signal" in z else np.zeros(n, int)
    stop_d = z["stop_d"].astype(float) if "stop_d" in z else np.full(n, np.inf)
    assert sig.shape == stop_d.shape == (n,), f
    return {int(s): (np.nan_to_num(ak[k]).astype(np.float32), valid[k], int(src[k]), int(sig[k]), float(stop_d[k])) for k, s in enumerate(step)}


def row_tags(src, sig, stop_d, obs, t=TAGS):
    """(signal, lane) tags of one row (module docstring)."""
    v = obs[OBS_V] if obs else np.inf
    signal = (sig in (1, 2) and stop_d <= t["signal_stop_d"]) or (sig == 1 and v < t["stop_v"])
    lane = src == 1 or (obs is not None and abs(obs[OBS_LANE]) >= t["lane_offset"])
    return bool(signal), bool(lane)


def row_weight(signal, lane, weights):
    w = [weights["signal"]] * signal + [weights["lane"]] * lane
    return float(max(w)) if w else 1.0


def episode_rows(d, wire, tags=TAGS, weights=None):
    """-> (rows, stats), or None while a req/out file of a 4 Hz row is still on its way down. rows: dicts {ep, j, req,
    ak, valid, source, version, signal, lane, w}, one per trainable 4 Hz row."""
    d = Path(d)
    steps = [json.loads(line) for line in open(d / "steps.jsonl")]
    lab = read_labels(d / "dagger_labels.npz", len(steps))
    sel = rows_4hz([s["t_sim"] for s in steps])
    if not all((d / steps[j][k]).exists() for j in sel for k in ("req", "out")): return None
    weights = weights or {"signal": 1.0, "lane": 1.0}
    rows, st = [], {"episode": str(d), "decisions": len(steps), "rows_4hz": len(sel), "unlabeled": 0, "no_valid": 0, "planner": 0, "route": 0,
                    "tag_signal": 0, "tag_lane": 0, "label_signal": 0}
    for j in sel:
        if j not in lab: st["unlabeled"] += 1; continue
        ak, valid, src, sig, stop_d = lab[j]
        if not valid.any(): st["no_valid"] += 1; continue
        v = wire.unpack((d / steps[j]["out"]).read_bytes()).get("lora_version")
        signal, lane = row_tags(src, sig, stop_d, steps[j].get("critic_obs"), tags)
        rows.append(dict(ep=str(d), j=int(j), req=steps[j]["req"], ak=ak, valid=valid, source=src, version=0 if v is None else int(v),
                         signal=signal, lane=lane, w=row_weight(signal, lane, weights)))
        st[SOURCES[src]] += 1; st["tag_signal"] += signal; st["tag_lane"] += lane; st["label_signal"] += sig > 0
    st["rows"] = len(rows)
    return rows, st


def sample(row, server, model_config):
    """a DAgger row -> one training item: the engine's own build_sample on the request (the prompt the actor's model saw),
    the training-mode prompt (future placeholders + label mask), the label."""
    from morai_vla.processor import get_preprocess_fn
    from rl.episodes import DROP, e2e
    wire, build_sample = e2e(server)
    s = build_sample(wire.unpack((Path(row["ep"]) / row["req"]).read_bytes()), key=f"{Path(row['ep']).parent.name}/{Path(row['ep']).name}/{row['j']}")
    s["tokenized_data"] = get_preprocess_fn(model_config, generation_mode=False)(data=s)
    for k in DROP: s.pop(k, None)
    s.update(ak=torch.from_numpy(row["ak"]), valid=torch.from_numpy(row["valid"]), label_source=row["source"],
             w=torch.tensor(float(row.get("w", 1.0))), tag=int(row.get("signal", False)) + 2 * int(row.get("lane", False)))
    return s
