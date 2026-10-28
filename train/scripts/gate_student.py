"""Student gate (MODEL_PLAN 4.5, after stage 1): ADE of the student (its frozen, transplanted expert reading the student
KV) against T* on the same samples, inference format (no CoC, Euler 10), K samples from the SAME initial noise for both
(so st_ade = student-vs-T* distance of matched samples measures how far the student KV moves the expert).
Usage (cwd train/): python scripts/gate_student.py --config-name student1 paths.teacher=<T*> +student=<student ckpt | start> \
    data.root=<share> +out=<dir> [+n=1000] [+k=6] [+dataset=train_dataset]   (default: the val_dataset of the config)
Writes <out>/gate.json (per sample) and <out>/summary.json (means, by cell, ratio = student ADE / T* ADE).
+shard=k/n: rows k::n only (each keeps its own seed = its row number) and gate.json only; merge(out, shard dirs) writes the
summary of all of them (scripts/pipeline_student.sh GATE_SHARDS: one process per GPU, 10/28).
R2 (user 10/28, --config-name student_ft: paths.teacher null): the student alone (no T*, no ratio); with paths.start_relabel
the summary also has student_ade_wo_relabel (the gate rows without the start-relabelled ones). +sets=[dynamic,curve,
start_relabel]: instead of the gate rows, those records (ft_select sets; start_relabel = its val rows with the relabelled
future) as early_stop.expert_ade means -> <out>/sets.json."""
import json, logging
from pathlib import Path

import hydra, hydra.utils as hyu, numpy as np, torch
from omegaconf import OmegaConf

from morai_vla import speed
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn
from morai_vla.student import StudentVLA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


@hydra.main(version_base=None, config_path="../configs", config_name="student1")
def main(cfg):
    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    K = cfg.get("k", 6)
    m = cfg.model
    speed.apply(None, cfg.get("speed"))                                               # vit_sdpa_grouped (math unchanged)
    shard, shards = (int(x) for x in str(cfg.get("shard", "0/1")).split("/"))
    T = TeacherVLA.build(cfg.paths.teacher, vlm_name_or_path=m.teacher_vlm, attn_implementation="sdpa", dtype=m.dtype, device_map={"": 0}).eval() \
        if cfg.paths.get("teacher") else None
    if cfg.student == "start":                                                         # the stage's starting student (before training)
        S = hyu.instantiate(m, _convert_="partial", distill=False, attn_implementation="sdpa").cuda().eval()
    else:
        S = StudentVLA.build(init=cfg.student, vlm_name_or_path=m.vlm_name_or_path, attn_implementation="sdpa", dtype=m.dtype, device_map={"": 0}).eval()
    if cfg.get("sets"):
        return records(cfg, S, out, K)
    dcfg = OmegaConf.to_container(cfg.data[cfg.get("dataset", "val_dataset")], resolve=True)
    dcfg.update(augment=False, max_samples=cfg.get("n", dcfg["max_samples"]),
                vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn", "generation_mode": True})
    ds = hyu.instantiate(dcfg, _convert_="partial", model_config=S.config)
    rows = []
    for i in range(shard, len(ds), shards):
        item = ds[i]
        r = dict(idx=i, key=item["key"], cell=str(ds.rows.cell.iloc[i]))
        gt = item["ego_future_xyz"][0, :, :2]
        pred = {}
        for name, model in (("teacher", T), ("student", S)) if T else (("student", S),):
            b = collate_fn([item], model_config=model.config)
            b = {k: ({kk: vv.cuda() for kk, vv in v.items()} if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            torch.manual_seed(i)                                                       # same initial noise for both models
            xyz, _ = model.sample_trajectories(b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"], num_samples=K)
            pred[name] = xyz[0, ..., :2].float().cpu()
            err = (pred[name] - gt).norm(dim=-1)                                      # (K, 64)
            r |= {f"{name}_ade": float(err.mean()), f"{name}_fde": float(err[:, -1].mean()), f"{name}_minade": float(err.mean(1).min())}
        if T: r["st_ade"] = float((pred["student"] - pred["teacher"]).norm(dim=-1).mean())
        rows.append(r)
        if i % 50 == 0: logging.info(json.dumps(r))
    if shards > 1:
        json.dump(rows, open(out / "gate.json", "w"), indent=1); return
    summarize(rows, out, cfg.paths.get("start_relabel"))


def records(cfg, S, out, K):
    from morai_vla.early_stop import expert_ade
    from morai_vla.finetune import FtValSamples
    sets = OmegaConf.to_container(cfg.callbacks.ft_select.sets, resolve=True)
    sets["start_relabel"] = [{"root": cfg.data.root, "split": ["val", "val_asset"], "keep": cfg.paths.start_relabel, "n": 300}]
    res = {}
    for name in cfg.sets:
        rows = expert_ade(S, FtValSamples(sets[name], S.config), K, 4, detail=True)
        res[name] = {k: round(float(np.mean([e[k] for _, e in rows])), 4) for k in ("ade", "fde", "lat", "lon")} | {"n": len(rows)}
    json.dump(res, open(out / "sets.json", "w"), indent=1); print(json.dumps(res))


def merge(out, parts, relabel=None):
    summarize(sorted((r for p in parts for r in json.load(open(Path(p) / "gate.json"))), key=lambda r: r["idx"]), Path(out), relabel)


def summarize(rows, out, relabel=None):
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "gate.json", "w"), indent=1)
    keys = [k for k in rows[0] if k not in ("idx", "key", "cell")]
    mean = lambda rs: {k: round(float(np.mean([r[k] for r in rs])), 3) for k in keys} | {"n": len(rs)}
    s = mean(rows)
    if "teacher_ade" in s: s["ratio"] = round(s["student_ade"] / s["teacher_ade"], 3)
    if relabel:                                                                       # R2: without the old "wait" GT rows
        import pandas as pd
        x = pd.read_parquet(relabel, columns=["episode", "i"]); skip = set(x.episode.astype(str) + "/" + x.i.astype(str))
        keep = [r for r in rows if r["key"] not in skip]
        s |= {"student_ade_wo_relabel": round(float(np.mean([r["student_ade"] for r in keep])), 3), "n_wo_relabel": len(keep)}
    s["by_cell"] = {c: mean([r for r in rows if r["cell"] == c]) for c in sorted({r["cell"] for r in rows})}
    json.dump(s, open(out / "summary.json", "w"), indent=1); print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
