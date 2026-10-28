"""Single-GPU check of the R2 fine-tune speed-ups before an 8-GPU run (README "R2 속도 최적화", 10/28): the real s2 student
+ T* (student_ft config, R2 sources only), real R2 micro-batches, forward + backward without an optimizer or DeepSpeed.
  identity  the same micro-batches (inputs built once, RNG seeded per micro-batch) with morai_vla.speed off, off again (the
            GPU's own run-to-run noise) and on: every loss component and the gradient norm of each trained part
  timing    s per micro-batch (synchronized) with speed off / on; under sdpa also per-device batch 2 (same samples, half
            the micro-batches; batch 2 is a recipe question: README); CUDA kernel launches and GPU kernel time per
            micro-batch (torch.profiler): launches >> and kernel time << wall time = dispatch-bound
  measure   early_stop.expert_ade on --rows R2 validation rows (the dynamic set): one row per prefill with speed off vs
            --eval-batch rows with speed on: per-row ADE max |difference|, mean ADE, seconds
Writes <out>/probe.json. Usage (cwd train/, the box venv, one GPU free):
  python scripts/ft_speed_probe.py --t2 <t2 final> --s2 <s2 final> --r2 /data/step0_r2 --out /workspace/out/probe \\
         [--attn sdpa|flash_attention_2] [--n 8] [--rows 32] [--eval-batch 4]"""
import argparse, json, platform, time
from pathlib import Path

import hydra.utils as hyu, torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from morai_vla import speed
from morai_vla.early_stop import expert_ade
from morai_vla.finetune import FtValSamples

ON = {"vit_sdpa_grouped": True, "teacher_vit_only": True, "lazy_loss_log": True}


def to_dev(x):
    return {k: to_dev(v) for k, v in x.items()} if isinstance(x, dict) else x.cuda() if torch.is_tensor(x) else x


def step(model, parts, batch, seed):
    """one micro-batch forward + backward -> (loss components, grad norm per part)."""
    model.loss_sums.clear(); model.zero_grad(set_to_none=True); torch.manual_seed(seed)
    model(**to_dev(batch)).loss.backward()
    comps = {k: float(s) / n for k, (s, n) in model.loss_sums.items() if "/" not in k}
    norm = lambda ps: float(torch.sqrt(sum((p.grad.float().pow(2).sum() for p in ps if p.grad is not None), torch.zeros((), device="cuda"))))
    return comps, {name: norm(ps) for name, ps in parts.items()}


def diff(a, b):
    return {k: abs(a[k] - b[k]) for k in a if k in b}


def timed(model, parts, batches, cfg_speed, prof=False):
    speed.apply(model, cfg_speed)
    for b in batches[:2]: step(model, parts, b, 0)                                  # warm-up (allocator, kernels, T* to the GPU)
    torch.cuda.synchronize(); t = time.time()
    for j, b in enumerate(batches[2:]): step(model, parts, b, j)
    torch.cuda.synchronize(); out = {"s_per_microbatch": round((time.time() - t) / max(len(batches) - 2, 1), 3)}
    if prof:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            step(model, parts, batches[2], 0); torch.cuda.synchronize()
        ev = p.key_averages()
        out["kernel_launches"] = int(sum(e.count for e in ev if "LaunchKernel" in e.key))
        out["gpu_kernel_s"] = round(sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ev) / 1e6, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("--t2", "--s2", "--r2", "--out"): ap.add_argument(k, required=True)
    ap.add_argument("--attn", default="sdpa"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=4)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg = compose("student_ft", overrides=[f"data.root={a.r2}", f"paths.teacher={a.t2}", f"paths.s2={a.s2}", f"paths.r2_root={a.r2}",
                                               f"paths.rule_ok={a.s2}/../rule_ok.parquet", f"paths.output_dir={out}", f"model.attn_implementation={a.attn}",
                                               "~data.train_dataset.sources.old_recovery", "~data.train_dataset.sources.old_curve",
                                               "~data.train_dataset.sources.start_relabel", "data.train_dataset.sources.replay.dataset.exclude=null"])
    model = hyu.instantiate(cfg.model, _convert_="partial").cuda().train()
    parts = {name: [p for path in paths for p in model.get_submodule(path).parameters() if p.requires_grad] for name, paths in model.PARTS.items()}
    ds = hyu.instantiate(cfg.data.train_dataset, _convert_="partial", model_config=model.config)
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
    order = list(range(min(len(ds), 4 * a.n + 4)))                                  # schedule order = the mixture as trained
    items = [ds[k] for k in order]                                                  # built once: augmentation jitters per call
    b1 = [collate([x]) for x in items[: 2 * a.n + 2]]
    res = {"gpu": torch.cuda.get_device_name(), "cpu": platform.processor() or platform.machine(), "attn": a.attn,
           "gt_rows": sum(bool(b["gt_row"][0]) for b in b1)}
    # identity: off, off again, on on the same micro-batches
    rows = []
    for j, b in enumerate(b1[: a.n]):
        speed.apply(model, {}); off = step(model, parts, b, j); off2 = step(model, parts, b, j)
        speed.apply(model, ON); on = step(model, parts, b, j)
        rows.append({"gt_row": bool(b["gt_row"][0]), "noise_loss": diff(off[0], off2[0]), "on_loss": diff(off[0], on[0]),
                     "noise_grad": diff(off[1], off2[1]), "on_grad": diff(off[1], on[1]), "loss_off": off[0], "grad_off": off[1]})
    mx = lambda key: max((max(r[key].values(), default=0.0) for r in rows), default=0.0)
    res["identity"] = {"max_noise_loss": mx("noise_loss"), "max_on_loss": mx("on_loss"), "max_noise_grad": mx("noise_grad"),
                       "max_on_grad": mx("on_grad"), "rows": rows}
    # timing
    res["timing"] = {"batch1_off": timed(model, parts, b1, {}, prof=True), "batch1_on": timed(model, parts, b1, ON, prof=True)}
    if a.attn != "flash_attention_2":                                               # FA2 rejects the padded prefix of a batch of 2
        b2 = [collate(items[i:i + 2]) for i in range(0, 4 * a.n + 4, 2)]
        res["timing"] |= {"batch2_off": timed(model, parts, b2, {}), "batch2_on": timed(model, parts, b2, ON)}
    res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    # measurement: batch 1 speed off (= the main run's default) vs eval-batch speed on
    model.zero_grad(set_to_none=True)
    dyn = dict(OmegaConf.to_container(cfg.callbacks.ft_select.sets.dynamic, resolve=True)[0], n=a.rows)
    samples = FtValSamples([dyn], model.config)
    got = {}
    for name, b, cfg_speed in (("batch1_off", 1, {}), (f"batch{a.eval_batch}_on", a.eval_batch, ON)):
        speed.apply(model, cfg_speed); torch.cuda.synchronize(); t = time.time()
        got[name] = (expert_ade(model, samples, 6, 4, detail=True, batch=b), round(time.time() - t, 1))
    (r1, t1), (rb, tb) = got.values()
    res["measure"] = {"rows": len(r1), "seconds": {k: v[1] for k, v in got.items()}, "max_row_ade_diff": max(abs(x["ade"] - y["ade"]) for (_, x), (_, y) in zip(r1, rb)),
                      "mean_ade": {k: sum(e["ade"] for _, e in v[0]) / len(v[0]) for k, v in got.items()}}
    speed.apply(model, {})
    json.dump(res, open(out / "probe.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "identity"} | {"identity": {k: v for k, v in res["identity"].items() if k != "rows"}}, indent=1))


if __name__ == "__main__":
    main()
