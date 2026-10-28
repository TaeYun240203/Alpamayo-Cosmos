"""QAD step 1 (one process, one GPU): AutoQuantize the stage-2 student (morai_vla.qad) on `quant.calib_samples` train
batches, scoring with the student's own training loss (T* in the loop; R2 student_qad_r2: GT losses, no T*), then save the quantized student:
<out>/ = HF student dir + modelopt_state.pth (later loads restore the quantizers) + quant_manifest.json + search.json.
Usage (cwd train/): python scripts/qad_search.py --config-name student_qad paths.teacher=<T*> model.init=<s2 final> \
    data.root=<share> data.train_dataset.rule_ok=<rule_ok parquet> +out=<dir>"""
import json, time
from pathlib import Path

import hydra, hydra.utils as hyu, numpy as np, torch
from alpamayo.common import misc

from morai_vla import qad


def to_dev(x):
    return {k: to_dev(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}


@hydra.main(version_base=None, config_path="../configs", config_name="student_qad")
def main(cfg):
    misc.seed_everything(42)
    q, out = cfg.quant, Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    model = hyu.instantiate(cfg.model, _convert_="partial").cuda()
    mix = cfg.data.train_dataset._target_.endswith("MixtureDataset")             # R2 (student_qad_r2): no max_samples there
    ds = hyu.instantiate(cfg.data.train_dataset, _convert_="partial", model_config=model.config, **({} if mix else {"max_samples": q.calib_samples}))
    if mix:                                                                       # evenly over the shuffled schedule = its shares
        ds = torch.utils.data.Subset(ds, np.linspace(0, len(ds) - 1, q.calib_samples).round().astype(int).tolist())
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=4)
    batches = _Lazy(list(dl))
    t = time.time()
    model, state = qad.search(model, batches, q.effective_bits, q.num_calib_steps, q.num_score_steps)
    model.save_pretrained(out)
    meta = qad.write_manifest(model, out)
    json.dump({"effective_bits_target": q.effective_bits, "seconds": round(time.time() - t), **meta,
               "best": {k: v for k, v in state.get("best", {}).items() if isinstance(v, (int, float, str))}},
              open(out / "search.json", "w"), indent=1, default=str)
    print(json.dumps(meta))


class _Lazy:
    """batches kept on the CPU, moved to the GPU one at a time (pixel values ~35 MB per sample)."""
    def __init__(self, cpu): self.cpu = cpu
    def __len__(self): return len(self.cpu)
    def __iter__(self): return (to_dev(b) for b in self.cpu)


if __name__ == "__main__":
    main()
