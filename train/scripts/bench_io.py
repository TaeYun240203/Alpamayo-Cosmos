"""Data-loader throughput (samples/s, tokenized + collated, no model) for a stage config, to compare with the GPU step rate.
Usage (cwd train/): python scripts/bench_io.py --config-name step2 data.root=<share> paths.alpamayo15=<release dir> [+n=400]
Only the model config is read (no weights)."""
import json, time
from pathlib import Path

import hydra, hydra.utils as hyu, torch
from alpamayo.utils.checkpoint_utils import remap_targets
from alpamayo_r1.config import AlpamayoR1Config

from morai_vla.model import A15_TO_A1


@hydra.main(version_base=None, config_path="../configs", config_name="step2")
def main(cfg):
    c = json.load(open(Path(cfg.paths.alpamayo15) / "config.json")); remap_targets(c, A15_TO_A1)
    for k in ("model_type", "architectures", "transformers_version", "dtype", "torch_dtype", "traj_token_ids"): c.pop(k, None)
    c["vlm_name_or_path"] = cfg.model.vlm_name_or_path
    mc = AlpamayoR1Config(**c)
    t = time.time()
    ds = hyu.instantiate(cfg.data.train_dataset, _convert_="partial", model_config=mc)
    print(f"dataset init {time.time() - t:.1f} s, {len(ds)} samples")
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=mc)
    tr = cfg.trainer; n = cfg.get("n", 400)
    dl = torch.utils.data.DataLoader(ds, batch_size=tr.per_device_train_batch_size, shuffle=True, collate_fn=collate,
                                     num_workers=tr.dataloader_num_workers, prefetch_factor=tr.dataloader_prefetch_factor, pin_memory=True)
    it = iter(dl); [next(it) for _ in range(max(8, tr.dataloader_num_workers))]                  # warm-up
    t = time.time(); k = 0
    for _ in range(n // tr.per_device_train_batch_size):
        k += len(next(it)["key"])
    dt = time.time() - t
    print(f"{k} samples in {dt:.1f} s: {k / dt:.1f} samples/s with {tr.dataloader_num_workers} workers")


if __name__ == "__main__":
    main()
