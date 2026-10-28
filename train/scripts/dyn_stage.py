"""Box-side staging of the dyn_relabel virtual episodes (scripts/dyn_relabel.py; B2 keeps no links). Run once after
step0_r2 and its _dyn_relabel/ are downloaded: for every virtual episode of _dyn_relabel/dyn_index.parquet, ROOT/<virtual>/
gets relative links to the original episode's sensor files and ROOT/_ego/<virtual>.npz links to _dyn_relabel/ego/. The
loader is unchanged: the dyn sources read index _dyn_relabel/dyn_index.parquet. Idempotent; a virtual episode whose
original (or ego table) is missing is skipped and counted (MoraiDataset would skip it too).
Usage: python train/scripts/dyn_stage.py R2_ROOT
"""
import sys
from pathlib import Path

import pandas as pd

SENSORS = ("cams.bin", "cams_index.npz", "lidar.bin", "lidar_index.npz")   # what morai_vla.data.Episode opens
EXTRA = ("meta.json",)


def main():
    root = Path(sys.argv[1])
    vi = pd.read_parquet(root / "_dyn_relabel/dyn_index.parquet", columns=["episode", "src_episode"]).drop_duplicates("episode")
    made = skipped = 0
    for v, src in zip(vi.episode, vi.src_episode):
        ego = root / "_dyn_relabel/ego" / f"{v}.npz"
        if not ego.exists() or not all((root / src / f).exists() for f in SENSORS): skipped += 1; continue
        d = root / v; d.mkdir(exist_ok=True)
        for f in SENSORS + EXTRA:
            if (root / src / f).exists() and not (d / f).is_symlink(): (d / f).symlink_to(Path("..") / src / f)
        e = root / "_ego" / f"{v}.npz"
        if not e.is_symlink(): e.symlink_to(Path("../_dyn_relabel/ego") / ego.name)
        made += 1
    print(f"dyn_stage: {made} virtual episodes staged, {skipped} skipped (original episode or ego table missing)")
    assert made, "nothing staged: is _dyn_relabel/ next to the episode folders?"


if __name__ == "__main__":
    main()
