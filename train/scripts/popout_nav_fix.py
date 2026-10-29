"""nav_text of the pop-out virtual rows from their own row (main 10/29 audit, reports/popout_audit_20261029.md): popout_relabel
gives a row the nav_text of the nearest index row of its episode (stale distance) or, for the ~80 % of episodes without index
rows, the bare "Follow the road" (no speed text) that no R1 / R2 row has: a string the model could read as "stop". Here every
row gets build_index.nav_text of its own source-row path_s (same route maneuvers and speed-exempt rule as every index).
The recording session's dyn_index.parquet stays as it is; the fixed copy is _dyn_relabel/dyn_index_nav.parquet (same rows
and keys, only nav_text changes), which student_lwf's pop-out sources read.
Usage (cwd train/): python scripts/popout_nav_fix.py <step0_popout>   (needs _dyn_relabel/dyn_index.parquet and _ego/<src>.npz)
"""
import sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morai_vla.build_index import data, maneuvers, nav_text  # noqa: E402


def main():
    root = Path(sys.argv[1]); M, lap, exempt = maneuvers(), float(data.path_xy()[1][-1]), data.kmap().speed_exempt_s
    v = pd.read_parquet(root / "_dyn_relabel/dyn_index.parquet")
    s = {e: np.load(root / "_ego" / f"{e}.npz")["path_s"].astype(float) for e in v.src_episode.unique()}
    new = [nav_text(float(s[e][i]), M, lap, exempt) for e, i in zip(v.src_episode, v.i)]
    bare = v.nav_text == "Follow the road"
    print(f"dyn_index {len(v)} rows: bare 'Follow the road' {int(bare.sum())} ({bare.mean():.1%}), nav_text changed {int((v.nav_text != new).sum())}")
    v.assign(nav_text=new).to_parquet(root / "_dyn_relabel/dyn_index_nav.parquet", index=False)
    assert not (pd.Series(new) == "Follow the road").any()
    print("wrote", root / "_dyn_relabel/dyn_index_nav.parquet")


if __name__ == "__main__":
    main()
