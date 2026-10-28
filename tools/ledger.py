"""Coverage ledger (DATA_PLAN.md §6): clean training hours per mission cell x weather x hour.

    uv run --python 3.12 --with-requirements tools/scorer/requirements.txt python tools/ledger.py [labels dir ...]

Reads every *.labels.parquet (live scoring output) under the given dirs (default: the Mac stack data dirs and the
local cache that `--pull` fills from winpc D:), the env-<w1|w2>.log weather/time switch logs, and prints hours of
traj_ok frames per cell against the plan's targets.
"""
import argparse
import bisect
import glob
import json
import os
import subprocess
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

MAC = os.path.expanduser("~/.claude-morai-mac")
CACHE = os.path.join(MAC, "labels-cache")
REMOTE = "winpc-wsl:/mnt/d/share/2026_MORAI_collect_20261018/"
# route section (section_annotator numbering) -> cell; targets in clean hours (DATA_PLAN.md §6)
CELLS = {3: "C1", 7: "C1", 4: "C2", 5: "C3", 6: "C3", 9: "C4", 10: "C5", 11: "C5", 12: "C5", 13: "C6", 14: "C6", 15: "C6"}
TARGET_H = {"C0": 4, "C1": 8, "C2": 7, "C3": 7, "C4": 5, "C5": 5, "C6": 4}
FRAME_S = 0.1


def env_table(w):
    rows = []
    for f in (os.path.join(MAC, f"env-{w}.log"), os.path.join(CACHE, f"env-{w}.wsl.log")):  # Mac runs; WSL runs (10/21-)
        for line in open(f) if os.path.exists(f) else []:
            t, wx, hr = line.split()
            rows.append((float(t), f"{wx[0]}{hr}"))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def env_at(table, t):
    ts, vals = table
    i = bisect.bisect_right(ts, t) - 1
    return vals[i] if i >= 0 else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--pull", action="store_true", help="rsync labels from winpc D: into the local cache first")
    args = ap.parse_args()
    if args.pull:
        os.makedirs(CACHE, exist_ok=True)
        subprocess.run(["rsync", "-a", "--include=*/", "--include=*.labels.parquet", "--exclude=*", REMOTE, CACHE], check=False)
        for w in ("w1", "w2"):  # runs orchestrated inside WSL log their weather/time switches there
            subprocess.run(["rsync", "-a", f"winpc-wsl:.claude-morai-mac/env-{w}.log", os.path.join(CACHE, f"env-{w}.wsl.log")],
                           check=False, stderr=subprocess.DEVNULL)
    dirs = args.dirs or [os.path.join(MAC, "win-1/data"), os.path.join(MAC, "win-2/data"), CACHE]
    files = sorted({os.path.realpath(f) for d in dirs for f in glob.glob(os.path.join(d, "**/*.labels.parquet"), recursive=True)})
    envs = {w: env_table(w) for w in ("w1", "w2")}
    hours, eps = defaultdict(float), []
    for f in files:
        name = os.path.basename(f).split(".")[0]
        w = "w2" if "-w2-" in name else "w1"
        df = pd.read_parquet(f, columns=["bag_time_ns", "route_section", "traj_ok"])
        summary = json.loads(pq.read_schema(f).metadata[b"scorer_summary"])
        eps.append((name, summary.get("score_s"), summary.get("competition_failed"), round(float(df.traj_ok.mean()), 2), len(df)))
        ok = df[df.traj_ok]
        for sec, t in zip(ok.route_section, ok.bag_time_ns):
            hours[(CELLS.get(int(sec), "C0"), env_at(envs[w], t / 1e9))] += FRAME_S / 3600
    print(f"{len(files)} episodes")
    for e in eps[-10:]:
        print("  ", *e)
    envs_seen = sorted({k[1] for k in hours})
    print("cell  target  total  " + "  ".join(f"{e:>5}" for e in envs_seen))
    for c in sorted(TARGET_H):
        row = [hours[(c, e)] for e in envs_seen]
        print(f"{c:4} {TARGET_H[c]:6.1f} {sum(row):6.2f}  " + "  ".join(f"{v:5.2f}" for v in row))


if __name__ == "__main__":
    main()
