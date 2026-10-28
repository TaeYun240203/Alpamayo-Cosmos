"""Episode -> keyframes -> history frames -> VLM observation -> CoC records (JSONL + official-format YAML)."""
import json, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from . import compose, data, events, frames, observe
from .vlm import BudgetExceeded

VERSION = "morai-coc-1.1"   # 1.1: drums cited only when seen; slow right turns at int551 keep the turn decision


def qa_pick(ep, t0, rate):
    return zlib.crc32(f"{ep}:{t0}".encode()) % 1000 < rate * 1000


def label_episode(ep_name, provider, vlm, out, workers=8, qa_rate=0.1, keep_frames=False):
    out = Path(out); (out / "labels").mkdir(parents=True, exist_ok=True)
    ep = data.load_episode(ep_name)
    K = events.detect(ep)
    if not K:
        (out / "labels" / f"{ep_name}.jsonl").write_text(""); return []
    for j, k in enumerate(K):
        k["key"] = f"k{j:03d}_{k['t0']:05d}"; k["qa"] = qa_pick(ep_name, k["t0"], qa_rate)
    need = [k for k in K if observe.comps_to_check(k["components"]) or k["npc"] or k["qa"]]
    # history frames (t -2..0) for observation; the QA sample keeps the full 2 Hz window for review / validation
    req = {k["key"]: (k["frames_bag_ns"] if k["qa"] else k["frames_bag_ns"][:5]) for k in need}
    dirs = provider.fetch(ep_name, req) if req else {}

    def one(k):
        hist = []
        if k["key"] in dirs:
            fs = sorted(Path(dirs[k["key"]]).glob("*.jpg"))[:5]
            hist = [(f"t={observe.HIST_T[i]:+.1f} s", f.read_bytes()) for i, f in enumerate(fs)]
        ver, veh = {}, None
        if hist and observe.comps_to_check(k["components"]): ver = observe.verify(vlm, k["components"], hist)
        if hist and k["npc"]: veh = observe.npc(vlm, k["npc"], hist)
        return record(out, ep, k, ver, veh, no_frames=len(hist) < 5 and bool(observe.comps_to_check(k["components"]) or k["npc"]))

    recs, err = [], None
    with ThreadPoolExecutor(workers) as ex:
        for f in [ex.submit(one, k) for k in K]:
            try: recs.append(f.result())
            except BudgetExceeded as e: err = e
    if err: raise err                                   # an interrupted episode stays unwritten, so a rerun redoes it
    with open(out / "labels" / f"{ep_name}.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    if not keep_frames:
        for k in K:
            if not k["qa"]: frames.drop(ep_name, k["key"])
    if err: raise err
    return recs


def record(out, ep, k, ver, veh, no_frames=False):
    """compose one keyframe and write its official-format YAML."""
    status, dec, comps, text = compose.compose(k, ver, veh)
    if no_frames: status, text = "no_frames", None
    rec = dict(version=VERSION, episode=ep.name, scene=ep.scene_id, key=k["key"], kind=k["kind"], t0=k["t0"], tau0=k["tau0"],
               bag_ns=k["frames_bag_ns"][4], frames_bag_ns=k["frames_bag_ns"], path_s=k["path_s"], section=k["section"], cell=k["cell"],
               env=k["env"], clean=k["clean"], qa=k["qa"], status=status, decision=dec, critical_components=comps,
               candidates=k["components"], verification=ver, vehicles=veh, coc=text)
    if text:
        d = Path(out) / "official" / ep.name; d.mkdir(parents=True, exist_ok=True)
        doc = {"clip_id": ep.name, "event_start_timestamp": rec["bag_ns"], "coc_prompt_version": VERSION,
               "final_content": {"ego_behavior_schema": {"effect_on_ego_behavior": text}},
               "structured": {"decision": dec, "critical_components": comps, "tau0": rec["tau0"], "kind": rec["kind"]}}
        yaml.safe_dump(json.loads(json.dumps(doc, default=float)),          # numpy scalars -> plain floats
                       open(d / f"cot_{rec['bag_ns']}.yaml", "w"), sort_keys=False, allow_unicode=True)
    return rec


def recompose_episode(ep_name, old, out, invalid=()):
    """re-run events + compose on an earlier run's records with their stored VLM answers (no Codex calls).
    old: that run's records of this episode; invalid: keys kept as 'invalid_no_signal' (no CoC).
    Returns (records, keys whose keyframe no longer matches the old one; those are not written)."""
    out = Path(out); (out / "labels").mkdir(parents=True, exist_ok=True)
    ep = data.load_episode(ep_name)
    K = {(k["t0"], k["kind"]): k for k in events.detect(ep)}
    recs, stale = [], []
    for r in old:
        k = K.get((r["t0"], r["kind"]))
        if r["key"] in invalid:
            recs.append(dict(r, version=VERSION, status="invalid_no_signal", coc=None))
        elif k is None or json.loads(json.dumps(k["components"], default=float)) != r["candidates"]:
            stale.append(r["key"])
        else:
            recs.append(record(out, ep, dict(k, key=r["key"], qa=r["qa"]), r["verification"] or {}, r["vehicles"], no_frames=r["status"] == "no_frames"))
    if stale: return recs, stale                     # nothing written: the caller decides
    with open(out / "labels" / f"{ep_name}.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return recs, stale
