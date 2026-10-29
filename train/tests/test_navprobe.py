"""scripts/navprobe.py on CPU: a tiny saved student (built as tests/test_lwf.py's fixture) on the synthetic step-0 root
(ft_synth); the keep's kind filter, both variants and the paired differences in summary.json, and differences of exactly 0
when both variants carry the same text (same seed, so only the text could move the plan)."""
import json, sys
from pathlib import Path

import pandas as pd, pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft_synth import make                                               # noqa: E402
from scripts.navprobe import M, main                                    # noqa: E402
from test_lwf import NARROW, RELEASE, TINY_T, V4                        # noqa: E402

SAME = "Follow the road. Speed limit 60 km/h"                           # ft_synth's index nav_text


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from morai_vla.model import TeacherVLA
    from morai_vla.student import StudentVLA
    if RELEASE is None: pytest.skip("no release config")
    d = tmp_path_factory.mktemp("navprobe"); root = make(d / "root")
    TeacherVLA.build("random:" + RELEASE, vlm_name_or_path=V4, attn_implementation="sdpa", dtype="float32", vlm_overrides=TINY_T,
                     expert_overrides=dict(hidden_size=128, intermediate_size=256, num_attention_heads=4, head_dim=64)).save_pretrained(d / "tstar")
    StudentVLA.build(teacher=str(d / "tstar"), init="random", vlm_name_or_path=V4, teacher_vlm=V4, attn_implementation="sdpa", dtype="float32",
                     vlm_overrides=NARROW, distill=False).save_pretrained(d / "st")
    pd.DataFrame({"episode": ["r2-a-003", "r2-a-003", "old-004"], "i": [20, 40, 40], "kind": ["onset", "onset", "rest"]}).to_parquet(d / "keep.parquet")
    return d, root


def run(d, root, out, *extra):
    return main(["--ckpt", str(d / "st"), "--root", str(root), "--keep", str(d / "keep.parquet"), "--out", str(d / out),
                 "--k", "2", "--device", "cpu", *extra])


def test_variants_and_differences(setup):
    d, root = setup
    s = run(d, root, "bug", "--kind", "onset")
    assert s["n"] == 2 and s["skipped"] == 0 and s["same_text"] == 0.0 and s["v0"] > 0
    assert set(s["variants"]) == {"bug", "fixed"} and all(set(v) == {"brake_rate", *M} for v in s["variants"].values())
    assert set(s["diff"]) == set(M) and all(set(v) == {"mean", "median"} for v in s["diff"].values())
    assert s["diff"]["dist3"]["mean"] != 0                                 # the replaced text reaches the prompt
    assert 0 <= s["only_bug_brakes"] <= 1 and s == json.load(open(d / "bug/summary.json"))
    rows = pd.read_csv(d / "bug/rows.csv")
    assert set(rows.kind) == {"onset"} and (rows.nav_text == SAME).all() and (rows.bug_v0 == rows.fixed_v0).all()


def test_same_text_gives_zero_difference(setup):
    d, root = setup
    s = run(d, root, "same", "--bug", SAME)
    assert s["n"] == 3 and s["same_text"] == 1.0
    assert all(v == {"mean": 0.0, "median": 0.0} for v in s["diff"].values()), s["diff"]
    assert s["variants"]["bug"] == s["variants"]["fixed"] and s["only_bug_brakes"] == s["only_fixed_brakes"] == 0.0
