"""Self-checks: python -m pytest tools/coc_labeler/tests (needs ~/.claude-morai-mac/labels-cache + manifests)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coc_labeler import compose, data, events          # noqa: E402
from coc_labeler.events import LAT, LON               # noqa: E402


def test_events_known_episode():
    # CP3 start, parked-vehicle obstacle at s 329 (right), red waits at 6254 and 4188 ~30-40 m before the lines
    K = events.detect(data.load_episode("run1021i-w1-006"))
    kinds = [k["kind"] for k in K]
    assert sum("signal_stop" in k for k in kinds) >= 2
    byp = [k for k in K if "bypass" in k["kind"]]
    assert byp and byp[0]["decision"]["lateral"] == LAT["out_nudge"] and byp[0]["decision"]["direction"] == "left"
    assert all(k["t0"] >= 30 and len(k["frames_bag_ns"]) == 17 for k in K)


def _k(kind, comps, lon, lat, d=None, **kw):
    return dict(dict(episode="e", t0=100, kind=kind, decision=dict(longitudinal=lon, lateral=lat, direction=d), components=comps,
                     src={"lon": kind, "lat": kind}, env="S13"), **kw)


SIG = dict(id="signal", category="Traffic lights", stop_line="6254", movement="straight", state="red", judged=True, distance_m=40.0)


def test_signal_needs_matching_readable_lamp():
    k = _k("signal_stop", [SIG], LON["stop"], LAT["keep"])
    assert compose.compose(k, {"signal": dict(visible="yes", lamps="red")}, None)[0] == "ok"
    assert compose.compose(k, {"signal": dict(visible="yes", lamps="unreadable")}, None)[0] == "unobserved"
    assert compose.compose(k, {"signal": dict(visible="yes", lamps="green")}, None)[0] == "unobserved"    # disagrees with GT
    far = _k("signal_stop", [dict(SIG, distance_m=70.0)], LON["stop"], LAT["keep"], env="F13")
    assert compose.compose(far, {"signal": dict(visible="yes", lamps="red")}, None)[0] == "unobserved"   # beyond fog range
    st = _k("signal_stop", [SIG], LON["stop"], LAT["keep"], standing=True)
    assert compose.compose(st, {"signal": dict(visible="yes", lamps="red")}, None)[3].startswith("Remain stopped")


def test_left_arrow_semantics():
    left = dict(SIG, stop_line="4356", movement="left")
    k = _k("signal_stop", [left], LON["stop"], LAT["keep"])
    s, _, _, t = compose.compose(k, {"signal": dict(visible="yes", lamps="green")}, None)   # through-green, no arrow
    assert s == "ok" and "left-turn arrow" in t


def test_unseen_obstacle_is_unobserved_even_with_other_cause():
    obst = dict(id="actor1", category="Critical objects", type="obstacle", side="right", motion="stationary", distance_m=30.0)
    k = _k("bypass+signal_pass", [obst, dict(SIG, state="green")], LON["set_speed"], LAT["out_nudge"], "left")
    k["src"] = {"lon": "signal_pass", "lat": "bypass"}
    ver = {"actor1": dict(visible="no", appearance=""), "signal": dict(visible="yes", lamps="green")}
    assert compose.compose(k, ver, None)[0] == "unobserved"
    ver["actor1"] = dict(visible="yes", appearance="orange drum")
    s, _, _, t = compose.compose(k, ver, None)
    assert s == "ok" and "orange drum" in t and "green light" in t


def test_work_drums_only_when_seen():
    # fog, drums 45 m ahead by the route, but the VLM saw only the water barriers on the left (v1 run1020a-w1-001 k026_02689 style)
    work = dict(id="work", category="ODD constraints", type="construction drums closing the right lane", feature_dist_m=10.0, drums_dist_m=45.0)
    k = _k("work_shift", [work], LON["adapt"], LAT["lc"], "left", env="F13")
    s, dec, comps, t = compose.compose(k, {"work": dict(visible="yes", appearance="red-white striped barriers")}, None)
    assert s == "ok" and "drums" not in t and t == "Slow down and change lanes to the left ahead of the construction zone."
    assert dec["lateral"] == LAT["lc"] and dec["longitudinal"] == LON["adapt"] and "drums" not in comps[0]["type"]
    s, _, comps, t = compose.compose(k, {"work": dict(visible="yes", appearance="orange drums, red-white barriers")}, None)
    assert s == "ok" and "construction drums close the right lane" in t and "drums" in comps[0]["type"]


def test_int551_slow_right_turn_keeps_turn_wording():
    # passes the unjudged int551 light on green and turns right slowly: only ~0.78 rad of heading change in the first 6 s
    k = next(k for k in events.detect(data.load_episode("run1021k-w1-004")) if k["t0"] == 1223)
    assert k["kind"] == "signal_pass" and k["decision"]["lateral"] == LAT["turn"] and k["decision"]["direction"] == "right"
    s, _, _, t = compose.compose(k, {"signal": dict(visible="yes", lamps="green")}, None)
    assert s == "ok" and t == "Turn right through the intersection on the green light."


def test_gerund_converts_both_verbs():
    assert compose._gerund("slow down and stop before the intersection") == "slowing down and stopping before the intersection"
