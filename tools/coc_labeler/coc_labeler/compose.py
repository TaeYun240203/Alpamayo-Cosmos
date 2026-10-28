"""Decision + verified critical components -> one CoC sentence in the release-prompt example style (AR1 Sec. 4.1:
decision grounding, causal locality, annotation economy). Each channel's decision must be explained by a clause from
the event that produced it, built only from components seen in the 2 s history; otherwise the keyframe is
'unobserved' (trains actions, gets no CoC)."""
import re
import zlib

from .events import LAT, LON

ARROW = re.compile(r"arrow")
DRUMS = re.compile(r"drum|barrel", re.I)


def _pick(key, options):
    return options[zlib.crc32(key.encode()) % len(options)]


def _appearance(c, ver):
    a = re.sub(r"^(an?|the)\s+", "", (ver or {}).get("appearance", "").strip().lower())
    return a if a and len(a.split()) <= 6 else c.get("type", "obstacle")


def signal_ok(c, ver):
    """(observed, state). Lamps must be readable in the history; with ground truth they must also agree with it."""
    if not ver or ver["visible"] == "no": return False, None
    lamps = ver.get("lamps", "").lower()
    if not lamps or "unread" in lamps: return False, None
    arr = bool(ARROW.search(lamps)); base = re.sub(r"green\s+left(-turn)?\s+arrow", "", lamps)
    red, yel, grn = "red" in base, "yellow" in base, "green" in base
    gt = c.get("state"); left = c.get("movement") == "left"
    if not c.get("judged"):
        st = "green left-turn arrow" if arr and left else "green" if grn and not red else "yellow" if yel and not red else "red" if red else None
        return st is not None, st
    if left:
        if gt == "green": return arr, "green left-turn arrow"
        return (red or (grn and not arr) or yel), ("red" if red else "yellow" if yel else "no left-turn arrow")
    if gt == "red": return red, "red"
    if gt == "yellow": return yel, "yellow"
    if gt == "green": return grn or arr, "green"
    return False, None


def _lower(s):
    return s[0].lower() + s[1:]


def _gerund(clause):
    first, _, rest = clause.partition(" ")
    G = {"stop": "stopping", "slow": "slowing", "maintain": "maintaining", "proceed": "proceeding", "continue": "continuing",
         "accelerate": "accelerating", "resume": "resuming", "decelerate": "decelerating", "turn": "turning", "keep": "keeping"}
    rest = re.sub(r"\band (\w+)", lambda m: "and " + G.get(m.group(1), m.group(1)), rest)   # 'slow down and stop' -> '... and stopping'
    return f"{G.get(first, first)} {rest}".strip()


def compose(k, ver, vehicles):
    """returns (status, decision, components, text)."""
    kinds = k["kind"].split("+"); dec = dict(k["decision"]); key = f"{k['episode']}:{k['t0']}"
    src = k.get("src") or {"lon": kinds[0], "lat": kinds[0]}
    by = {c["id"]: c for c in k["components"]}; d = dec.get("direction")
    vehicles = vehicles or []
    seen = lambda cid: (ver.get(cid) or {}).get("visible") == "yes"
    tunnel = lambda c: " in the tunnel" if c.get("tunnel") else ""
    left_veh = [v for v in vehicles if v["position"] in ("ahead_left_lane", "alongside_left", "behind_left_lane")]
    circ = [v for v in vehicles if v["position"].startswith("in_roundabout") or v["motion"].startswith("crossing")]
    lead = [v for v in vehicles if v["position"] == "ahead_same_lane" and v["distance"] in ("near", "mid")]
    vdesc = lambda v: v["description"].lower().rstrip(".")
    slow = dec["longitudinal"] in (LON["adapt"], LON["gap"])
    cl, comps = {}, []                                   # kind -> (text, channels covered, component ids)

    def veh(v, pose=None):
        comps.append(dict(id="vlm_vehicle", category="Critical objects", type=v["description"], pose=pose or v["position"],
                          motion=v["motion"], distance=v["distance"], source="vlm", visible=True))

    fog = str(k.get("env", "")).startswith("F")
    for kind in kinds:
        if kind in ("signal_stop", "signal_go", "signal_pass"):
            c = by["signal"]; ok, st = signal_ok(c, ver.get("signal"))
            # lamp colour survives to ~60 m in fog / beyond 80 m in sun at this camera (2026-10-23 study): farther reads are guesses
            if not ok or (c.get("distance_m") or 0) > (60 if fog else 90): continue
            go = kind != "signal_stop"
            if go and st in ("red", "yellow", "no left-turn arrow") and c["movement"] == "right" and not c["judged"]:
                cl[kind] = ("Turn right at the intersection to follow the route.", {"lat"}, ["junction"]); continue
            if go and st in ("red", "no left-turn arrow"): continue          # never write 'proceed on red'
            if not go and st.startswith("green"): continue
            comps.append(dict(c, state=st, visible=True)); turn = dec["lateral"] == LAT["turn"]
            if kind == "signal_stop" and k.get("standing"):
                t = ("Keep waiting for the green left-turn arrow at the intersection ahead." if st == "no left-turn arrow" else
                     f"Remain stopped for the {st} traffic light ahead.")
                cl[kind] = (t, {"lon"}, ["signal"])
            elif kind == "signal_stop":
                t = ("Stop and wait for the green left-turn arrow at the intersection ahead." if st == "no left-turn arrow" else
                     _pick(key, [f"Stop for the {st} traffic light ahead.", f"Decelerate to a stop because the traffic light ahead is {st}.",
                                 f"Slow down and stop before the intersection for the {st} traffic light."]))
                cl[kind] = (t, {"lon"}, ["signal"])
            elif turn:
                t = (f"Turn {d} at the intersection as the {'left-turn arrow' if 'arrow' in st else 'light'} turns green." if kind == "signal_go"
                     else f"Turn {d} through the intersection on the {'green left-turn arrow' if 'arrow' in st else st + ' light'}.")
                cl[kind] = (t, {"lat", "lon"}, ["signal"])
            elif kind == "signal_go" and "arrow" in st:
                cl[kind] = ("Accelerate toward the intersection as the left-turn arrow turns green.", {"lon"}, ["signal"])
            elif kind == "signal_go":
                cl[kind] = (_pick(key, ["Accelerate toward the intersection as the traffic light turns green.",
                                        "Resume speed through the intersection because the light has turned green."]), {"lon"}, ["signal"])
            else:
                cl[kind] = ("Continue through the intersection as the light turns yellow too close to stop." if st == "yellow"
                            else "Proceed through the intersection on the green light.", {"lon"}, ["signal"])
        elif kind == "stopline_stop":
            cl[kind] = ("Stop at the stop line before entering the intersection.", {"lon"}, ["stopline"]); comps.append(dict(by["stopline"], visible=True))
        elif kind == "turn":
            comps.append(dict(by["junction"], visible=True))
            cl[kind] = (f"Slow down and turn {d} at the intersection to follow the route." if slow else f"Turn {d} at the intersection to follow the route.",
                        {"lat", "lon"} if slow else {"lat"}, ["junction"])
        elif kind == "curve":
            c = by["curve"]
            # geometry is only a cause once it can be seen: not from inside the tunnel, not beyond fog range
            if c.get("behind_tunnel_exit") and c.get("feature_dist_m", 0) > 15: continue
            if c.get("feature_dist_m", 0) > (50 if fog else 100): continue
            comps.append(dict(c, visible=True))
            what = "the right-curving ramp" if c["type"] == "ramp curve" else f"the {'sharp ' if c['type'] == 'sharp curve' else ''}{c['dir']} curve"
            if c.get("visible_bend") not in (None, c["dir"], "straight"): what = "the curving road"   # a bend the other way shows first
            cl[kind] = (_pick(key, [f"Slow down for {what} ahead.", f"Reduce speed for {what} ahead."]) if c.get("slows")
                        else f"Keep lane and maintain a safe speed through {what}.", {"lon"}, ["curve"])
        elif kind.startswith("rbt_"):
            comps.append(dict(by["rbt"], visible=True))
            if circ: veh(circ[0])
            if kind == "rbt_slow":
                cl[kind] = (f"Slow down at the roundabout entrance for the {vdesc(circ[0])} circulating ahead." if circ else "Slow down for the roundabout ahead.", {"lon"}, ["rbt"])
            elif kind == "rbt_stop":
                if circ: cl[kind] = (f"Stop at the roundabout entrance to yield to the {vdesc(circ[0])} circulating from the left.", {"lon"}, ["rbt"])
                else: dec["longitudinal"] = LON["stop"]; cl[kind] = ("Stop at the yield line before entering the roundabout.", {"lon"}, ["rbt"])
            elif kind == "rbt_enter":
                cl[kind] = (f"Turn right into the roundabout after the {vdesc(circ[0])} has passed." if circ and circ[0]["motion"] in ("moving_away", "crossing_left_to_right")
                            else "Turn right into the roundabout as the circulating lane is clear.", {"lat"}, ["rbt"])
            else:
                cl[kind] = ("Turn right to exit the roundabout.", {"lat"}, ["rbt"])
        elif kind == "merge":
            comps.append(dict(by["merge"], visible=True))
            left_veh = [v for v in left_veh if v["distance"] in ("near", "mid")]
            if left_veh: veh(left_veh[0])
            if left_veh and dec["longitudinal"] == LON["gap"]:
                cl[kind] = (f"Adjust speed to merge {d} {'behind' if left_veh[0]['position'] == 'ahead_left_lane' else 'next to'} the {vdesc(left_veh[0])} in the {d} lane.", {"lat", "lon"}, ["merge"])
            else:
                if dec["longitudinal"] == LON["gap"] and src["lon"] == "merge": dec["longitudinal"] = LON["set_speed"]
                cl[kind] = (f"Merge {d} onto the highway as the acceleration lane ends" + (f", keeping clear of the {vdesc(left_veh[0])} in the {d} lane." if left_veh else "."), {"lat"}, ["merge"])
        elif kind in ("merge_wait", "merge_slow"):
            # a gap is negotiated with traffic beside the ego or closing in on the left; far vehicles ahead are not the reason
            gapv = [v for v in left_veh if v["position"] in ("alongside_left", "behind_left_lane")
                    or (v["distance"] in ("near", "mid") and v["motion"] in ("approaching_ego", "same_direction_similar_speed", "stopped"))]
            if not gapv: continue
            v0 = gapv[0]; veh(v0); comps.append(dict(by.get("merge", {"id": "merge", "category": "Road events", "type": "acceleration lane"}), visible=True))
            ahead = v0["position"] == "ahead_left_lane"
            cl[kind] = ((f"Stop at the end of the acceleration lane and wait for a gap {'behind' if ahead else 'after'} the {vdesc(v0)} in the left lane." if kind == "merge_wait"
                         else (f"Slow down to merge behind the {vdesc(v0)} in the left lane." if ahead else f"Slow down to let the {vdesc(v0)} in the left lane pass before merging.")), {"lon"}, ["merge"])
        elif kind in ("work_shift", "work_slow", "toll_slow", "toll_lane", "toll_stop"):
            fid = "toll" if kind.startswith("toll") else "work"
            if not seen(fid) or by[fid].get("feature_dist_m", 0) > {"work": 80 if fog else 150, "toll": 100 if fog else 250}[fid]: continue
            # drums are cited only when the VLM saw them: in fog it often sees just the water barriers on the left
            drums = by[fid].get("drums_dist_m", 1e9) <= (60 if fog else 120) and bool(DRUMS.search((ver.get(fid) or {}).get("appearance", "")))
            comps.append(dict(by[fid], visible=True, appearance=_appearance(by[fid], ver.get(fid)), **({} if drums or fid != "work" else {"type": "construction zone"})))
            cl[kind] = {"work_shift": (f"{'Slow down and change' if slow else 'Change'} lanes to the {d} " + ("as construction drums close the right lane ahead." if drums else "ahead of the construction zone."), {"lat", "lon"} if slow else {"lat"}),
                        "work_slow": ("Slow down for the construction zone ahead.", {"lon"}),
                        "toll_slow": ("Slow down for the toll gate ahead.", {"lon"}),
                        "toll_stop": ("Stop at the toll gate ahead.", {"lon"}),
                        "toll_lane": (f"{'Slow down and change' if slow else 'Change'} lanes to the {d} to line up for the toll gate ahead.", {"lat", "lon"} if slow else {"lat"})}[kind] + ([fid],)
        elif kind in ("bypass", "obstacle_stop", "ped_yield"):
            c = next(x for x in k["components"] if x["id"].startswith("actor"))
            if not seen(c["id"]): continue
            a = _appearance(c, ver.get(c["id"])); comps.append(dict(c, visible=True, appearance=a))
            where = f"on the {c['side']}" if c.get("side") in ("left", "right") and c["side"] != d else "blocking the lane"
            if kind == "bypass":
                pre = "Slow down and nudge" if dec["longitudinal"] == LON["adapt"] and src["lon"] == "bypass" else "Nudge"
                t = (f"{pre} {d} out of the lane to pass the {a} {where}{tunnel(c)}." if dec["lateral"] == LAT["out_nudge"]
                     else f"{pre} {d} within the lane to keep clearance from the {a} {where}{tunnel(c)}.")
                cl[kind] = (t, {"lat", "lon"} if pre.startswith("Slow") else {"lat"}, [c["id"]])
            elif kind == "obstacle_stop":
                cl[kind] = (f"Stop behind the {a} blocking the lane ahead{tunnel(c)}.", {"lon"}, [c["id"]])
            else:
                cl[kind] = (f"Stop to yield to the {a} crossing the road ahead{tunnel(c)}." if k.get("stops") else f"Slow down to yield to the {a} ahead{tunnel(c)}.", {"lon"}, [c["id"]])
        elif kind in ("hw_slow", "stop_unexplained"):
            if not lead: continue
            veh(lead[0], "ahead_same_lane"); dec["longitudinal"] = LON["lead"]
            cl[kind] = (f"Stop behind the {vdesc(lead[0])} ahead." if kind == "stop_unexplained"
                        else f"Slow down to keep a safe distance from the {vdesc(lead[0])} ahead.", {"lon"}, ["vlm_vehicle"])

    lon_need = dec["longitudinal"] not in (None, LON["set_speed"]); lat_need = dec["lateral"] not in (None, LAT["keep"])
    lon_c = cl.get(src["lon"]) if lon_need else None
    lat_c = cl.get(src["lat"]) if lat_need else None
    if lon_need and not (lon_c and "lon" in lon_c[1]) and not (lat_c and "lon" in lat_c[1]): return "unobserved", dec, comps, None
    if lat_need and not (lat_c and "lat" in lat_c[1]): return "unobserved", dec, comps, None
    if not lon_need and not lat_need:                     # steady driving with a visible reason (green light, curve)
        if not cl: return "unobserved", dec, comps, None
        return "ok", dec, comps, next(iter(cl.values()))[0]
    if lat_c and (not lon_c or lon_c is lat_c or "lon" in lat_c[1]):
        text = lat_c[0]
        extra = [v for kd, v in cl.items() if v is not lat_c and kd in ("signal_go", "signal_pass") and "lon" in v[1]]
        if extra and not lon_need: text = text.rstrip(".") + " while " + _gerund(_lower(extra[0][0]).rstrip(".")) + "."
    elif lon_c and not lat_c:
        text = lon_c[0]
    else:
        lt, ln = lat_c[0].rstrip("."), lon_c[0].rstrip(".")
        if ln.startswith("Stop"): text = f"{lt}, then {_lower(ln)}."
        elif set(lat_c[2]) & set(lon_c[2]) and ln.startswith(("Slow", "Reduce")): text = "Slow down and " + _lower(lt) + "."
        else: text = f"{lt} while {_gerund(_lower(ln))}."
    return "ok", dec, comps, text
