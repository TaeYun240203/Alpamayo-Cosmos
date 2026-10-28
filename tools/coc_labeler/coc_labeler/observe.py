"""VLM observation on the 2 s history window only (t = -2.0 .. 0.0 s): future frames are never shown, so a cited
cause cannot come from after the keyframe (AR1 causal locality). Two closed-form queries:
  verify(): is each privileged component visible, how does it look, what do the signal lamps show
  npc():    which other vehicles matter (only where no vehicle ground truth exists: roundabout, ramp merge, highway)"""
HIST_T = [-2.0, -1.5, -1.0, -0.5, 0.0]

INSTR = ("You verify driving-scene facts for an autonomous-driving dataset. You see only front-camera frames from the two "
         "seconds BEFORE a decision moment (t = 0). Report only what is visible in these frames; never infer from what usually "
         "happens next. If something is too small, dark, foggy or occluded to identify, say 'no' or 'uncertain'.")

VERIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "visible", "first_visible_t", "side", "appearance", "lamps"],
    "properties": {"id": {"type": "string"}, "visible": {"type": "string", "enum": ["yes", "no", "uncertain"]},
                   "first_visible_t": {"type": ["number", "null"]}, "side": {"type": "string", "enum": ["left", "right", "center", "ahead", "unknown"]},
                   "appearance": {"type": "string"}, "lamps": {"type": "string"}}}}}}

POS = ["ahead_same_lane", "ahead_left_lane", "ahead_right_lane", "alongside_left", "alongside_right", "behind_left_lane",
       "behind_right_lane", "in_roundabout_left", "in_roundabout_ahead", "in_roundabout_right", "oncoming"]
MOT = ["approaching_ego", "moving_away", "crossing_left_to_right", "crossing_right_to_left", "same_direction_similar_speed", "stopped", "unclear"]
NPC_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["vehicles"], "properties": {"vehicles": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["description", "position", "distance", "motion", "first_visible_t"],
    "properties": {"description": {"type": "string"}, "position": {"type": "string", "enum": POS},
                   "distance": {"type": "string", "enum": ["near", "mid", "far"]}, "motion": {"type": "string", "enum": MOT},
                   "first_visible_t": {"type": ["number", "null"]}}}}}}

SITUATION = {"roundabout": "The ego is approaching or waiting at the entrance of a single-lane roundabout; circulating traffic comes from the ego's left.",
             "merge": "The ego is in an on-ramp acceleration lane that ends; it has to merge into the highway lane on its left.",
             "lead": "The ego is driving on a multi-lane highway.",
             "any": "The ego is driving along its route."}


def locator(c):
    d = c.get("distance_m"); far = f"about {d:.0f} m ahead at t=0" if d is not None else "ahead"
    if c["id"] == "signal":
        return (f"the traffic signal head that controls the ego's lane at the intersection ahead (the light is {far}; the ego's route goes "
                f"{c['movement']} there). For 'lamps', name every lit lamp of that head, e.g. 'red', 'yellow', 'green', 'green + green left arrow', "
                f"'red + green left arrow', or 'unreadable'.")
    if c["id"].startswith("actor") and c["category"] == "Critical objects" and c.get("motion") == "stationary":
        return (f"an object placed on the road {far}, near the {c['side']} side of the ego's lane: a scenario obstacle such as a drum, "
                f"barricade, box, bollard, concrete barrier or parked vehicle (not moving traffic)")
    if c["id"].startswith("actor"):
        return f"a person (a pedestrian, possibly walking a bicycle) on or beside the road {far}"
    if c["id"] == "work":
        return "construction channelizing devices on the road ahead: orange drums, red/white water-filled barriers or barricades"
    if c["id"] == "toll":
        return "a toll gate: a toll plaza canopy or booths across the road ahead"
    return c.get("type", c["id"])


def verify(vlm, comps, history):
    items = [c for c in comps if c["id"] in comps_to_check(comps)]
    if not items: return {}
    text = ("The images above are front-camera frames at t = -2.0, -1.5, -1.0, -0.5 and 0.0 s (t = 0 is the decision moment).\n"
            "Check each item ONLY in these frames:\n" + "\n".join(f"- id={c['id']}: {locator(c)}" for c in items) +
            "\nFor every item return: visible (yes/no/uncertain), first_visible_t (earliest t at which you can identify it, or null), "
            "side relative to the ego's lane, appearance (2-4 words naming what you actually see, colour + object, no location, e.g. "
            "'orange-white striped drum'; empty if not visible), lamps (traffic signals only: the lit lamps in the latest frame where the "
            "head facing the ego is readable; 'unreadable' if its lamps cannot be seen, e.g. viewed from the side; else empty).")
    images = list(history)
    if any(c["id"] == "signal" for c in items) and history:     # same pixels, enlarged: small far heads tokenise poorly
        images.append(("2x enlargement of the upper-centre region (x 320-960, y 60-380) of the t=+0.0 s frame", zoom(history[-1][1])))
    out = vlm.ask(INSTR, text, images, VERIFY_SCHEMA, "verify")
    return {r["id"]: r for r in out["items"]}


def zoom(jpg):
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(jpg)); w, h = im.size
    box = (w // 4, h // 12, 3 * w // 4, int(h * 0.53))
    b = io.BytesIO(); im.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.LANCZOS).save(b, "JPEG", quality=92)
    return b.getvalue()


def comps_to_check(comps):
    return [c["id"] for c in comps if c["id"] == "signal" or c["id"].startswith("actor") or c["id"] in ("work", "toll")]


def npc(vlm, situation, history):
    text = ("The images above are front-camera frames at t = -2.0, -1.5, -1.0, -0.5 and 0.0 s (t = 0 is the decision moment). "
            + SITUATION[situation] +
            "\nList the other moving or stopped traffic vehicles visible in these frames that could influence the ego's speed or lane choice "
            "in the next seconds. Ignore parked scenario props, construction devices and vehicles on other roads or behind barriers. "
            "For each: description (colour + type, at most 5 words), position, distance at the latest frame it is visible "
            "(near < 15 m, mid 15-40 m, far > 40 m), motion, first_visible_t. Return an empty list if there are none.")
    return vlm.ask(INSTR, text, history, NPC_SCHEMA, "npc")["vehicles"]
