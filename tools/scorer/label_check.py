"""DAgger label checker: one decision's 6.4 s label (the shadow collection driver's rollout, dagger_label.py) judged by
the competition rules as scorer.py judges them, against the best the t0 state allowed.

    from label_check import LabelChecker
    chk = LabelChecker()
    r = chk.check(dict(x=, y=, yaw=, v=), rollout, signal=chk.signal_at(tlgt_rows, t0, ego), objects=rec.objects(tw))
    r["verdict"], r["valid"], r["reasons"], r["tags"], r["label"], r["oracle"], r["better"]

Verdicts
  pass              no violation
  pass_unavoidable  violations, but no reference from the same t0 state does clearly better on them without costing
                    more overall (unavoidable, within tolerance, or the cheaper side of a rule conflict)
  fail              a reference does, or the red-light rules below say so: the row leaves training (valid False)
  unrecoverable     no way back from t0 (route missed, wrong way, no way back to the road or the checkpoint): the row
                    leaves training (valid False), counted per reason (user 10/28)
  unjudgeable       t0 cannot be judged (unknown light at a crossing): kept
Red light (user 10/28): a car that cannot stop before its front axle reaches the judged line even at the collection
driver's braking limit goes through (stopping in the junction fails, red_stop_in_junction); one whose front axle is
already past goes once the way is clear (objects at constant velocity) and fails standing on a clear way
(stay_after_cross). The scorer's crossing is the front axle centre over the line's polyline (its painted centreline).
Collisions: an object running into a standing ego or from behind its rear axle is not the label's, except one in the
lane the label is changing into once the body is over the boundary, or any vehicle from behind inside the highway
merge (user 10/28: lane_change_rear, whole horizon, every GT vehicle). A car that was farther back at t0 than the
shadow's rear range (frenet_gap.patch REAR_RANGE_M) could not be seen: pass_unavoidable, lane_change_rear_hidden, and
debug_fail True so shadow debugging still counts it (valid untouched).
Moving objects the model could not see at t0 (visible(), shared with the shadow labeller), other than vehicles behind
it, are left out of every judgment; a label running into one is pass_unavoidable (user 10/28: collision:hidden_at_t0).
Only t0 information: the light state at t0 (a crossing more than yellow_max_s after a t0 yellow is unknown) and
objects moved at constant velocity. References: the label's own path
shifted sideways or braked to a stop, and the collection driver's pure pursuit back to the route lane, within
label_check.yaml's weak-side vehicle limits. The judgments are scorer.py's: KMap lane paint, drivable area, route and
checkpoints, _cross_params stop line crossings, _signal_ahead light states, thresholds.yaml constants.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scorer import CELL_M, HERE, KMap, _cross_params, _runs, _signal_ahead, load_config

N, DT = 64, 0.1
VERDICTS = ("pass", "pass_unavoidable", "fail", "unjudgeable", "unrecoverable")
UNRECOVERABLE = ("start:off_route", "start:wrong_way", "inoperable:unrecoverable", "checkpoint:missed")
RULES = ("signal", "collision", "speed", "lane", "inoperable", "checkpoint")
EPS = dict(signal=0.5, collision=0.5, checkpoint=0.5, speed=0.05, lane=0.05, inoperable=0.05)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _seg_dist(p, a, b):
    """(P, 2) points to (S, 2) segments a-b -> (P, S) distances (scorer._point_seg_dist for many points)."""
    ab, ap = b - a, p[:, None] - a[None]
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.hypot(*(ap - t[..., None] * ab).transpose(2, 0, 1))


def _near(pts, a, b, reach, grid):
    """scorer._near for many points: each grid cell's segments are tested once against all its points."""
    hit = np.zeros(len(pts), bool)
    ok = np.flatnonzero(np.isfinite(pts).all(1))
    if not len(ok):
        return hit
    cells, inv = np.unique(np.floor(pts[ok] / CELL_M).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    for u, (gx, gy) in enumerate(cells):
        seg = grid.get((int(gx), int(gy)))
        if seg is not None:
            k = ok[inv == u]
            hit[k] = (_seg_dist(pts[k], a[seg], b[seg]) <= reach[seg]).any(1)
    return hit


def _overlap(ca, ha, ua, cb, hb, ub):
    """Rectangles (centre, half length/width, unit heading; broadcast) overlap, separating axis test."""
    va, vb = np.stack([-ua[..., 1], ua[..., 0]], -1), np.stack([-ub[..., 1], ub[..., 0]], -1)
    d, sep = cb - ca, False

    def dot(p, q):
        return np.abs((p * q).sum(-1))
    for ax in (ua, va, ub, vb):
        r = ha[..., 0] * dot(ua, ax) + ha[..., 1] * dot(va, ax) + hb[..., 0] * dot(ub, ax) + hb[..., 1] * dot(vb, ax)
        sep = sep | (dot(d, ax) > r)
    return ~sep


def _segment_boxes(a, b, c, u, h):
    """Segments a[p]-b[p] (P, 2) through rectangles (centre c, unit heading u, half sizes h; J): (P, J), slab test."""
    v = np.stack([-u[:, 1], u[:, 0]], -1)

    def local(p):
        d = p[:, None] - c[None]
        return np.stack([(d * u[None]).sum(-1), (d * v[None]).sum(-1)], -1)
    qa = local(a)
    dq = local(b) - qa
    lo, hi = np.zeros(qa.shape[:2]), np.ones(qa.shape[:2])
    for k in range(2):
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (-h[None, :, k] - qa[..., k]) / dq[..., k], (h[None, :, k] - qa[..., k]) / dq[..., k]
        flat, inside = dq[..., k] == 0, np.abs(qa[..., k]) <= h[None, :, k]
        lo = np.maximum(lo, np.where(flat, np.where(inside, 0.0, np.inf), np.minimum(ta, tb)))
        hi = np.minimum(hi, np.where(flat, np.where(inside, 1.0, -np.inf), np.maximum(ta, tb)))
    return lo <= hi


def visible(x0, y0, yaw0, objects, cfg):
    """Which t0 objects [(id, label, x, y, yaw, vx, vy, size x, size y)] the model could see from the ego pose (base_link):
    the centre or a corner inside the LiDAR BEV with a clear line from the LiDAR past every other vehicle, or other object
    at least occluder_m long. cfg = label_check.yaml "visibility". Shared with the shadow labeller's yield/AEB.
    shortcut: 2D, no heights or LiDAR points (low props do not hide; anything tall enough is a vehicle or >= 2 m long);
    count LiDAR returns instead once the shadow records them."""
    if not len(objects):
        return np.zeros(0, bool)
    behind, ahead, side = cfg["bev_m"]
    c = np.array([o[2:4] for o in objects], float)
    yaw = np.array([o[4] for o in objects], float)
    u, h = np.stack([np.cos(yaw), np.sin(yaw)], -1), np.array([o[7:9] for o in objects], float) / 2
    v = np.stack([-u[:, 1], u[:, 0]], -1)
    k = np.array([[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]], float)   # centre and corners
    pts = c[:, None] + k[None, :, :1] * h[:, None, :1] * u[:, None] + k[None, :, 1:] * h[:, None, 1:] * v[:, None]
    ce, se = math.cos(yaw0), math.sin(yaw0)
    fx = ce * (pts[..., 0] - x0) + se * (pts[..., 1] - y0)
    fy = -se * (pts[..., 0] - x0) + ce * (pts[..., 1] - y0)
    seen = (fx >= -behind) & (fx <= ahead) & (np.abs(fy) <= side)
    occ = np.flatnonzero([str(o[1]) == "vehicle" or max(o[7], o[8]) >= cfg["occluder_m"] for o in objects])
    if len(occ):
        sensor = np.array([x0 + ce * cfg["lidar_x_m"], y0 + se * cfg["lidar_x_m"]])
        cut = _segment_boxes(np.repeat(sensor[None], pts.size // 2, 0), pts.reshape(-1, 2), c[occ], u[occ], h[occ])
        cut = cut.reshape(len(objects), 5, len(occ)) & (np.arange(len(objects))[:, None, None] != occ[None, None])
        seen &= ~cut.any(2)
    return seen.any(1)


class LabelChecker:
    def __init__(self, cfg=None, kmap=None, check_cfg=HERE / "label_check.yaml"):
        self.cfg = cfg or load_config()
        self.lc = check_cfg if isinstance(check_cfg, dict) else yaml.safe_load(Path(check_cfg).read_text())
        self.km = km = kmap or KMap.from_files(self.cfg)
        self.lap = float(km.s[-1]) if np.hypot(*(km.path[0] - km.path[-1])) < 1.0 else None   # the route is a loop
        self.stops = {sid: km.route_stops[sid] for sid in self.cfg["signal"]["stop_lines"] if sid in km.route_stops}
        veh = self.cfg["vehicle"]
        self.wheels = np.array(list(veh["wheels_base_link_m"].values()), float)
        self.front = float(veh["front_axle_m"])
        self.path_reach = np.full(len(km.path) - 1, float(self.cfg["inoperable"]["route_corridor_m"]))

    # ------------------------------------------------------------------ t0 inputs
    def _s_hint(self, ego):
        s = ego.get("s")
        if s is None or not np.isfinite(s):   # KMap.progress's first fix, with the wrong way penalised instead of barred
            km = self.km
            d = np.hypot(*(km.path - (ego["x"], ego["y"])).T)
            d[:-1] += 20.0 * (np.cos(km.heading - ego["yaw"]) < 0)
            s = km.s[int(np.argmin(d))]
        return float(s)

    def signal_at(self, lights, t0, ego):
        """t0 GT state (red/yellow/green/unknown) for the next judged stop line, as scorer.score reads it. lights:
        [(t, light index, MORAI status)] on t0's clock; shadow_rec "tlgt" rows [kind, t, index, status] work as they are."""
        sig = pd.DataFrame([tuple(r[-3:]) for r in lights], columns=["t", "light", "status"])
        cfg = self.cfg["signal"]
        stops = [(st, cfg["stop_lines"][sid]) for sid, st in self.stops.items()]
        front_s = self._s_hint(ego) + self.front
        return str(_signal_ahead(np.array([t0], float), np.array([front_s]), sig, stops, cfg["max_age_s"])[0])

    def _frenet(self, pts, s_hint, reach):
        """Route (s, d left +, route heading) of pts (P, 2). Each point projects onto the route within
        [s_hint - 15, s_hint + its reach + 15]: keeps it off other passes of the route and the loop's other end."""
        km = self.km
        hi_s = s_hint + float(np.max(reach)) + 15.0
        lo = max(int(np.searchsorted(km.s, s_hint - 15.0)) - 1, 0)
        hi = min(int(np.searchsorted(km.s, hi_s)) + 1, len(km.s) - 1)
        a, b, sa = km.path[lo:hi], km.path[lo + 1:hi + 1], km.s[lo:hi]
        if self.lap and hi_s > self.lap:
            k = min(int(np.searchsorted(km.s, hi_s - self.lap)) + 1, len(km.s) - 1)
            a, b, sa = np.vstack([a, km.path[:k]]), np.vstack([b, km.path[1:k + 1]]), np.r_[sa, km.s[:k] + self.lap]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-12)
        ap = pts[:, None] - a[None]
        u = np.clip((ap * ab).sum(-1) / L2, 0, 1)
        dist = np.hypot(*(ap - u[..., None] * ab).transpose(2, 0, 1))
        dist[sa[None] > s_hint + reach[:, None] + 15.0] = np.inf
        j, r = dist.argmin(1), np.arange(len(pts))
        cross = ab[j, 0] * ap[r, j, 1] - ab[j, 1] * ap[r, j, 0]
        return sa[j] + u[r, j] * np.sqrt(L2[j]), np.where(cross < 0, -1, 1) * dist[r, j], np.arctan2(ab[j, 1], ab[j, 0])

    def _mod(self, s):
        return s % self.lap if self.lap else s

    def _ahead(self, s_from, s_to):
        """route distance from s_from forward to s_to (over the loop end)."""
        return (s_to - s_from) % self.lap if self.lap else s_to - s_from

    # ------------------------------------------------------------------ trajectories
    @staticmethod
    def _label(x0, y0, h0, v0, traj, ak):
        """-> X, Y, H, V (N + 1) from t0: traj (64, 3 or 4) map x, y, yaw[, v] at 0.1-6.4 s, or ak (64, 2) a m/s^2 and
        kappa 1/m (the action space's unicycle)."""
        if traj is not None:
            tr = np.asarray(traj, float)
            X, Y, H = np.r_[x0, tr[:, 0]], np.r_[y0, tr[:, 1]], np.unwrap(np.r_[h0, tr[:, 2]])
            V = np.r_[v0, tr[:, 3]] if tr.shape[1] > 3 else np.r_[v0, np.hypot(np.diff(X), np.diff(Y)) / DT]
            return X, Y, H, V
        X, Y, H, V = [x0], [y0], [h0], [v0]
        for a, k in np.asarray(ak, float):
            v1 = max(V[-1] + a * DT, 0.0)
            ds = 0.5 * (V[-1] + v1) * DT
            hm = H[-1] + 0.5 * k * ds
            X.append(X[-1] + ds * math.cos(hm)), Y.append(Y[-1] + ds * math.sin(hm)), H.append(H[-1] + k * ds), V.append(v1)
        return tuple(np.array(z) for z in (X, Y, H, V))

    def _references(self, L, s0, steer0):
        """[(name, X, Y, H, V)]: the label's path shifted sideways (its own timing or braking), braked to a stop on its
        own path, and the collection pp back to the route centreline (label speeds, braking, or pulling away gently)."""
        o = self.lc["oracle"]
        X, Y, H, V = L
        t = np.arange(N + 1) * DT
        vb = np.where(t < o["brake_delay_s"], V[0], np.maximum(V[0] - o["brake_mps2"] * (t - o["brake_delay_s"]), 0.0))
        sig_b = np.r_[0, np.cumsum(0.5 * (vb[1:] + vb[:-1]) * DT)]
        step = np.hypot(np.diff(X), np.diff(Y))
        sig_l = np.r_[0, np.cumsum(step)]
        keep = np.r_[True, step > 1e-3]
        ps, px, py, ph = sig_l[keep], X[keep], Y[keep], H[keep]
        e = np.arange(0.5, max(sig_b[-1], sig_l[-1]) + 20.0 - ps[-1], 0.5)    # the path goes on straight past its end
        ps, ph = np.r_[ps, ps[-1] + e], np.r_[ph, np.full(len(e), ph[-1])]
        px, py = np.r_[px, px[-1] + e * math.cos(ph[-1])], np.r_[py, py[-1] + e * math.sin(ph[-1])]

        def at(sx, sy, sh, sig, v):
            return np.interp(sig, ps, sx), np.interp(sig, ps, sy), np.interp(sig, ps, sh), v

        refs = [("brake",) + at(px, py, ph, sig_b, vb)]
        g = np.arange(0.0, ps[-1], 0.5)
        hg = np.interp(g, ps, ph)
        k_used = float(np.abs(np.diff(hg)).max() / 0.5) if len(g) > 1 else 0.0
        vmax = max(float(V.max()), 0.1)
        W = o["wheelbase_m"]
        k_cap = min(math.tan(o["steer_max_rad"]) / W, o["lat_accel_mps2"] / vmax ** 2)
        avail = k_cap - k_used
        for d in o["shifts_m"]:
            if avail <= 1e-3:
                break
            D = o["shift_slack"] * max(math.pi * math.sqrt(abs(d) / (2 * avail)),
                                       (W * vmax * abs(d) * math.pi ** 3 / (2 * o["steer_rate_radps"])) ** (1 / 3))
            u = np.minimum(ps / D, 1.0)
            off, slope = d * 0.5 * (1 - np.cos(np.pi * u)), d * np.pi / (2 * D) * np.sin(np.pi * u)
            sx, sy, sh = px - np.sin(ph) * off, py + np.cos(ph) * off, ph + np.arctan(slope)
            refs.append((f"shift{d:+.1f}",) + at(sx, sy, sh, sig_l, V))
            refs.append((f"shift{d:+.1f}_brake",) + at(sx, sy, sh, sig_b, vb))
        if steer0 is None:
            i = int(np.searchsorted(sig_l, 1.0))
            steer0 = math.atan(W * (H[i] - H[0]) / sig_l[i]) if i <= N and sig_l[i] > 0.5 else 0.0
        vg = np.minimum(V[0] + o["go_accel_mps2"] * t, max(V[0], o["go_speed_mps"]))
        for name, vt in (("route", V), ("route_brake", vb), ("route_go", vg)):
            refs.append((name,) + self._pursuit(X[0], Y[0], H[0], steer0, vt, s0))
        return refs

    def _pursuit(self, x, y, yaw, steer, vt, s0):
        """control/cores/pp.hpp on the route centreline from the t0 pose at the speeds vt (N + 1, 0.1 s), with its
        steering angle and rate limits and a lateral acceleration cap; a lookahead point behind turns at full lock."""
        o, km = self.lc["oracle"], self.km
        gain, lmin, lmax = o["lookahead"]
        W, smax, rate = o["wheelbase_m"], o["steer_max_rad"], o["steer_rate_radps"]
        reach = s0 + float(vt.sum()) * DT + lmax + 20.0
        lo = max(int(np.searchsorted(km.s, s0 - 5.0)) - 1, 0)
        P, S = km.path[lo:], km.s[lo:]
        if self.lap and reach > self.lap:
            k = int(np.searchsorted(km.s, reach - self.lap)) + 1
            P, S = np.vstack([P, km.path[1:k]]), np.r_[S, km.s[1:k] + self.lap]
        near = int(np.argmin(np.hypot(*(P[:60] - (x, y)).T)))
        steer = float(np.clip(steer, -smax, smax))
        dt, out = DT / 2, [(x, y, yaw, vt[0])]
        for k in range(2 * N):
            v = 0.5 * (np.interp(k * dt, np.arange(N + 1) * DT, vt) + np.interp((k + 1) * dt, np.arange(N + 1) * DT, vt))
            near += int(np.argmin(np.hypot(*(P[near:near + 60] - (x, y)).T)))
            j = min(int(np.searchsorted(S, S[near] + min(max(gain * v, lmin), lmax))), len(S) - 1)
            dx, dy = P[j, 0] - x, P[j, 1] - y
            tx, ty = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
            want = math.atan2(2 * W * ty, tx * tx + ty * ty) if tx > 0 else math.copysign(smax, ty)
            cap = min(smax, math.atan(W * o["lat_accel_mps2"] / v ** 2)) if v > 0.1 else smax
            steer = min(max(min(max(want, -cap), cap), steer - rate * dt), steer + rate * dt)
            yr = v * math.tan(steer) / W
            hm = yaw + 0.5 * yr * dt
            x, y, yaw = x + v * math.cos(hm) * dt, y + v * math.sin(hm) * dt, yaw + yr * dt
            if k % 2:
                out.append((x, y, yaw, vt[(k + 1) // 2]))
        return tuple(np.array(z) for z in zip(*out))

    # ------------------------------------------------------------------ judging
    def _measure(self, X, Y, H, V, ctx, strict):
        """(C, N + 1) trajectories -> violation amounts per rule (C,) and what the decision needs. strict: a reference
        (margins against it), else the label (judged as the scorer does)."""
        km, m, lc = self.km, self.lc["margin"], self.lc
        C = len(X)
        cum = np.c_[np.zeros(C), np.cumsum(np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1)), axis=1)]
        s, d, _ = self._frenet(np.c_[X.ravel(), Y.ravel()], ctx["s0"], cum.ravel())
        s, d = s.reshape(C, -1), d.reshape(C, -1)
        sm = self._mod(s)
        free = ~(ctx["blackout_all"] | np.any([(sm >= a) & (sm <= b) for a, b in lc["blackout_s"]], axis=0))
        c, sn = np.cos(H), np.sin(H)
        wheels = [np.c_[(X + c * wx - sn * wy).ravel(), (Y + sn * wx + c * wy).ravel()] for wx, wy in self.wheels]
        out = {}

        # lane: a wheel on solid paint outside the blackout
        reach = km.seg_hw + self.cfg["vehicle"]["tire_half_width_m"] + (m["ref_lane_grow_m"] if strict else 0.0)
        paint = np.any([_near(w, km.seg_a, km.seg_b, reach, km.grid) for w in wheels], axis=0).reshape(C, -1) & free
        out["lane"], out["contact"] = paint[:, 1:].sum(1) * DT, paint

        # inoperable: all four wheels off the lanelets and the route corridor outside the blackout
        off = free.ravel().copy()
        for w in wheels:
            i = np.flatnonzero(off)
            off[i[_near(w[i], km.path[:-1], km.path[1:], self.path_reach, km.path_grid)]] = False
            i = np.flatnonzero(off)
            off[i[km.on_road(w[i])]] = False
        off = off.reshape(C, -1)
        out["inoperable"], out["off0"] = off[:, 1:].sum(1) * DT, off[:, 0]

        # speed: Ego Vehicle Status speed over the limit outside the exempt route span
        lo, hi = km.speed_exempt_s
        over = (np.abs(V) * 3.6 > self.cfg["speed"]["limit_kph"] + (0.0 if strict else m["label_speed_kph"])) \
            & ~((sm >= lo) & (sm <= hi))
        out["speed"], out["speed_onsets"] = over[:, 1:].sum(1) * DT, (over[:, 1:] & ~over[:, :-1]).sum(1)

        # signal: front axle crossing a judged stop line; the t0 state holds for the line ahead (scorer _signal_ahead)
        fa = self.front + (m["ref_stop_line_m"] if strict else 0.0)
        F = np.stack([X + c * fa, Y + sn * fa], -1)
        red, unknown, t_cross = np.zeros(C), np.zeros(C), np.full(C, np.nan)
        for sid, s_stop in self.stops.items():
            if self._ahead(ctx["front_s0"] + 5.0, s_stop + 5.0) > cum.max() + 15.0:
                continue
            line = km.stop_lines[sid]
            for k in range(C):
                u = np.full(N, np.nan)
                for a, b in zip(line[:-1], line[1:]):
                    u = np.fmin(u, _cross_params(F[k, :-1], F[k, 1:], a, b))
                hit = np.flatnonzero(~np.isnan(u))
                if not len(hit):
                    continue
                tc = (hit[0] + u[hit[0]]) * DT
                state = ctx["signal"] if sid == ctx["target"] else "unknown"
                if state == "yellow" and tc > lc["signal"]["yellow_max_s"]:
                    state = "unknown"   # K-City shows yellow before green too (R1: 107 of 141 yellows): red or green next
                if sid == ctx["target"]:
                    t_cross[k] = tc
                red[k] += state == "red"
                unknown[k] += state == "unknown"
        out.update(signal=red, signal_unknown=unknown, t_cross=t_cross)

        # collision: ego box vs object boxes moved at constant velocity; overlaps at t0 are ignored
        out["collision"], out["struck"] = np.zeros(C), np.zeros(C)
        out["blocked"], out["hit_cv"] = np.zeros(C, bool), np.zeros(C, bool)
        out["lane_change_rear"], out["lane_change_rear_hidden"] = np.zeros(C, bool), np.zeros(C, bool)
        ob = ctx["objects"]
        if ob is not None:
            r, f, hw = lc["body_m"]
            ec = np.stack([X + c * (r + f) / 2, Y + sn * (r + f) / 2], -1)[:, :, None]
            eu = np.stack([c, sn], -1)[:, :, None]
            eh = np.array([(f - r) / 2, hw])
            grow = np.where(ob["standing"], m["ref_static_grow_m"], m["ref_object_grow_m"]) if strict \
                else np.full(len(ob["c"]), -m["label_object_shrink_m"])
            oh = np.maximum(ob["half"] + grow[:, None], 0.05)
            oc = ob["c"][None] + np.arange(N + 1)[:, None, None] * DT * ob["v"][None]
            trust = np.where(~ob["standing"] & (ob["label"] == "vehicle"), lc["objects"]["vehicle_trust_s"], np.inf)
            hit = _overlap(ec, eh, eu, oc[None], oh[None, None], ob["u"][None, None]) & ~ob["ignore"]
            hit_all = hit[:, 1:]
            out["hit_cv"] = hit_all.any((1, 2))   # every object at constant velocity, whole horizon (way clear?)
            hit = (hit & (np.arange(N + 1)[:, None] * DT <= trust + 1e-9))[:, 1:]
            first, any_ = hit.argmax(1) + 1, hit.any(1)
            if strict:   # references: every contact, no trust cut (a phantom only makes a reference look worse)
                out["collision"] = hit_all.any(1).sum(1).astype(float)
                ahead = ob["standing"] & ~ob["ignore"] & (ob["label"] != "vehicle")
                rel = ob["c"][None] - np.c_[X[:, -1], Y[:, -1]][:, None]
                fx = c[:, -1:] * rel[..., 0] + sn[:, -1:] * rel[..., 1]
                fy = -sn[:, -1:] * rel[..., 0] + c[:, -1:] * rel[..., 1]
                out["blocked"] = (V[:, -1] < lc["tolerance"]["stopped_mps"]) & (
                    ahead & (fx > 0) & (fx < f + 15.0) & (np.abs(fy) < 2.5)).any(1)
            else:
                k = np.arange(C)[:, None]
                vx = V[k, first]
                rel = oc[first, np.arange(len(ob["c"]))[None]] - np.stack([X[k, first], Y[k, first]], -1)
                fx = c[k, first] * rel[..., 0] + sn[k, first] * rel[..., 1]
                mine = any_ & (vx >= lc["tolerance"]["stopped_mps"]) & (fx > 0)   # moving, object ahead of the rear axle
                lcr = self._lane_change_rear(X, Y, c, sn, s, hit_all, oc, ob["v"], ob["label"]) & ~mine
                # behind at t0 beyond the shadow's rear range (frenet_gap.patch gapOk): neither it nor the model saw it
                behind = -(c[:, :1] * (ob["c"][None, :, 0] - X[:, :1]) + sn[:, :1] * (ob["c"][None, :, 1] - Y[:, :1]))
                unseen = lcr & (behind > lc["visibility"]["rear_roi_m"] + ob["half"][None, :, 0])
                out["lane_change_rear"], out["lane_change_rear_hidden"] = (lcr & ~unseen).any(1), unseen.any(1)
                mine = mine | (lcr & ~unseen)
                any_ = any_ | lcr
                out["collision"], out["struck"] = mine.sum(1).astype(float), (any_ & ~mine).sum(1).astype(float)

        # checkpoints the label passes: passed farther than the radius is a skip; so is ending short of one still
        # farther off the route than the radius (stopping short only defers the skip)
        rad = self.cfg["checkpoints"]["radius_m"] - (m["ref_checkpoint_m"] if strict else 0.0)
        if "cps" not in ctx:   # set by the label's measure, the first
            ctx["cps"] = [(cp, ctx["s0"] + a) for cp, a in ((cp, self._ahead(ctx["s0"], cs))
                          for cp, cs in zip(km.checkpoints, km.cp_s)) if 0.5 < a <= s[0, -1] - ctx["s0"]]
        skip = np.zeros(C)
        for cp, cs in ctx["cps"]:
            for k in range(C):
                pts = np.c_[X[k], Y[k]]
                skip[k] += (_seg_dist(cp[None], pts[:-1], pts[1:]).min() if s[k, -1] >= cs else abs(d[k, -1])) > rad
        out["checkpoint"] = skip

        out["progress"], out["off_route_end"] = s[:, -1], np.abs(d[:, -1]) > lc["route_lane_m"]
        w = lc["cost"]
        out["cost"] = (w["signal"] * red + w["collision"] * out["collision"] + w["speed_onset"] * out["speed_onsets"]
                       + w["speed_per_s"] * out["speed"] + w["lane_per_s"] * out["lane"]
                       + w["inoperable"] * ((out["inoperable"] > 0) & ~out["off0"]) + w["inoperable_per_s"] * out["inoperable"]
                       + w["checkpoint"] * skip + w["off_route_end"] * out["off_route_end"])
        return out

    def _lane_change_rear(self, X, Y, c, sn, s, hit, oc, ov, labels):
        """(C, O): a moving object reaching the label from behind its rear axle when at t0 the ego (body centre) was half
        a lane or more off that object's constant-velocity path, so the label moved into its lane (a car following in
        the ego's own lane is not, whatever the route does there); or any vehicle so inside the highway merge.
        hit (C, N, O) contacts over the whole horizon, s (C, N + 1) route progress."""
        r, f, _ = self.lc["body_m"]
        out = np.zeros(hit.shape[::2], bool)
        for k in range(len(X)):
            ex, ey = X[k, 0] + c[k, 0] * (r + f) / 2, Y[k, 0] + sn[k, 0] * (r + f) / 2
            for o in np.flatnonzero(hit[k].any(0)):
                j = int(np.argmax(hit[k, :, o])) + 1
                rx, ry = oc[j, o] - (X[k, j], Y[k, j])
                speed = math.hypot(*ov[o])
                if c[k, j] * rx + sn[k, j] * ry > 0 or speed < 0.1:
                    continue   # ahead of the rear axle (the ordinary rule), or standing
                if labels[o] == "vehicle" and any(a <= self._mod(s[k, j]) <= b for a, b in self.lc["merge_s"]):
                    out[k, o] = True
                else:
                    ux, uy = ov[o] / speed
                    out[k, o] = abs(ux * (ey - oc[0, o, 1]) - uy * (ex - oc[0, o, 0])) >= self.lc["lane_width_m"] / 2
        return out

    def _objects(self, ob, x0, y0, h0):
        """t0 object tuples -> the arrays _measure reads (None without objects); overlaps at t0 are marked ignore."""
        if not ob:
            return None
        vel = np.array([o[5:7] for o in ob], float)
        standing = np.hypot(*vel.T) < self.lc["objects"]["static_mps"]
        yaw = np.array([o[4] for o in ob], float)
        out = dict(c=np.array([o[2:4] for o in ob], float), v=np.where(standing[:, None], 0.0, vel),
                   u=np.stack([np.cos(yaw), np.sin(yaw)], -1), half=np.array([o[7:9] for o in ob], float) / 2,
                   label=np.array([str(o[1]) for o in ob]), standing=standing)
        r, f, hw = self.lc["body_m"]
        out["ignore"] = _overlap(np.array([x0 + math.cos(h0) * (r + f) / 2, y0 + math.sin(h0) * (r + f) / 2]),
                                 np.array([(f - r) / 2, hw]), np.array([math.cos(h0), math.sin(h0)]), out["c"],
                                 np.maximum(out["half"] - self.lc["margin"]["label_object_shrink_m"], 0.05), out["u"])
        return out

    def check(self, ego, traj=None, ak=None, signal="unknown", objects=(), blackout=False, steer=None, debug=False):
        """ego: x, y, yaw, v (GT, base_link = rear axle, map frame) and optionally s (route s hint); traj (64, 3|4) map
        x, y, yaw[, v] at 0.1-6.4 s or ak (64, 2); signal: t0 state for the next judged stop line (signal_at);
        objects: t0 GT [(id, label, x, y, yaw, vx, vy, size x, size y)] (dagger_label Rec.objects); blackout: GPS
        blackout at t0; steer: t0 steering angle (rad) if known. -> dict (module docstring)."""
        res = dict(verdict="unjudgeable", valid=True, debug_fail=False, reasons=[], tags=[], better={}, oracle=dict.fromkeys(RULES),
                   label=dict.fromkeys(RULES + ("signal_state", "t_cross", "struck", "lane_penalty_s", "cost", "paint_t0",
                                                "off_road_t0", "d_t0", "hidden_hits")))   # None: not judged (early unjudgeable)
        x0, y0, h0, v0 = (float(ego[k]) for k in ("x", "y", "yaw", "v"))
        L = self._label(x0, y0, h0, v0, traj, ak)
        if not all(np.isfinite(z).all() for z in L):
            res["reasons"].append("input:nonfinite")
            return res
        s_hint = self._s_hint(ego)
        if not np.isfinite(s_hint):
            res["reasons"].append("start:off_map")
            return res
        s0, d0, hr = (float(z[0]) for z in self._frenet(np.array([[x0, y0]]), s_hint, np.zeros(1)))
        st = self.lc["start"]
        if abs(d0) > st["off_route_m"] or abs(_wrap(h0 - hr)) > st["wrong_way_rad"]:
            res["reasons"].append("start:off_route" if abs(d0) > st["off_route_m"] else "start:wrong_way")
            res["verdict"], res["valid"] = "unrecoverable", False
            return res

        # the judged stop line ahead (its zone runs to 5 m past it, as in scorer._signal_ahead)
        front_s0 = s0 + self.front
        target = min((k for k in self.stops if self._ahead(front_s0, self.stops[k] + 5.0) >= 0),
                     key=lambda k: self._ahead(front_s0, self.stops[k] + 5.0), default=None)
        past = target is not None and self._ahead(front_s0, self.stops[target] + 5.0) < 5.0
        signal = signal if signal in ("red", "yellow", "green") else "unknown"
        ctx = dict(s0=s0, front_s0=front_s0, signal=signal, target=target, objects=None,
                   blackout_all=bool(blackout) and not any(a <= self._mod(s0) <= b for a, b in self.lc["blackout_s"]))
        if ctx["blackout_all"]:
            res["tags"].append("blackout_unmapped")   # GPS out where the map has no blackout: no lane/inoperable judging
        ob = [o for o in objects if math.hypot(o[2] - x0, o[3] - y0)
              < 6.4 * (v0 + math.hypot(o[5], o[6])) + self.lc["objects"]["search_m"]]
        # only moving objects can surprise the label; vehicles behind are judged anyway (lane changes, the merge)
        seen = visible(x0, y0, h0, ob, self.lc["visibility"]) | np.array(
            [math.hypot(o[5], o[6]) < self.lc["objects"]["static_mps"]
             or (str(o[1]) == "vehicle" and math.cos(h0) * (o[2] - x0) + math.sin(h0) * (o[3] - y0) < 0) for o in ob], bool)
        ctx["objects"] = self._objects([o for o, s_ in zip(ob, seen) if s_], x0, y0, h0)
        hidden = self._objects([o for o, s_ in zip(ob, seen) if not s_], x0, y0, h0)

        refs = self._references(L, s0, steer)
        names = [r_[0] for r_ in refs]
        lab = self._measure(*(z[None] for z in L), ctx, strict=False)
        ref = self._measure(*(np.stack([r_[i] for r_ in refs]) for i in (1, 2, 3, 4)), ctx, strict=True)
        tol = self.lc["tolerance"]
        v_ref = max(float(L[3].mean()), tol["min_ref_speed_mps"])
        cost_l = float(lab["cost"][0])
        cost_r = ref["cost"] + np.maximum(lab["progress"][0] - ref["progress"], 0.0) / v_ref

        sg = self.lc["signal"]
        tc = float(lab["t_cross"][0])
        stop_m = v0 * sg["max_brake_delay_s"] + v0 ** 2 / (2 * sg["max_brake_mps2"])
        stoppable = target is not None and not past and self._ahead(front_s0, self.stops[target]) >= stop_m
        go_clear = not ref["hit_cv"][names.index("route_go")]
        fails = []
        for rule in RULES:
            # route_go (pulling away) stands in only for leaving the grass: as a lane alternative it would fail every
            # label waiting with a wheel on paint for traffic the 2 s vehicle horizon or the data does not show
            usable = np.array([n != "route_go" or rule == "inoperable" for n in names])
            amount, best = float(lab[rule][0]), float(ref[rule][usable].min())
            res["label"][rule], res["oracle"][rule] = round(amount, 2), round(best, 2)
            if amount <= EPS[rule]:
                continue
            if rule == "signal":   # a red crossing: the user's rules, not the references
                if stoppable:
                    res["better"]["signal"] = "max_brake"
                    res["reasons"].append("signal:avoidable")
                    fails.append(rule)
                elif (L[3][np.arange(N + 1) * DT >= tc].min() < sg["junction_stop_mps"]) and go_clear:
                    res["reasons"].append("signal:red_stop_in_junction")
                    fails.append(rule)
                else:
                    res["reasons"].append("signal:unavoidable")
                continue
            rel, ab = tol[rule]
            if amount <= best * (1 + rel) + ab:
                res["reasons"].append(f"{rule}:{'unavoidable' if best > EPS[rule] else 'tolerated'}")
                continue
            # stopping behind a standing obstacle only defers passing it: no alternative to the paint it takes to pass
            blocked = ref["blocked"] & (rule in ("lane", "inoperable"))
            cand = (ref[rule] * (1 + rel) + ab < amount) & ~blocked & usable
            better = cand & (cost_r < cost_l - tol["cost_s"])
            if better.any():
                k = int(np.flatnonzero(better)[np.argmin(cost_r[better])])
                res["better"][rule] = names[k]
                res["reasons"].append("collision:lane_change_rear" if rule == "collision" and lab["lane_change_rear"][0]
                                      else f"{rule}:avoidable")
                fails.append(rule)
            else:   # better on this rule only at a cost: within the cost margin, or dearer (the label's is the cheaper side)
                res["reasons"].append(f"{rule}:{'tolerated' if (cand & (cost_r < cost_l)).any() else 'conflict'}")

        sig = ctx["signal"]
        if lab["signal"][0] and not stoppable:
            res["tags"].append("red_unstoppable_" + ("go" if "signal:unavoidable" in res["reasons"] else "stop"))
        if past and sig == "red":   # front axle already over the line: the light is judged; go once the way is clear
            stay = lab["progress"][0] - s0 < 2.0
            if stay and go_clear:
                res["reasons"].append("signal:stay_after_cross")
                fails.append("signal")
            res["tags"].append("red_past_line_" + ("go" if not stay else "stay" if go_clear else "wait"))
        elif (sig == "red" and target is not None and not np.isfinite(tc)
              and self._ahead(front_s0, self.stops[target]) < lab["progress"][0] + self.front - front_s0):
            res["tags"].append("red_line_bypassed")   # past the line's route s beside its painted extent: not judged
            # (the scorer's geometry), counted for the user
        if sig == "yellow" and np.isfinite(tc) and not lab["signal"][0]:
            stop_m = self._ahead(s0, self.stops[target]) - self.lc["signal"]["stop_offset_m"]
            res["tags"].append("yellow_stoppable_go" if v0 ** 2 / (2 * self.lc["signal"]["yellow_stop_mps2"]) <= stop_m
                               else "yellow_dilemma_go")
        if lab["off_route_end"][0] and not ref["off_route_end"].all():
            res["tags"].append("ends_off_route_lane")   # no rule broken yet, but a reference gets back to the route lane
        unjudged = []
        if lab["signal_unknown"][0]:
            unjudged.append("signal:unknown")
        if lab["off0"][0] and ref["inoperable"].min() >= N * DT - 0.05:
            unjudged.append("inoperable:unrecoverable")
        if lab["checkpoint"][0] and ref["checkpoint"].min() > EPS["checkpoint"]:
            unjudged.append("checkpoint:missed")
        res["reasons"] += unjudged
        hits = self._measure(*(z[None] for z in L), dict(ctx, objects=hidden), strict=False)["collision"][0] if hidden else 0
        if hits:   # what the model could not see at t0: the label's late reaction is legitimate
            res["reasons"].append("collision:hidden_at_t0")
        res["label"]["hidden_hits"] = int(hits)
        if lab["lane_change_rear_hidden"][0]:   # cut in on a car too far back for the shadow to see (user 10/28)
            res["reasons"].append("collision:lane_change_rear_hidden")
        violated = bool(hits) or lab["lane_change_rear_hidden"][0] or any(res["label"][r_] > EPS[r_] for r_ in RULES)
        lost = any(r_ in UNRECOVERABLE for r_ in unjudged)
        res["verdict"] = ("fail" if fails else "unrecoverable" if lost else "unjudgeable" if unjudged
                          else "pass_unavoidable" if violated else "pass")
        res["valid"] = not fails and not lost
        res["debug_fail"] = bool(fails) or "collision:lane_change_rear_hidden" in res["reasons"]   # shadow debugging only
        res["label"].update(signal_state=sig, t_cross=round(tc, 2) if np.isfinite(tc) else None,
                            struck=int(lab["struck"][0]), lane_penalty_s=self._lane_penalty(lab["contact"][0]),
                            cost=round(cost_l, 2), paint_t0=bool(lab["contact"][0, 0]), off_road_t0=bool(lab["off0"][0]),
                            d_t0=round(d0, 2))
        if debug:
            res["refs"] = {n: dict(xy=np.c_[r_[0], r_[1]], v=r_[3], cost=round(float(cost_r[i]), 2),
                                   **{k: float(ref[k][i]) for k in RULES + ("blocked",)}) for i, (n, *r_) in enumerate(refs)}
            res["label_xy"] = np.c_[L[0], L[1]]
        return res

    def _lane_penalty(self, contact):
        """scorer.score's lane penalty over the horizon (t0 included): +5 s per full period of continuous contact."""
        lane, period = self.cfg["lane"], round(self.cfg["lane"]["period_s"] / DT)
        return float(sum((i1 - i0 + 1) // period * lane["penalty_s"] for i0, i1 in _runs(contact)))
