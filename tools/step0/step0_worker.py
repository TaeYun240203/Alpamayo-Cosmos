"""Step 0, winpc side (Noetic chroot: python3.8, numpy 1.17, cv2 4.2, rosbag). One pass over each episode's bags:
  cameras -> resized JPEG (Front 448x256, Left/Right 352x256, Cam7 256x192, q92) stamped with sim-clock tau
  /velodyne_packets -> per-scan BEV via train/morai_vla/lidar_bev.py (copy it next to this file on winpc): each packet
                       motion-compensated into the ego frame at the scan's last packet, with motion integrated from the
                       _ego GT speed and yaw rate over tau (the same code the inference uses with competition_status + IMU)
Outputs <OUT>/<ep>/: cams.bin + cams_index.npz, lidar.bin + lidar_index.npz, meta.json (written last; its presence = done).
Usage (inside the chroot): python3 step0_worker.py WORKER NWORKERS          full step 0
                           python3 step0_worker.py lidar WORKER NWORKERS [DST]   LiDAR only, for finished episodes
                           python3 step0_worker.py one EP | lidar-one EP [DST]"""
import glob, json, os, sys, time, zlib

import cv2, numpy as np, rosbag

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "train", "morai_vla")]
import lidar_bev as L   # noqa: E402

ROOT = os.environ.get("STEP0_ROOT", "/mnt/2026_MORAI_collect_20261018")   # R2: /mnt/2026_MORAI_collect_R2_20261027
OUT = os.environ.get("STEP0_OUT", "/mnt/2026_MORAI_step0")                 # R2: /mnt/2026_MORAI_step0_R2
CAMS = {"/image/front/compressed_10hz": (0, (448, 256)), "/image/left/compressed_10hz": (1, (352, 256)),
        "/image/right/compressed_10hz": (2, (352, 256)), "/image/camera_7/compressed_10hz": (3, (256, 192))}
LIDAR = "/velodyne_packets"
ODOM_TAG = "lidar_bev.Odometry: _ego v, yaw_rate at packet bag_ns, integrated over tau"


class Clock:
    """bag time (ns) -> sim tau and 2-D pose, from the Mac-side ego table (linear extrapolation at the ends)."""
    def __init__(self, f):
        z = np.load(f); self.ns = z["bag_ns"].astype(np.float64); self.tau = z["tau"]
        self.v, self.r = z["v"].astype(np.float64), z["yaw_rate"].astype(np.float64)
        k = max(2, len(self.ns) // 20)
        self.slope0 = (self.tau[k] - self.tau[0]) / max(self.ns[k] - self.ns[0], 1)
        self.slope1 = (self.tau[-1] - self.tau[-k]) / max(self.ns[-1] - self.ns[-k], 1)

    def tau_at(self, ns):
        ns = np.asarray(ns, np.float64); t = np.interp(ns, self.ns, self.tau)
        t = np.where(ns < self.ns[0], self.tau[0] + (ns - self.ns[0]) * self.slope0, t)
        return np.where(ns > self.ns[-1], self.tau[-1] + (ns - self.ns[-1]) * self.slope1, t)

    def vr(self, ns):
        return np.interp(ns, self.ns, self.v), np.interp(ns, self.ns, self.r)


class Lidar:
    """per-scan BEVs of one episode -> <d>/lidar.bin (zlib (2, 320, 160) uint8 height, density per scan) + lidar_index.npz,
    written to tmp files and swapped in with os.replace (the morai_vla.data layout)."""
    def __init__(self, d, clk):
        self.d, self.clk, self.odom = d, clk, L.Odometry()
        self.f = open(os.path.join(d, "lidar.bin.tmp"), "wb"); self.off, self.len, self.ns, self.pose, self.npts = [], [], [], [], []

    def add(self, scan):
        stamps = np.array([p.stamp.to_nsec() for p in scan.packets], np.float64)
        v, r = self.clk.vr(stamps)
        h, den, n, pose = L.scan_bev_stream(b"".join(bytes(p.data) for p in scan.packets), self.clk.tau_at(stamps), v, r, self.odom)
        b = zlib.compress(np.stack([h, den]).tobytes(), 1)
        self.off.append(self.f.tell()); self.len.append(len(b)); self.f.write(b)
        self.ns.append(int(stamps[-1])); self.pose.append(pose); self.npts.append(n)

    def close(self):
        self.f.close(); ns = np.array(self.ns, np.int64)
        np.savez(os.path.join(self.d, "lidar_index.tmp.npz"), offset=np.array(self.off, np.int64), length=np.array(self.len, np.int32),
                 tau=self.clk.tau_at(ns).astype(np.float32), pose=np.array(self.pose, np.float64).reshape(-1, 3), bag_ns=ns,
                 shape=np.array([L.H, L.W]), npts=np.array(self.npts, np.int32), odom=np.array(ODOM_TAG))
        os.replace(os.path.join(self.d, "lidar.bin.tmp"), os.path.join(self.d, "lidar.bin"))
        os.replace(os.path.join(self.d, "lidar_index.tmp.npz"), os.path.join(self.d, "lidar_index.npz"))
        return len(ns)


def bags(ep):
    return sorted(glob.glob("%s/*/%s_*.bag" % (ROOT, ep))) or glob.glob("%s/*/%s.bag" % (ROOT, ep))


def lidar_only(ep, dst=OUT):
    """rewrite <dst>/<ep>/lidar.bin + lidar_index.npz from the bags (cameras untouched); skipped if already odometry-based."""
    d = os.path.join(dst, ep); idx = os.path.join(d, "lidar_index.npz")
    if os.path.exists(idx) and "odom" in np.load(idx).files and not os.environ.get("STEP0_FORCE_LIDAR"): return "skip"   # R2 10/27: _ego v fixed
    paths, ego = bags(ep), os.path.join(OUT, "_ego", ep + ".npz")
    if not paths or not os.path.exists(ego): return "missing " + ("bags" if not paths else "ego")
    if not os.path.isdir(d): os.makedirs(d)
    t0 = time.time(); lid = Lidar(d, Clock(ego))
    for path in paths:
        b = rosbag.Bag(path)
        for _, msg, _ in b.read_messages(topics=[LIDAR]): lid.add(msg)
        b.close()
    return "ok scans=%d %.0fs" % (lid.close(), time.time() - t0)


def episode(ep):
    d = os.path.join(OUT, ep)
    if os.path.exists(os.path.join(d, "meta.json")): return "skip"
    paths = bags(ep)
    ego = os.path.join(OUT, "_ego", ep + ".npz")
    if not paths or not os.path.exists(ego): return "missing " + ("bags" if not paths else "ego")
    if not os.path.isdir(d): os.makedirs(d)
    clk = Clock(ego); t0 = time.time()
    cams = open(os.path.join(d, "cams.bin.tmp"), "wb"); idx = []; off = 0
    lid = Lidar(d, clk)
    for path in paths:
        b = rosbag.Bag(path)
        for topic, msg, t in b.read_messages(topics=list(CAMS) + [LIDAR]):
            if topic == LIDAR:
                lid.add(msg)
            else:
                cid, size = CAMS[topic]
                im = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                if im is None: continue
                ok, enc = cv2.imencode(".jpg", cv2.resize(im, size, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
                buf = enc.tobytes(); cams.write(buf)
                idx.append((cid, t.to_nsec(), off, len(buf))); off += len(buf)
        b.close()
    cams.close()
    I = np.array(idx, np.int64).reshape(-1, 4)
    np.savez(os.path.join(d, "cams_index.npz"), cam=I[:, 0].astype(np.uint8), bag_ns=I[:, 1], tau=clk.tau_at(I[:, 1]).astype(np.float32),
             offset=I[:, 2], length=I[:, 3].astype(np.int32), cam_names=np.array(["front", "left", "right", "cam7"]))
    scans = lid.close()
    os.rename(os.path.join(d, "cams.bin.tmp"), os.path.join(d, "cams.bin"))
    meta = dict(episode=ep, bags=[os.path.basename(p) for p in paths], frames={n: int((I[:, 0] == i).sum()) for n, i in (("front", 0), ("left", 1), ("right", 2), ("cam7", 3))},
                scans=scans, seconds=round(time.time() - t0, 1), cam_bytes=off)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    return "ok %s" % meta


if __name__ == "__main__":
    cv2.setNumThreads(1)
    if sys.argv[1] in ("one", "lidar-one"):
        print(episode(sys.argv[2]) if sys.argv[1] == "one" else lidar_only(*sys.argv[2:4])); sys.exit(0)
    fn, a = (lidar_only, sys.argv[2:]) if sys.argv[1] == "lidar" else (episode, sys.argv[1:])
    k, n = int(a[0]), int(a[1]); dst = a[2] if len(a) > 2 else OUT
    eps = sorted({os.path.basename(p).rsplit(".", 1)[0] for p in glob.glob(os.path.join(OUT, "_ego", "*.npz"))})
    mine = [e for e in eps if zlib.crc32(e.encode()) % n == k]
    if fn is lidar_only: mine = [e for e in mine if os.path.exists(os.path.join(OUT, e, "meta.json"))]
    log = open(os.path.join(OUT, "_logs", "%s_%02d.log" % ("lidar" if fn is lidar_only else "worker", k)), "a")
    for e in mine:
        try:
            st = fn(e, dst) if fn is lidar_only else fn(e)
        except Exception as ex:
            st = "FAILED %s: %s" % (type(ex).__name__, ex)
        log.write("%s %s %s\n" % (time.strftime("%H:%M:%S"), e, st)); log.flush()
