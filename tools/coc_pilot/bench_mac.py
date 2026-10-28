"""Step-0 CPU cost on the Mac: same work as winpc bench_step0.py/par2.py, rosbags instead of rosbag."""
import sys, time, os
import numpy as np, cv2
from pathlib import Path
from multiprocessing import Pool

CAM = {"/image/front/compressed_10hz": (448, 256), "/image/left/compressed_10hz": (352, 256),
       "/image/right/compressed_10hz": (352, 256), "/image/camera_7/compressed_10hz": (256, 192)}
OTHER = ["/collect_only/ego_status", "/vehicle/competition_status", "/sensors/gps/fix", "/sensors/imu/data"]
VERT = np.radians(np.array([-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15], dtype=np.float32))


def decode_scan(packets):
    raw = np.frombuffer(b"".join(bytes(p.data) for p in packets), dtype=np.uint8).reshape(-1, 1206)[:, :1200]
    blk = raw.reshape(-1, 12, 100)
    az = (blk[:, :, 2].astype(np.uint16) | (blk[:, :, 3].astype(np.uint16) << 8)).astype(np.float32) / 100.0
    ch = blk[:, :, 4:].reshape(-1, 12, 32, 3)
    dist = (ch[..., 0].astype(np.uint16) | (ch[..., 1].astype(np.uint16) << 8)).astype(np.float32) * 0.002
    a = np.radians(az)[..., None] + np.where(np.arange(32) >= 16, np.radians(0.1), 0.0)
    w = np.tile(VERT, 2)[None, None, :]
    xy = dist * np.cos(w)
    pts = np.stack([xy * np.cos(a), -xy * np.sin(a), dist * np.sin(w)], -1).reshape(-1, 3)
    return pts[dist.reshape(-1) > 0.1]


def bev(pts):
    m = (pts[:, 0] > -20) & (pts[:, 0] < 60) & (np.abs(pts[:, 1]) < 20)
    p = pts[m]; r = ((60 - p[:, 0]) / 0.25).astype(np.int32); c = ((20 - p[:, 1]) / 0.25).astype(np.int32)
    img = np.zeros((320, 160, 3), np.float32)
    np.maximum.at(img[:, :, 0], (r, c), p[:, 2] + 1.4)
    np.add.at(img[:, :, 2], (r, c), 1.0)
    return img


def work(args):
    data, size, reps = args
    cv2.setNumThreads(1)
    for _ in range(reps):
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imencode(".jpg", cv2.resize(im, size, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return reps


if __name__ == "__main__":
    from rosbags.highlevel import AnyReader
    path = Path(sys.argv[1])
    t0 = time.time(); n = {}; imgs = {k: [] for k in CAM}; scans = []
    with AnyReader([path]) as r:
        conns = [c for c in r.connections if c.topic in list(CAM) + OTHER + ["/velodyne_packets"]]
        for c, ts, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)  # full deserialize of every message, like rosbag read_messages
            n[c.topic] = n.get(c.topic, 0) + 1
            if c.topic in CAM and len(imgs[c.topic]) < 60: imgs[c.topic].append(bytes(m.data))
            if c.topic == "/velodyne_packets" and len(scans) < 40: scans.append(m.packets)
        dur = (r.end_time - r.start_time) / 1e9
    t_read = time.time() - t0
    cv2.setNumThreads(1)
    t1 = time.time(); k = 0
    for topic, lst in imgs.items():
        for d in lst:
            work((d, CAM[topic], 1)); k += 1
    t_img = (time.time() - t1) / k
    t2 = time.time()
    for s in scans: bev(decode_scan(s))
    t_lid = (time.time() - t2) / len(scans)
    print("bag_sec=%.1f read_decode_s=%.1f size_MB=%.0f" % (dur, t_read, os.path.getsize(path) / 1e6))
    print("counts", {k.split("/")[2] if k.count("/") > 1 else k: v for k, v in n.items()})
    print("per image ms=%.2f (4-cam mix)  per lidar scan ms=%.1f" % (t_img * 1e3, t_lid * 1e3))
    front = imgs["/image/front/compressed_10hz"][0]
    t = time.time(); work((front, (448, 256), 100)); print("front only ms=%.2f" % ((time.time() - t) * 10))
    for w in (1, 5, 8, 11, 14):
        with Pool(w) as p:
            t = time.time(); p.map(work, [(front, (448, 256), 200)] * w)
            print("workers=%2d front img/s=%.0f" % (w, 200 * w / (time.time() - t)))
