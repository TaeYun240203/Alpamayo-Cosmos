# Step 0: training inputs on the restored sim clock

The output lives on winpc at `D:\share\2026_MORAI_step0\` (WSL `/mnt/d/share/2026_MORAI_step0/`). There is one folder per episode:

| File | Content |
|---|---|
| `cams.bin` + `cams_index.npz` | Every camera frame, resized and re-encoded as JPEG q92: Front 448×256, Left/Right 352×256, Cam7 256×192. The index holds `cam` (0 front, 1 left, 2 right, 3 cam7), `bag_ns`, sim-clock `tau`, `offset`, `length` |
| `lidar.bin` + `lidar_index.npz` | Per VLP16 scan, a BEV of 320×160 at 0.25 m/px (x 60…−20 m forward, y +20…−20 m left). Each packet is motion-compensated into the ego frame at the scan's last packet. One zlib blob per scan holds (2, 320, 160) uint8: `height` (0 = empty, else 0.05·h − 1.0 m above base_link ground) and `density` (point count, max 255). The index has `offset`, `length`, `tau`, `bag_ns`, `pose` (odometry x, y, yaw at the scan's last packet), `shape`, `npts` and the marker key `odom` |
| `lidar.npz` | Old format, from before 10/24 evening. It used GT-pose compensation and is no longer read or uploaded |
| `meta.json` | Frame and scan counts, timing. Its presence marks the episode done |

`_ego/<ep>.npz` (from `ego_tables.py` on the Mac) is the 10 Hz tau grid: `tau`, `bag_ns`, `x`, `y`, `yaw`, `v`, `a`, `yaw_rate`, `path_s`, `d` (lateral offset to the route, left +), `section`, `signal` (0 unknown, 1 red, 2 yellow, 3 green for the next judged line), `gps_blackout`, `clean`, `fog`, `hour`, `scene`.

Training-time assembly (agreed design):
- 4 frames per camera at tau 0/−0.1/−0.2/−0.3 s with 0.08–0.12 s jitter: the nearest `cams_index.tau`.
- 4 scans at the same tau grid, packed 2 per BEV with `lidar_bev.pack_bev`, which warps the older scan into the newer one's frame by the relative odometry pose. Channels: newer height, older height, newer density ×16.

## LiDAR motion: speed + yaw rate only (10/24)
- All LiDAR geometry lives in [`train/morai_vla/lidar_bev.py`](../../train/morai_vla/lidar_bev.py). It is pure numpy and also runs in this chroot. The step-0 worker, the training loader and the inference code all import this one file.
- Motion comes only from speed and yaw rate, integrated by `Odometry`.
  - Here: the `_ego` GT `v` and `yaw_rate`, interpolated to each packet's bag_ns and integrated over sim time tau.
  - Inference: competition_status speed + IMU yaw rate. The docstring explains how.
- Why not GT poses: the `_ego` x, y are Ego status placed at arrival time and jitter ±30 % over short spans, and the competition has no GT pose.
- Check on 20 episodes (3,150 consecutive-scan pairs, static structure >1 m tall and >6 m away):

| | GT pose (old) | odometry (new) |
|---|---|---|
| pairs aligned (best shift 0) | 54 % | 67 % |
| >10 m/s: aligned | 35 % | 52 % |
| >10 m/s: mean shift | 1.84 px | 1.59 px |

  - Scaling the translation by 0.8–1.25 or the yaw change by −1–2 aligns worse, so sign and scale are right.
- Test: `train/tests/test_lidar_bev.py` (line and circle motion; runs under pytest or as a plain script in the chroot).
- Ego history and future (a, κ) come from the `_ego` grid. Nav text comes from `path_s` and `tools/coc_labeler/route.yaml`.

## Run
```
python ego_tables.py ~/.cache/step0_ego --jobs 8        # Mac
COPYFILE_DISABLE=1 tar cf - -C ~/.cache/step0_ego . | ssh winpc-wsl 'tar xf - -C /mnt/d/share/2026_MORAI_step0/_ego'
scp step0_worker.py run.sh launch.sh ../../train/morai_vla/lidar_bev.py winpc-wsl:step0/ && ssh winpc-wsl 'cd ~/step0 && ./launch.sh 16'
# LiDAR only, for finished episodes (skips those whose lidar_index.npz already has `odom`):
ssh winpc-wsl 'cd ~/step0 && for k in $(seq 0 15); do nohup ./run.sh step0_worker.py lidar $k 16 > /dev/null 2>&1 & done'
ssh winpc-wsl python3 - < check.py                      # progress / failures
```
Workers run inside the Noetic chroot (rosbag, numpy 1.17, cv2 4.2). The launcher can be re-run safely, because finished episodes are skipped.

## R2 collection (10/27)
Same layout in `D:\share\2026_MORAI_step0_R2\` (B2 `step0_r2/`), for the QA-ok R2 episodes only (`_meta/ok_episodes.txt`), from `D:\share\2026_MORAI_collect_R2_20261027`:
```
python ego_tables.py OUT --only ok.txt --jobs 8                                    # labels copied into labels-cache/{mac-win-1,mac-win-2,mac-bc-1}
ssh winpc-wsl 'cd ~/step0_r2 && STEP0_ROOT=/mnt/2026_MORAI_collect_R2_20261027 STEP0_OUT=/mnt/2026_MORAI_step0_R2 STEP0_DIR=2026_MORAI_step0_R2 ./launch.sh 16'
python -m morai_vla.build_index _index --ego OUT --held-out 40100027,40100029,50100001  # cwd train/
python r2_meta.py _meta --qa qa_all.jsonl --ok ok.txt --ego OUT --index _index/index.parquet --r2logs R2LOGS
```
- `_labels/`: the scorer labels (`traj_ok` already excludes noise windows and collisions; the noise flag is `/collect_meta/noise_active`).
- `_meta/r2_episodes.parquet`: one row per episode: kind A/B/D, scene, start pose/speed (D offset class in `start_cls`), actors (id, class, path type, timing class, speed), noise seconds, collision/AEB/yield intervals on tau, QA numbers, index rows (10 Hz and every 5th = 2 Hz).
- No `_coc/` (no CoC labels for R2).
