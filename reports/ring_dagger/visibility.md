# Roundabout entry: can WoWmayo see the ring cars on its left? (2026-10-29)

The method is geometry only. No real model-input frame from the ring entry exists on the Mac (details at the end).
Re-run with `nice -n 10 /usr/bin/python3 reports/ring_dagger/ring_visibility.py` (about 2 s). All numbers are in `visibility.csv`.

## Answer
- **Approach (front axle s 870–880, 17.5–7.5 m before the wait line).** Every left ring car that arrives in T = 1–6 s is inside the **Front** image:
  - s 880: 17–85 px wide × 9–27 px tall;
  - s 870: 17–48 px wide × 6–15 px tall.
  - The car is also in the Left camera and in the BEV while it is within 20 m to the side.
- **At the wait line (front s 887.5, standing), the cars the ego must yield to leave the Front image.** Bearings are from the rear axle, left +.
  - Some cars sit at +51° to +54°, outside the Front ±45°: 3–4 s away at 8 m/s, 2–3 s at 10 m/s, 4–6 s at 5 m/s. They are seen **only by the Left camera, and they are tiny**: 10–20 px wide × 6–11 px tall. At 8 m/s they are 11–13 × 6–8 px.
  - One ViT token is 32×32 px.
  - The BEV loses these cars once they are more than 20 m to the side. That happens at T ≥ 4 s at 8 m/s.
  - At T 6 s (8 m/s) and T 5–6 s (10 m/s), the car is on the far side of the ring, 39–44 m away. It comes back into the Front image as a 9–12 px tall box, seen across the island.

| wait line, 8 m/s (measured p50 6.3 / p90 7.7) | T 2 s | T 3 s | T 4 s | T 5 s |
|---|---|---|---|---|
| range @ bearing | 11.5 m @ +44° | 19 m @ +54° | 27 m @ +51° | 34 m @ +44° |
| (i) Front 448×256 | 14 % of car only | no | no | 31 % of car, 20×15 |
| (ii) Left 352×256 | **44×18** | **11×8** | **13×6** | **14×4** |
| Cam7 256×192 | 49 % of car | no | no | 18×9 |
| LiDAR BEV | yes, 5 rings | yes, 2 rings | 19 % of car (>20 m lateral) | **no** |

| wait line, 5 m/s | T 2 s | T 3 s | T 4 s | T 5 s |
|---|---|---|---|---|
| (i) Front | 201×68 | 32 % of car | no | no |
| (ii) Left | 110×44 (65 % of car) | 55×20 | 20×11 | 10×8 |
| LiDAR BEV | 6 rings | 5 rings | 3 rings | 2 rings |

At 10 m/s (above the measured p90):

| wait line, 10 m/s | T 2 s | T 3 s | T 4 s | T 5 s |
|---|---|---|---|---|
| (i) Front | no | no | partial | 42×11 |
| (ii) Left | 20×11 | 12×6 | 14×4 | 17×4 |
| LiDAR BEV | yes | 59 % of car | no | no |

- **(iii) LiDAR BEV only:** no case. Every car that the BEV sees is also inside some camera image.
- **(iv) Invisible:** no case. In all 54 cells (3 decision points × 3 speeds × 6 T), at least 50 % of the car is inside at least one camera image.
- **The gap is resolution, not coverage.** At the wait line, a car 3–5 s away is 4–8 px tall in the side camera. That is a quarter of a token or less. For T ≥ 4 s at 8 m/s, the car also has no BEV point. With a VLP16, cars at 19–35 m get 1–2 rings anyway.

Figures:
- [fig_a_topdown.png](fig_a_topdown.png): ego frame at the 3 decision points, camera FOV wedges, BEV box, ring cars coloured by T.
- [fig_b_visibility.png](fig_b_visibility.png): every (decision point, speed, T) cell. It shows px size per camera, BEV rings, range and bearing.
- [fig_c_projection.png](fig_c_projection.png): predicted boxes drawn on blank canvases at the model-input size, for the standing ego. The Left camera column shows how small the cars are at T 3–6 s.
- [fig_d_ring_speeds.png](fig_d_ring_speeds.png): measured ring speeds.

## What is measured / taken from code, and what is assumed

**Measured or taken from code:**
- **Cameras:** pose and native size from `MORAI-eval-dagger/spec/0918sensors.json`.

  | camera | native | FOV | position (x, y, z) | yaw / pitch | model input | fx / fy at input |
  |---|---|---|---|---|---|---|
  | Front | 1280×720 | 90° | 1.9, 0, 1.2 | 0° / 2° down | 448×256 | 224 / 228 |
  | Left | 640×480 | 130° | 1.15, ±0.65, 1.2 | +70° / 10° down | 352×256 | 82 / 80 |
  | Right | 640×480 | 130° | 1.15, ±0.65, 1.2 | −70° / 10° down | 352×256 | 82 / 80 |
  | Cam7 | 640×480 | 90° | 0, 0, 1.5 | 0° / 10° up | 256×192 | 128 / 128 |

  - Model input sizes come from `train/morai_vla/processor.py`.
  - The yaw and pitch signs are asserted against the `sensor_mounts.yaml` quaternions.
- **LiDAR BEV:** from `train/morai_vla/lidar_bev.py`.
  - Extent: x −20…60 m, |y| < 20 m, 0.25 m cells, 320×160.
  - VLP16 at (1.5, 0, 1.403) m, beams at ±15° in 2° steps.
- **Ring circle:**
  - Centre (−102.36, 343.88), r = 18.4 m. This is the `dagger_label.py` fit on Object Info.
  - Counter-clockwise: 100 % of moving samples in 6 local eval laps (10/28) go that way.
  - Speeds on the arc before the entry: moving p10/p50/p90 = 3.7 / 6.3 / 7.7 m/s.
  - Size: every ring car in those logs is 4.8×2.1×1.5 m.
- **Route and conflict point:**
  - The route is `route.json`. The front axle is base_link + 3.0 m.
  - Conflict point = where the route crosses r = 18.4 m: s 899.25, angle −53.6°. "T" is the time until the ring car's centre reaches it.

**Assumed:**
- **Camera intrinsics:**
  - The FOV is horizontal with square pixels: fx = fy = W / (2 tan(FOV/2)). This is the stack's camera_info convention (MORAI-UDP-Bridge-Node).
  - It was not re-calibrated on pixels here.
  - Resize is per-axis, with no lens distortion.
- **Car box:** 4.6×1.9×1.5 m, as asked. With the measured 4.8×2.1, the pixel sizes are about 4–10 % larger.
- **Car motion:** cars drive exactly on the circle at constant speed. Not modelled: cars that join from other arms, cars hiding each other, the gore crash cushion and guardrail, and the ego's own body in the side views.
  - The cars sit near the horizon, above the ego's mirror and body (`train/logs/sample_cams.jpg` shows where the body is).
- **Ego pose:** yaw = route tangent (86° at the wait line). If the standing ego is turned further right, the cars move further out of the Front image.
- **Central island:** the edge radius (15 m) and the height are **unknown**.
  - The viewer (chase-cam) frames show a flat grass island with a low curb and lamp posts: `~/morai-eval/pod/20261028-pre_r2ft/laps/…_lap01.mp4` at 188 s and 215 s.
  - Only far-side cars look across it. At the wait line that is T ≥ 5 s at 8 m/s and T ≥ 4 s at 10 m/s. Over the island, the sightline is at least 1.2 m high.
  - So the island itself should not hide a car. Trees or signs taller than about 1.2 m would.
- **LiDAR counts:** "rings" = VLP16 beams that cross 0.1–1.5 m height at the car's nearest face. This is geometry, not ray-cast.

## Model input frames
None of these is a model-input frame at the ring entry:
- The lap videos in `~/morai-eval` are 540×540 viewer chase-cam footage.
- The only `*.req.bin` dumps (`~/claude-bench/rl-tiny`) are at s 148.
- `train/logs/sample_cams.jpg` is at a signalised intersection.

So fig c uses blank canvases at the model-input size.
