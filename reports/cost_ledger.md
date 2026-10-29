# Cost ledger (every paid box)

| id | rented | $/h | purpose | target | end condition | destroyed | $ spent | waste ($, reason) |
|---|---|---|---|---|---|---|---|---|
| 58129242 | 10/29 04:09 | 72.68 | dynbal ft+QAD (popout) | ft 1000 + QAD 800 step | deliver to B2 runs/dynbal then safe_destroy | 10/29 06:01 | ~$202 (2.78 h) | ~$23 idle 19 min (B2 upload unchecked, synth_hist cols) + ~$66 vs Brev $24/h not searched |
| 58143395 | 10/29 05:02 | 0.76 | history probe (s2/R2ft/r2stop/dynbal) | 156 rows x 4 models x 3 hist x 2 seeds | results on B2 runs/histprobe, destroy on user's OK | 10/29 05:57 | 0.77 | 0.05: first runner failed (b2 sync does not create /workspace/ck), fixed and relaunched |
| 58146365 | 10/29 05:30 | 0.60 | dev: gRPC ring NPC set + ego-timed mover director | ring set n-lap swap + must_stop/pass_front verified in sim | verified or 2 h -> B2 sync, ask, destroy | 10/29 06:42 | 0.75 | ~$0.07: v3 ran old code (ATTACH skipped the code sync); eval_pod trap destroyed the dev box at run end without asking |
| 58150309 | 10/29 06:06 | 0.56 | lane-return test (R2ft/r2stop/dynbal) | 198 R2 D val rows x 3 models x 3 conditions | preds on B2 runs/lanetest, destroy on user's OK | 10/29 06:57 | 0.51 | - |
| 58154249 | 10/29 06:46 | 0.81 | r2ft lap videos v3b: continuous laps + world director, first real-sim run | 8 continuous laps with videos; lap 1: pool scene, mover timing, ring per-lap set, respawn | 8 laps or 330 min -> B2 runs/eval_vps, eval_pod destroys; lap 1 fails -> logs to B2, ask, destroy | 10/29 08:48 | 1.65 | destroyed 08:48 by eval_pod's trap after my kill missed it (zsh: unquoted $P not split): not asked; on-box records of v3d/v3e lost |
| 58160695 | 10/29 07:52 | 34.22 | LwF dynstop ft+QAD (R2ft teacher, popout v2) | ft 2000 + QAD 800 step, onset 제동률 기록 | deliver on B2 runs/lwf, then safe_destroy | | | |
| 58166613 | 10/29 08:52 | 0.64 | r2ft lap videos v3f: time seed, stuck_loop skip (user '지금 띄워' 10/29 08:49) | 8 continuous laps with videos | 8 laps or 330 min -> eval_pod destroys | 10/29 10:54 | 1.32 | - |
