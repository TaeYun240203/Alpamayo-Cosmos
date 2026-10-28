# coc_labeler — Chain-of-Causation labels for the MORAI dataset

Replaces the NVlabs `alpamayo-coc-autolabeler` for this dataset. It follows the AR1 paper (arXiv 2511.00088, Sec. 4) more closely than the released autolabeler prompt.

| Paper requirement | How it is done here |
|---|---|
| Keyframes 0.5 s before the behaviour change (Tab. 3) | `events.py` uses the ego state on the restored sim clock plus privileged events: signals, scenario actors and the route annotation |
| Closed-set decision, at most 1 longitudinal + 1 lateral (Tab. 1) | Taken from the ego's ground-truth motion, never from a VLM. Keyframes ≤ 1.5 s apart are merged |
| Critical components (Tab. 2) | Candidates come from privileged information: signal state and movement (`route.yaml`), scenario actors (manifests), permanent route features (`route.yaml`) |
| Causal locality | The VLM sees **only the 2 s history frames** (t = −2…0 s). Each candidate must be confirmed visible there; otherwise the keyframe is `unobserved` and gets no CoC |
| Annotation economy, proximate cause | One cause per channel. The sentence is composed from templates in the release-prompt example style |
| Vehicles without ground truth (roundabout, merge, highway) | A closed-form VLM query on the history frames (`observe.npc`) returns position, distance and motion enums. Rules pick the cause |
| QA (Tab. 4, ~10 % human) | Every label gets a decision-vs-GT check (`qa.gt_check`). `review.html` holds a 10 % sample for human audit. The official 3-stage VLM validation is optional |

## Run
```
cd tools/coc_labeler
python -m coc_labeler.cli --out OUT --all --frames winpc --max-used-pct 45 --workers 8 --shard 0/3   # x3 shards
python -m coc_labeler.cli summarize OUT
python -m coc_labeler.official_val OUT --judge gpt-6-astra --max 120
python -m coc_labeler.cli recompose OLD_OUT NEW_OUT     # new compose/events code on stored VLM answers, no Codex calls
```
- `--frames winpc` extracts only the needed JPEGs inside the winpc Noetic chroot; bags never move. A folder path instead reads local `<ep>_<i>.bag` files.
- Codex is called with the ChatGPT OAuth token in `~/.codex/auth.json`, with our own instructions (no CLI system prompt), strict JSON output and a disk cache (`~/.cache/coc_labeler/vlm.jsonl`). `--max-used-pct` is an absolute weekly-usage ceiling shared by all shards (`--budget-pct` would be counted per process).
- The run is resumable: finished episodes (`OUT/labels/<ep>.jsonl`) are skipped.

Outputs:
- `labels/<ep>.jsonl`: full records, including candidates, verification answers and vehicles.
- `official/<ep>/cot_<bag_ns>.yaml`: the autolabeler output format, so its validation scripts accept it.
- `coc_dataset.parquet`: one row per keyframe.
- `review.html`: the human audit sample.
- `summary.json`: counts and GT agreement.

## Inputs (synced caches under `~/.claude-morai-mac/`)
`labels-cache/` (`tools/ledger.py --pull`), `manifests/` (winpc `SaveFile/Scenario/R_KR_PR_K-city_2025/*-manifest.json`), `episode-scenes.log` + `.wsl.log`, env logs.

## Known limits
- No vehicle ground truth exists (the bags have no `/objects`). Vehicle causes come only from the VLM, restricted to the history frames.
- The collection driver acts on ground truth: it holds red lights 30–40 m before the line and starts bypasses before an obstacle is camera-visible. Such keyframes come out `unobserved` by design.
- `route.yaml` is specific to the competition route.

## Runs (2026-10-24, parquets copied to winpc `D:/share/2026_MORAI_step0/_coc`)

| | v1 `morai-coc-1.0` (`~/coc_runs/coc_v1`, `coc_dataset.parquet`) | v1.1 `morai-coc-1.1` (`~/coc_runs/coc_v1_1`, `coc_dataset_v1_1.parquet`) |
|---|---|---|
| How | Full run with Codex verification | `cli recompose` of v1: same keyframes, stored VLM answers, current events/compose code. 0 Codex calls |
| Episodes | 1365 (run1021a-w2-027 is too short). 8 have labels but no bags (`has_step0 = False`) | same |
| Keyframes | 18,517. Status: ok 13,671 · unobserved 3,849 · invalid_no_signal 851 · no_frames 146 | same |
| `invalid_no_signal` | 851 keyframes at s 160: v1 wrongly treated s 160 as signalized, and MGeo has no light for the route there | carried over from v1, no CoC |
| Training set (`use`: observed cause, decision agrees with ego GT, bags exist) | 13,386: C0 3,622 · C1 981 · C2 1,232 · C3 2,349 · C4 1,189 · C5 2,675 · C6 1,338 | 13,386, same in every cell |
| Decision vs ego GT | longitudinal 98.9 %, lateral 99.1 % | same |
| Sentences changed from v1 | | 382 labels. 332 work-zone labels cited drums the VLM did not see (241 in fog). They now read "… ahead of the construction zone" (component type `construction zone`). 8 slow right turns at int551 read "Proceed through …". Their decision is now Turn right, and they read "Turn right through the intersection on the green light." 13 read "while slowing down and stop" and now say "stopping". 29 bypass labels right after a departure now keep the departure's longitudinal decision ("Slow down and nudge" → "Nudge"), as 217 other ok v1 labels already did. `events.py` changed during the v1 run |
| Official 3-stage validation, 120 random QA labels, gpt-6-astra | stage 1 98.3 % · stage 2 95.8 % · stage 3 98.3 % · **all 94.2 %** (C2 0.91, C3 0.92, C5 0.89, C6 0.82, others 1.0) | not rerun |
| For comparison: the release autolabeler prompt run by Codex, on pilot keyframes | 53–60 % all-stage | |
| Distinct sentences (ok) | 877 | 883 |
| Codex cost | ~670 verification calls per 1 % of the weekly Pro window. The whole run (with validation) took the window from 14 % to 44 % | 0 |
