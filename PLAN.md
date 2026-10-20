# bdd-yolo-robustness: plan

## RESUME POINT (update every session)
- **>>> NEXT SESSION START HERE (written 2026-10-19 ~22:40) <<<**
  1-2. DONE 23:05: all 10 runs + evals finished. TEST 4 (descriptive): night mAP 0: 14.94, 100: 17.29,
     250: 17.13, 500: 17.91, 1000: 18.59, 2000: 19.77; half-gain target 17.35 first reached at 500 as
     coded, BUT 100 misses it by only 0.06 (far inside seed noise ~0.7) -> write "~100-500, most of the
     first jump comes from the first 100 images". Day stays 27.2-28.2 across doses. Single seed per dose.
  3. Ask user: OK to install TensorRT 10.16.1.11 (~4.3 GB, recommended) vs 11.x (see M4 PREP).
  4. `m4.py export`, `bench`, `eval`; add test 5 to compare.py; judge. 5. README (state test 3 as
  "small day cost under the bar"; train-class caveat + 9-class sensitivity), push, then DeepPCB.
  Scripts written this session: compare.py, analyze_m3.py, m4.py (all uncommitted; no git repo yet).
- **2026-10-19 20:25 VERDICTS (compare.py, runs/compare.json)**: TEST 1 PASS (+12.37 [+9.50,
  +14.50]; 9-class sensitivity +10.71 [+8.42, +12.95]). TEST 2 PASS: night B-C +2.93 mean over
  seeds (s0 +2.47 [+1.30, +3.81], s1 +3.39 [+2.50, +4.63]) vs seed spread A 1.06 / B 0.66 / C 0.27.
  TEST 3 PASS AS WRITTEN BUT BORDERLINE: day B-C -0.76 mean (s0 -0.48 [-1.18, +0.22], s1 -1.04
  [-1.77, -0.29] = CI excludes 0) -> report "small day cost, under the 1.0 bar", never "no cost".
  Dose so far: 0 -> 14.94, 100 -> 17.29, 500 -> 17.91 (100 night images already give most of it).
  Dose jobs running (dose250 started ~20:20), then `python compare.py` for test 4. Then M4 TensorRT.
- **>>> 2026-10-20 ~12:10 M4 DONE, TEST 5 JUDGED. NEXT = README, git init, push, then DeepPCB <<<**
  User chose the workaround: INT8 engines keep 13/88 post-concat convs FP16 (m4.py FP16_PINS, asserted).
  mAP50-95 (day / night / dusk): torch 27.38/17.91/21.82; FP16 engine 27.30/17.78/21.86; INT8 day-calib
  26.57/16.61/21.12; INT8 mixed 26.67/16.92/20.60.
  TEST 5: FAIL AS WRITTEN. Day-calib INT8 loss day +0.80 [+0.41,+1.16], night +1.30 [+0.75,+2.13];
  night-minus-day +0.4955 (prints +0.50!) [-0.13,+1.38] -> NOT > 0.5, and CI includes 0. Write: "night
  loses more in point terms, but the excess is at the bar and not distinguishable from 0". Vs FP16
  engine: +0.44. Mixed calib: night loss 0.98, excess +0.28 [-0.39,+1.02] -> "within the bar", but
  there was no established penalty to remove, so never write "mixed calibration fixes night".
  LATENCY (runs/m4/latency.json, ms median model time): torch FP32 6.19, torch "fp16" 6.53 (quantize=16
  on .pt probably not real half -> verify or drop row), TRT FP16 1.47 (4.2x), INT8 day 2.18, mixed 2.20
  -> INT8 SLOWER than FP16 here; likely the 13 FP16 islands' reformat layers (unproven, say "likely").
  Gotcha: a `pgrep -f "<cmd>"` wait loop matches its own bash command line -> use `pgrep -f "[x]..."`.
- **2026-10-20 ~11:10 INT8 BUILD BLOCKED (TRT 10.16 bug)**: fp16.engine built OK (runs/m4/fp16.engine).
  Every INT8 build fails: `MyelinCheckException utils.cpp:4054 CHECK(0)` -> `Could not find any
  implementation for node {ForeignNode[Quantize 0...output0_castOut]}`. Calibration itself runs fine
  (MINMAX, Ultralytics' calibrator). NOT the head pins: fails with opt level 2, with only the FP32
  sigmoid pin, and with no pins at all (m4_diag.py, runs/m4_diag_*.log). Tiny conv / attention nets
  build INT8 fine (tools/trt_diag/tiny_int8.py). Bisect (tools/trt_diag/trt_cut.py, trt_layer.py):
  blocks 0-5 OK, fails from block 6 (C3k2 c3k=True), exactly at the 1x1 conv+SiLU after a Concat
  (m.0/cv3, cv2). Extra network outputs to split Myelin regions: still fails. Pinning those
  post-concat convs to FP16: 2 or 4 pinned still fails; all 13 `(m.0/cv3|cv2)` convs (13/88 = 15%)
  -> builds (tools/trt_diag/trt_pin.py). No public report of this error found. ASKED USER: workaround
  (13 convs FP16, report as deviation) vs try another TRT 10.x (~4 GB) vs skip INT8.
- **2026-10-20 TRT INSTALLED + TEST 5 REFERENCE FIXED BEFORE ANY ENGINE RESULT**: tensorrt-cu12
  10.16.1.11 in .venv-vision (only 3 new pkgs, torch untouched; env diff checked). Test 5 "INT8 loss"
  = B_s0 torch (runs/eval/B_s0__*) minus INT8 engine, per split, paired bootstrap (primary, literal
  reading). Sensitivity: FP16 engine minus INT8 engine (same static 640x640 input, isolates the
  quantisation itself from the .pt letterbox-shape difference). Rule: night loss - day loss > 0.5
  -> PASS; then same for int8_mixed ("removes it" = mixed night-day loss diff <= 0.5).
- **2026-10-19 22:30 M4 PREP**: `m4.py` written (prep/export/bench/eval); `m4.py prep` DONE:
  calib_day = day_val 300 (sha 886920815866), calib_mixed = 150 day_val + 150 night_val
  (sha 35182661dd50), 0 overlap with any test split. Engines = B_s0, static batch 1, 640.
  **BLOCKED on user OK for TensorRT install (~4.3 GB wheel)**. Not in env yet. Options:
  TRT 10.16.1.11 (2026-04, classic IInt8 entropy calibrator, no extra deps; recommended) vs
  TRT 11.x (strongly typed, calibrator removed, ultralytics routes INT8 via nvidia-modelopt Q/DQ
  = extra deps that may try to touch torch). GPU = sm_120 (needs TRT >= 10.8). Ultralytics 8.4.182
  uses `quantize=16/8`, not half/int8. Test 5 judging not yet in compare.py.
- **2026-10-19 17:35**: `JOBS="A_s1 B_s1 C_s1" make start` RUNNING (~1.5 h). `compare.py` written:
  judges tests 1-4 from runs/eval/*.npz (paired bootstrap = same seed -> same image indices per
  split; caches runs/eval/boot/; writes runs/compare.json; skips tests whose runs are missing).
  Test 2 rule as coded: B-C (mean over seeds) must exceed the larger of |B_s0-B_s1|, |C_s0-C_s1|,
  else TIE. Seed-0 compare (23 min with training running; cached now): test 1 PASS, A day-night
  +12.37 [+9.50, +14.50] (zero-shot COCO +11.75 [+8.82, +13.80]); night B0-C0 +2.47 [+1.30, +3.81]
  paired; day B0-C0 -0.48 [-1.18, +0.22] (test 3 PASS on s0 only, re-judge with s1). "Mean of
  empty slice" warning = resamples with no `train` gt, harmless (nanmean).
  **M3 first pass (`analyze_m3.py` -> runs/m3.json)**: size mix day vs night is ~identical
  (small 55.8/54.8%, medium 31.8/32.2%, large 12.4/13.0%) -> size does NOT explain the drop; A_s0
  gap inside every bucket (small +5.4, medium +16.1, large +22.4, all CIs > 0). Night has fewer
  boxes/img (16.4 vs 20.6; sparser labels? see traps). Traffic light worst hit (22.7 -> 9.3, B
  recovers to 14.1); traffic sign barely drops (28.2 -> 26.5). **CAVEAT FOUND**: day_test has 0
  `train` gt (AP nan, dropped from mean) but night_test has 2 (AP 0, counted) -> night mAP is a
  10-class mean with a 0, day a 9-class mean. A_s0 night 9-class = 16.60 vs 14.94 headline; test 1
  gap ~10.7 not 12.4. B vs C on the same split is unaffected (both score train 0). Asked user
  whether to add a 9-class sensitivity line (pre-registered number stays primary). After s1 finishes: `python compare.py`, then the dose jobs (~2.5 h, daytime only).
- **>>> NEXT SESSION START HERE (written 2026-10-18 ~21:30) <<<**
  Training queue RUNNING (`make start`, started 2026-10-18 ~21:25): A_s0 B_s0 C_s0 A_s1 B_s1 C_s1
  dose100/250/1000/2000_s0, each trained (40 ep, last epoch scored) then evaluated on
  day/night/dusk test -> `runs/eval/<run>__<split>.json`. ~1 h per arm, ~10 h total.
  `make status` = progress; `make start` again resumes after a crash/sleep (finished runs and
  evals are skipped, interrupted runs resume from last.pt). Then: write `compare.py` (paired
  bootstrap A vs B vs C, seed spread) and judge tests 1-4 exactly as written below.
- **2026-10-18 23:40: queue CUT SHORT on purpose** (user does not run the laptop overnight).
  Only A_s0, B_s0, C_s0 run tonight (C finishes ~00:15). Still to run, by day:
  `JOBS="A_s1 B_s1 C_s1" make start` (~1.5 h, needed for test 2's seed spread), then
  `JOBS="dose100_s0 dose250_s0 dose1000_s0 dose2000_s0" make start` (~2.5 h). Runs take ~27-33
  min each, not 1 h. So far (mAP50-95, 10 classes): A day 27.3 / night 14.9; B day 27.4 /
  night 17.9. C (00:30, done) day 27.9 / dusk 21.9 / night 15.4. First look: B-C night +2.5,
  C-B day +0.5. NOT judged: needs compare.py (paired bootstrap) + the s1 seeds for test 2.
  GPU idle, queue finished; nothing running.
- **M0 DONE 2026-10-18**: selftest 20/20 (splits, leakage, labels, box counts), selftest_eval
  8/8 (our mAP == pycocotools to 1e-6 on day+night, all 4 size ranges). Images verified:
  10000/10000 sizes match the HF listing; samples.json sha256 = HF LFS oid. Timing: 1 epoch of
  arm A = ~1.4 min incl. val, 4.2 GB VRAM -> 40 epochs chosen before any trained result.
- **M1(a) zero-shot COCO yolo11s, 8 shared classes** (mAP50-95 [95% CI], mAP50):
  day 23.78 [22.39, 25.45] / 39.10; night 12.03 [10.81, 14.64] / 21.71; dusk 17.63 / 29.91.
  Night halves zero-shot mAP before any of our training. (CI skew = rare classes, e.g. train.)
- **2026-10-18**: project started. Shared env `../.venv-vision` (py3.12, torch 2.11+cu128,
  ultralytics 8.4.182; `.venv` here is a symlink to it, ~6 GB on disk). Dataset = BDD100K
  official val 10k via HF mirror `dgural/bdd100k` pinned at `c2e7f266`, 0.70 GB, into
  `data/raw/`. Tests below written BEFORE any training.

## Question
A YOLO detector trained only on daytime driving images: how much worse is it at night, and
does a little night data fix it **beyond what the same amount of extra day data would do**?
Then: how fast can it run with TensorRT, and does INT8 cost more accuracy at night than by day?

## Data (fixed before any result)
BDD100K val 10k (1280x720, every image from a different video, so no clip leakage).
Counts: daytime 5258, night 3929, dawn/dusk 778, undefined 35 (dropped).
10 classes (BDD detection set): pedestrian, rider, car, truck, bus, train, motorcycle,
bicycle, traffic light, traffic sign. `other vehicle` (85), `trailer` (2), `other person` (1)
are dropped. Splits are assigned by a hash of the file name (stable, no RNG state):

| split | from | images | use |
|---|---|---|---|
| day_train | daytime | 3158 | training, every arm |
| day_extra | daytime | 500 | arm C only (equal-count control) |
| day_val | daytime | 300 | monitoring only |
| day_test | daytime | 1300 | headline |
| night_pool | night | 2329 | night added to training (nested subsets) |
| night_val | night | 300 | monitoring only |
| night_test | night | 1300 | headline |
| dusk_test | dawn/dusk | 778 | in-between condition |

**No checkpoint selection**: every arm is scored with its last epoch, so no arm gets to pick
its best epoch on data that favours it.

## Milestones
- **M0 setup**: env, download, `prepare.py` (YOLO layout + split lists), `selftest.py`.
- **M1 baselines**: (a) COCO-pretrained YOLO11 zero-shot on the classes COCO shares with BDD
  (person->pedestrian, car, truck, bus, train, motorcycle, bicycle, traffic light) on day_test
  vs night_test. Tells us whether night is harder for ANY detector, before our training.
  (b) arm A = fine-tune on day_train only. Report day/night/dusk mAP50-95 + mAP50.
- **M2 the fix**: arm B = day_train + 500 night; arm C = day_train + 500 day_extra. Same
  epochs, same seed set. Then a dose curve: day_train + {100, 250, 500, 1000, 2000} night.
- **M3 why**: per-class and per-box-size (small/medium/large) AP, day vs night. Is the night
  drop just "night images have smaller/darker objects"?
- **M4 deploy**: export arm B (or best arm) to TensorRT FP16 and INT8 (calibrated on day
  images only, then on mixed). Latency (batch 1, 640) vs mAP, day and night.

## Pre-registered tests (judged exactly as written, failures reported as failures)
1. **Night gap exists**: arm A night_test mAP50-95 is lower than day_test by more than the
   image-bootstrap 95% CI of the difference. Also report the zero-shot COCO gap for scale.
2. **Night data helps beyond more data**: arm B beats arm C on night_test mAP50-95, and the
   gap (B-C) is larger than the spread between 2 seeds of the same arm. If it is not, the
   verdict is "tie", whatever the sign.
3. **No day cost**: arm B day_test mAP50-95 is within 1.0 point of arm C (night data does not
   hurt day performance).
4. **Dose**: night_test mAP rises with night count; report the count that recovers half of
   the A-to-2000 gain. No pass/fail, descriptive.
5. **INT8 night penalty**: INT8 (day-only calibration) loses more mAP50-95 on night_test than on
   day_test, by more than 0.5 points. Then check whether mixed calibration removes it.

## Known traps
- Night images may simply contain fewer/smaller labelled objects; M3's size split checks this.
- Labels at night may be sparser (annotators miss dark objects): a detector that finds a real
  but unlabelled object is scored as a false positive. Mention when reading night precision.
- `train` has 15 boxes in all of BDD val: its AP is noise. Report it, never headline it.
- 640 px input shrinks 1280x720 by 2x; traffic signs/lights get tiny. Fixed for all arms.
