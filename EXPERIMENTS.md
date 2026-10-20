# Experiment design

The tests, splits and decisions for this project, written before the results they apply to. Results are in [README.md](README.md).

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

## Setup checks
- selftest 20/20 (splits, leakage, labels, box counts), selftest_eval
8/8 (our mAP == pycocotools to 1e-6 on day+night, all 4 size ranges). Images verified:
10000/10000 sizes match the HF listing; samples.json sha256 = HF LFS oid. Timing: 1 epoch of
arm A = ~1.4 min incl. val, 4.2 GB VRAM -> 40 epochs chosen before any trained result.

## Test 5 and INT8 setup, fixed before any engine result
- tensorrt-cu12
10.16.1.11 in .venv-vision (only 3 new pkgs, torch untouched; env diff checked). Test 5 "INT8 loss"
= B_s0 torch (runs/eval/B_s0__*) minus INT8 engine, per split, paired bootstrap (primary, literal
reading). Sensitivity: FP16 engine minus INT8 engine (same static 640x640 input, isolates the
quantisation itself from the .pt letterbox-shape difference). Rule: night loss - day loss > 0.5
-> PASS; then same for int8_mixed ("removes it" = mixed night-day loss diff <= 0.5).
- `m4.py prep`: calib_day = day_val 300 (sha 886920815866), calib_mixed = 150 day_val + 150 night_val
  (sha 35182661dd50), 0 overlap with any test split. Engines = B_s0, static batch 1, 640.

## TensorRT 10.16 INT8 build failure
Every INT8 build fails: `MyelinCheckException utils.cpp:4054 CHECK(0)` -> `Could not find any
implementation for node {ForeignNode[Quantize 0...output0_castOut]}`. Calibration itself runs fine
(MINMAX, Ultralytics' calibrator). NOT the head pins: fails with opt level 2, with only the FP32
sigmoid pin, and with no pins at all (m4_diag.py, runs/m4_diag_*.log). Tiny conv / attention nets
build INT8 fine (tools/trt_diag/tiny_int8.py). Bisect (tools/trt_diag/trt_cut.py, trt_layer.py):
blocks 0-5 OK, fails from block 6 (C3k2 c3k=True), exactly at the 1x1 conv+SiLU after a Concat
(m.0/cv3, cv2). Extra network outputs to split Myelin regions: still fails. Pinning those
post-concat convs to FP16: 2 or 4 pinned still fails; all 13 `(m.0/cv3|cv2)` convs (13/88 = 15%)
-> builds (tools/trt_diag/trt_pin.py). No public report of this error found.
Chosen: the 13 convs stay FP16 (`FP16_PINS` in `m4.py`), reported as a deviation.
