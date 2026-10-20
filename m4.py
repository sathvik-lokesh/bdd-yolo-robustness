"""M4 deploy: TensorRT FP16 / INT8 engines of arm B_s0, latency and day/night mAP (test 5).

    python m4.py prep      # calibration lists + yamls (no GPU)
    python m4.py export    # FP16, INT8 day-calibrated, INT8 mixed-calibrated -> runs/m4/*.engine
    python m4.py bench     # batch-1 640 latency, PyTorch FP32/FP16 vs each engine -> runs/m4/latency.json
    python m4.py eval      # evaluate.py on day/night/dusk for each engine (tags B_s0_<variant>)

Calibration uses monitoring splits only, never a test image, and both sets hold 300 images:
day = day_val (300); mixed = first 150 of day_val + first 150 of night_val.
Test 5 is then judged in compare.py from the eval files (INT8 night loss vs day loss, > 0.5 points).

DEVIATION (2026-10-20): TensorRT 10.16 cannot build ANY INT8 engine of this network (Myelin CHECK(0) at the
1x1 conv after each C2f/C3k Concat; bisect in tools/trt_diag/, notes in PLAN.md). Workaround chosen by the
user: the INT8 engines keep those 13 convs (`(m.0/cv3|cv2)/conv/Conv`, 13 of 88) in FP16, on top of
Ultralytics' own head pins. So "INT8" here means 75 of 88 convs in INT8.
"""
import json
import re
import shutil
import subprocess
import sys
import time

import numpy as np

import prepare as P

WEIGHTS = P.ROOT / "runs" / "B_s0" / "weights" / "last.pt"
OUT = P.ROOT / "runs" / "m4"
ARMS = P.ROOT / "data" / "arms"
VARIANTS = {"fp16": {"quantize": 16}, "int8_day": {"quantize": 8, "calib": "calib_day"},
            "int8_mixed": {"quantize": 8, "calib": "calib_mixed"}}
SPLITS = ("day_test", "night_test", "dusk_test")
FP16_PINS = re.compile(r"^/model\.\d+/(m\.0/cv3|cv2)/conv/Conv$")  # see DEVIATION above
N_PINS = 13


def pin_post_concat_convs():
    """Wrap TensorRT's builder so INT8 builds keep the FP16_PINS convs in FP16 (the 10.16 Myelin workaround)."""
    import tensorrt as trt
    orig = trt.Builder.build_serialized_network

    def build(self, network, config):
        if config.get_flag(trt.BuilderFlag.INT8):
            n = 0
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if FP16_PINS.search(layer.name):
                    layer.precision = trt.float16
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, trt.float16)
                    n += 1
            assert n == N_PINS, f"expected {N_PINS} pinned convs, got {n}"
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            print(f"m4: pinned {n} post-concat convs to FP16 (TRT 10.16 INT8 workaround)")
        return orig(self, network, config)

    trt.Builder.build_serialized_network = build


def prep():
    """Write the two 300-image calibration lists and an Ultralytics yaml for each ('val' = calib set)."""
    day = (P.SPLITS / "day_val.txt").read_text().split()
    night = (P.SPLITS / "night_val.txt").read_text().split()
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(P.CLASSES))
    for name, imgs in (("calib_day", day), ("calib_mixed", day[:150] + night[:150])):
        sha = P.write_list(P.SPLITS / f"{name}.txt", imgs)
        (ARMS / f"{name}.yaml").write_text(
            f"path: {P.ROOT / 'data'}\ntrain: {P.SPLITS / 'day_train.txt'}\nval: {P.SPLITS / name}.txt\n"
            f"names:\n{names}\n")
        print(f"{name}: {len(imgs)} images, sha256 {sha[:12]}")


def export():
    """Build each engine (static batch 1, 640) and move it out of the weights folder so none overwrites another."""
    from ultralytics import YOLO
    pin_post_concat_convs()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, v in VARIANTS.items():
        dst = OUT / f"{name}.engine"
        if dst.exists():
            print(f"{name}: exists, skipped")
            continue
        kw = {"data": str(ARMS / f"{v['calib']}.yaml"), "fraction": 1.0} if "calib" in v else {}
        f = YOLO(str(WEIGHTS)).export(format="engine", imgsz=640, batch=1, quantize=v["quantize"], **kw)
        shutil.move(f, dst)
        print(f"{name}: {dst}")


def bench(n=300, warmup=50):
    """Median / p90 model latency (ms, Ultralytics' 'inference' timer) on one fixed day_test image."""
    from ultralytics import YOLO
    img = (P.SPLITS / "day_test.txt").read_text().split()[0]
    models = {"torch_fp32": (str(WEIGHTS), {}), "torch_fp16": (str(WEIGHTS), {"quantize": 16})}
    models |= {k: (str(OUT / f"{k}.engine"), {}) for k in VARIANTS if (OUT / f"{k}.engine").exists()}
    res = {}
    for name, (w, kw) in models.items():
        m = YOLO(w, task="detect")
        for _ in range(warmup):
            m.predict(img, imgsz=640, verbose=False, **kw)
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = m.predict(img, imgsz=640, verbose=False, **kw)[0]
            t.append((r.speed["inference"], 1000 * (time.perf_counter() - t0)))
        t = np.array(t)
        res[name] = {"inference_ms_median": float(np.median(t[:, 0])), "inference_ms_p90": float(np.percentile(t[:, 0], 90)),
                     "end_to_end_ms_median": float(np.median(t[:, 1]))}
        print(f"{name:11s} inference {res[name]['inference_ms_median']:6.2f} ms (p90 "
              f"{res[name]['inference_ms_p90']:.2f})  end-to-end {res[name]['end_to_end_ms_median']:6.2f} ms")
    (OUT / "latency.json").write_text(json.dumps(res, indent=1))


def evaluate_all():
    """Score every engine on the three test splits with the same evaluate.py as the trained arms."""
    for name in VARIANTS:
        for split in SPLITS:
            tag = f"B_s0_{name}"
            if (P.ROOT / "runs" / "eval" / f"{tag}__{split}.json").exists():
                continue
            subprocess.run([sys.executable, "-E", "-s", "-u", "evaluate.py", "--weights", str(OUT / f"{name}.engine"),
                            "--split", split, "--tag", tag], check=True, cwd=P.ROOT)


if __name__ == "__main__":
    {"prep": prep, "export": export, "bench": bench, "eval": evaluate_all}[sys.argv[1]]()
