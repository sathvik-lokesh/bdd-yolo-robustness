"""Diagnose the INT8 TensorRT build failure (Myelin CHECK(0) on the precision-constrained head, TRT 10.16, sm_120).

    python m4_diag.py <variant> [calib]   # calib = calib_day (default) or calib_mixed

Runs Ultralytics' own INT8 export unchanged (same ONNX, calibrator, MINMAX algorithm, dataset, metadata) and
changes ONE builder setting just before build_serialized_network, so a working variant differs from the
stock export by that setting only:
    opt2 / opt1 / opt0   builder_optimization_level (default 3); head constraints kept as Ultralytics sets them
    sigmoid_only         drop the FP16 pins on the head output convs + DFL, keep the FP32 head Sigmoid
    none                 drop every head precision pin (plain implicit INT8 with FP16 fallback)
Writes runs/m4/diag_<variant>_<calib>.engine on success.
"""
import shutil
import sys

import tensorrt as trt

import m4

VARIANT = sys.argv[1]
CALIB = sys.argv[2] if len(sys.argv) > 2 else "calib_day"
_orig = trt.Builder.build_serialized_network


def patched(self, network, config):
    """Apply VARIANT to the config / layer precisions, then build as normal."""
    if VARIANT.startswith("opt"):
        config.builder_optimization_level = int(VARIANT[3:])
    elif VARIANT in ("sigmoid_only", "none"):
        n = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if not layer.precision_is_set:
                continue
            if VARIANT == "sigmoid_only" and layer.precision == trt.float32:
                continue
            layer.reset_precision()
            for j in range(layer.num_outputs):
                layer.reset_output_type(j)
            n += 1
        if VARIANT == "none":
            config.clear_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        print(f"[diag] reset precision on {n} layers")
    else:
        raise SystemExit(f"unknown variant {VARIANT}")
    print(f"[diag] variant {VARIANT}, opt level {config.builder_optimization_level}")
    return _orig(self, network, config)


trt.Builder.build_serialized_network = patched

if __name__ == "__main__":
    from ultralytics import YOLO
    m4.OUT.mkdir(parents=True, exist_ok=True)
    f = YOLO(str(m4.WEIGHTS)).export(format="engine", imgsz=640, batch=1, quantize=8,
                                     data=str(m4.ARMS / f"{CALIB}.yaml"), fraction=1.0)
    dst = m4.OUT / f"diag_{VARIANT}_{CALIB}.engine"
    shutil.move(f, dst)
    print(f"[diag] OK {dst}")
