"""Fine-tune a COCO-pretrained YOLO11 on one training arm.

Every arm uses the same model, epochs, image size, batch and augmentation; only the training
list changes (data/arms/<arm>.yaml). Output: runs/<arm>_s<seed>/weights/last.pt.

    python train.py --arm A --seed 0
    python train.py --arm A --seed 0 --resume   # continue an interrupted run
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# Fixed for every arm (see PLAN.md). Changing any of these means re-running every arm.
HP = dict(model="yolo11s.pt", epochs=40, imgsz=640, batch=16, workers=6, cos_lr=True,
          close_mosaic=5, amp=True, plots=True)


def main():
    """Train one arm (or resume it) and print where the weights went."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=HP["epochs"], help="only for smoke runs")
    ap.add_argument("--fraction", type=float, default=1.0, help="only for smoke runs")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    name = f"{a.arm}_s{a.seed}" + ("" if a.epochs == HP["epochs"] and a.fraction == 1.0 else "_smoke")
    run = ROOT / "runs" / name
    if a.resume:
        YOLO(run / "weights" / "last.pt").train(resume=True)
    else:
        hp = {k: v for k, v in HP.items() if k != "model"} | dict(epochs=a.epochs, fraction=a.fraction)
        YOLO(ROOT / "weights" / HP["model"]).train(
            data=str(ROOT / "data" / "arms" / f"{a.arm}.yaml"), project=str(ROOT / "runs"),
            name=name, exist_ok=True, seed=a.seed, deterministic=True, **hp)
    (run / "DONE").write_text("finished\n")  # queue.sh skips runs that have this
    print(f"done: {run / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
