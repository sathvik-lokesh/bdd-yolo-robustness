"""Check evaluate.py's mAP against pycocotools on real predictions, per area range.

Runs the zero-shot COCO model on the first 200 day_test and 200 night_test images, scores the
same predictions with both implementations and requires agreement to 1e-6.
"""
import contextlib
import io
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import evaluate as E
import prepare as P


def coco_reference(images, preds, classes):
    """mAP50-95 per area range from pycocotools on the same predictions."""
    gt = {"images": [], "annotations": [], "categories": [{"id": c} for c in classes]}
    dt = []
    for i, img in enumerate(images):
        gt["images"].append({"id": i, "width": E.W, "height": E.H})
        gc, gb = E.load_gt(img)
        for c, b in zip(gc, gb):
            if c in classes:
                w, h = b[2] - b[0], b[3] - b[1]
                gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": i, "category_id": int(c),
                                          "bbox": [b[0], b[1], w, h], "area": w * h, "iscrowd": 0})
        pc, pb, ps = preds[img]
        for c, b, s in zip(pc, pb, ps):
            dt.append({"image_id": i, "category_id": int(c), "score": float(s),
                       "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = gt
        g.createIndex()
        ev = COCOeval(g, g.loadRes(dt), "bbox")
        ev.params.areaRng = [list(E.AREAS[a]) for a in ("all", "small", "medium", "large")]
        ev.evaluate()
        ev.accumulate()
    out = {}
    for k, a in enumerate(("all", "small", "medium", "large")):
        p = ev.eval["precision"][:, :, :, k, -1]  # T, R, K at maxDets=100
        per = [p[:, :, j] for j in range(p.shape[2])]
        per = [x.mean() for x in per if (x > -1).all()]
        out[a] = float(np.mean(per))
    return out


def main():
    """Compare both implementations; exit 1 on any mismatch."""
    ok = True
    for split in ("day_test", "night_test"):
        images = (P.SPLITS / f"{split}.txt").read_text().split()[:200]
        preds = E.predict(str(P.ROOT / "weights" / "yolo11s.pt"), images, coco=True)
        m = E.build(images, preds, E.SHARED)
        ref = coco_reference(images, preds, E.SHARED)
        for a in ("all", "small", "medium", "large"):
            mine = m.map(area=a)[0]
            good = abs(mine - ref[a]) < 1e-6
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  {split} {a:6s} ours {mine:.6f}  pycocotools {ref[a]:.6f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
