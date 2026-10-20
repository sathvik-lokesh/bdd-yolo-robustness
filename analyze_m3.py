"""M3: why is night worse? Per-size and per-class AP, day vs night, plus the ground-truth mix.

    python analyze_m3.py         # prints tables, writes runs/m3.json

Known trap (PLAN.md): night images may just hold smaller objects. If that were the whole story,
the day-night gap would shrink to ~0 inside each COCO size bucket. The per-size gaps of arm A
get image-bootstrap CIs (cached in runs/eval/boot/); everything else is point estimates.
"""
import json
from pathlib import Path

import numpy as np

import prepare as P
from evaluate import AREAS, Matches, load_gt

EVAL = P.ROOT / "runs" / "eval"
BOOT = EVAL / "boot"
SPLITS = ("day_test", "night_test", "dusk_test")
SIZES = ("small", "medium", "large")
RUNS = ("coco_yolo11s", "A_s0", "B_s0", "C_s0", "A_s1", "B_s1", "C_s1")


def summ(tag, split):
    f = EVAL / f"{tag}__{split}.json"
    return json.loads(f.read_text())["main"] if f.exists() else None


def gt_mix(split):
    """Box counts per size bucket and per class, plus boxes per image."""
    imgs = (P.SPLITS / f"{split}.txt").read_text().split()
    size = dict.fromkeys(SIZES, 0)
    cls = np.zeros(len(P.CLASSES), int)
    for img in imgs:
        c, b = load_gt(img)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        for s in SIZES:
            lo, hi = AREAS[s]
            size[s] += int(((area >= lo) & (area < hi)).sum())
        cls += np.bincount(c, minlength=len(P.CLASSES))
    n = sum(size.values())
    return {"images": len(imgs), "boxes_per_image": n / len(imgs),
            "size_share": {s: size[s] / n for s in SIZES}, "class_count": dict(zip(P.CLASSES, cls.tolist()))}


def boot_size(tag, split, area):
    f = BOOT / f"{tag}__{split}__{area}.npy"
    if not f.exists():
        BOOT.mkdir(parents=True, exist_ok=True)
        np.save(f, 100 * Matches.load(EVAL / f"{tag}__{split}.npz").bootstrap(1000, area=area))
    return np.load(f)


def main():
    out = {"gt": {s: gt_mix(s) for s in SPLITS}}
    print("Ground truth mix")
    print(f"  {'split':11s} boxes/img  small  medium  large")
    for s, g in out["gt"].items():
        sh = g["size_share"]
        print(f"  {s:11s} {g['boxes_per_image']:8.2f}  {sh['small']:5.1%}  {sh['medium']:6.1%}  {sh['large']:5.1%}")

    print("\nmAP50-95 by box size (points)")
    out["per_size"] = {}
    for tag in RUNS:
        row = {s: summ(tag, s) for s in SPLITS}
        if not row["day_test"]:
            continue
        out["per_size"][tag] = {s: {k: 100 * v for k, v in r["per_size"].items()} for s, r in row.items() if r}
        p = out["per_size"][tag]
        print(f"  {tag:13s} " + "  ".join(f"{z}: day {p['day_test'][z]:5.2f} night {p['night_test'][z]:5.2f}"
                                          for z in SIZES))

    print("\nArm A_s0 day - night inside each size bucket (95% CI, unpaired image bootstrap)")
    out["A_gap_by_size"] = {}
    for z in SIZES:
        d = out["per_size"]["A_s0"]["day_test"][z] - out["per_size"]["A_s0"]["night_test"][z]
        b = boot_size("A_s0", "day_test", z) - boot_size("A_s0", "night_test", z)
        b = b[~np.isnan(b)]
        c = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        out["A_gap_by_size"][z] = {"gap": d, "ci95": c}
        print(f"  {z:7s} {d:+6.2f} [{c[0]:+.2f}, {c[1]:+.2f}]")

    print("\nPer-class AP50-95, arm A_s0 / B_s0 (day -> night), with night gt boxes")
    out["per_class"] = {}
    for tag in ("A_s0", "B_s0"):
        dy, nt = summ(tag, "day_test")["per_class"], summ(tag, "night_test")["per_class"]
        out["per_class"][tag] = {c: {"day": 100 * dy[c], "night": 100 * nt[c]} for c in P.CLASSES}
    for c in P.CLASSES:
        a, b = out["per_class"]["A_s0"][c], out["per_class"]["B_s0"][c]
        n = out["gt"]["night_test"]["class_count"][c]
        print(f"  {c:14s} n={n:5d}  A {a['day']:5.2f} -> {a['night']:5.2f}   B {b['day']:5.2f} -> {b['night']:5.2f}")
    (P.ROOT / "runs" / "m3.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
