"""Judge pre-registered tests 1-5 (PLAN.md) from the per-image match data in runs/eval/*.npz.

    python compare.py            # prints the verdicts, writes runs/compare.json

Differences between two runs on the SAME split use a paired image bootstrap: Matches.bootstrap
with a fixed seed draws the same image indices for every run on a split, so subtracting two
bootstrap arrays pairs them. Day vs night (different images) is an unpaired difference.
Bootstrap arrays are cached in runs/eval/boot/. Tests whose runs are missing are skipped.
"""
import json
from pathlib import Path

import numpy as np

import prepare as P
from evaluate import Matches

EVAL = Path(__file__).resolve().parent / "runs" / "eval"
BOOT = EVAL / "boot"
N_BOOT = 1000
DOSES = [0, 100, 250, 500, 1000, 2000]  # 0 = arm A, 500 = arm B
# Sensitivity only, found after pre-registration: day_test has no `train` gt (class dropped from
# the mean) but night_test has 2 (AP 0, counted), so the headline night mean has one extra zero.
NO_TRAIN = [c for c, n in enumerate(P.CLASSES) if n != "train"]


def have(tag, split):
    return (EVAL / f"{tag}__{split}.npz").exists()


def point(tag, split, no_train=False):
    """mAP50-95 in points, from the json evaluate.py wrote (no_train: mean over the 9 other classes)."""
    r = json.loads((EVAL / f"{tag}__{split}.json").read_text())["main"]
    if not no_train:
        return 100 * r["mAP50_95"]
    return 100 * np.nanmean([v for k, v in r["per_class"].items() if k != "train"])


def boot(tag, split, no_train=False):
    """Bootstrap mAP50-95 (points) for one run/split, cached."""
    f = BOOT / f"{tag}__{split}{'__no_train' if no_train else ''}.npy"
    if not f.exists():
        BOOT.mkdir(parents=True, exist_ok=True)
        m = Matches.load(EVAL / f"{tag}__{split}.npz")
        np.save(f, 100 * m.bootstrap(N_BOOT, classes=NO_TRAIN if no_train else None))
    return np.load(f)


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def fmt(d, c):
    return f"{d:+.2f} [{c[0]:+.2f}, {c[1]:+.2f}]"


def test1(out):
    """Night gap of arm A (seed 0), plus the zero-shot COCO gap for scale."""
    for tag in ("A_s0", "coco_yolo11s"):
        if not (have(tag, "day_test") and have(tag, "night_test")):
            continue
        d = point(tag, "day_test") - point(tag, "night_test")
        c = ci(boot(tag, "day_test") - boot(tag, "night_test"))
        out[f"test1_{tag}"] = {"day_minus_night": d, "ci95": c}
        print(f"  {tag:13s} day - night = {fmt(d, c)}")
    r = out.get("test1_A_s0")
    if r:
        out["test1"] = "PASS" if r["ci95"][0] > 0 else "FAIL"
        print(f"  TEST 1: {out['test1']} (CI of the difference excludes 0: {r['ci95'][0] > 0})")


def seeds(arm, split):
    return [t for t in (f"{arm}_s0", f"{arm}_s1") if have(t, split)]


def test2_3(out):
    """B vs C on night (test 2, needs 2 seeds each) and on day (test 3)."""
    for split, name in (("night_test", "test2"), ("day_test", "test3")):
        b, c = seeds("B", split), seeds("C", split)
        if not (b and c):
            continue
        for s in ("s0", "s1"):
            if f"B_{s}" in b and f"C_{s}" in c:
                d = point(f"B_{s}", split) - point(f"C_{s}", split)
                dc = ci(boot(f"B_{s}", split) - boot(f"C_{s}", split))
                print(f"  {split} B_{s} - C_{s} = {fmt(d, dc)} (paired bootstrap)")
        gap = np.mean([point(t, split) for t in b]) - np.mean([point(t, split) for t in c])
        r = {"B_minus_C_seed_mean": gap, "seeds_B": len(b), "seeds_C": len(c)}
        if name == "test2":
            # "spread between 2 seeds of the same arm": |s0 - s1| per arm; the gap must beat the larger one
            spread = {arm: abs(point(t[0], split) - point(t[1], split))
                      for arm, t in (("A", seeds("A", split)), ("B", b), ("C", c)) if len(t) == 2}
            r["seed_spread"] = spread
            print(f"  {split} B - C (mean over seeds) = {gap:+.2f}; seed spread |s0-s1| = "
                  + ", ".join(f"{k} {v:.2f}" for k, v in spread.items()))
            if "B" in spread and "C" in spread:
                worst = max(spread["B"], spread["C"])
                r["verdict"] = "PASS" if gap > worst else "TIE"
                print(f"  TEST 2: {r['verdict']} (B - C {gap:+.2f} vs larger B/C seed spread {worst:.2f})")
            else:
                print("  TEST 2: not judged yet (needs B_s1 and C_s1)")
        else:
            r["verdict"] = "PASS" if abs(gap) <= 1.0 else "FAIL"
            print(f"  {split} B - C (mean over seeds) = {gap:+.2f}")
            print(f"  TEST 3: {r['verdict']} (|B - C| on day within 1.0 point)")
        out[name] = r


def sensitivity(out):
    """Tests 1-3 recomputed without `train` on both splits. Not a verdict; the headline stays as registered."""
    r = {}
    if have("A_s0", "day_test") and have("A_s0", "night_test"):
        d = point("A_s0", "day_test", True) - point("A_s0", "night_test", True)
        c = ci(boot("A_s0", "day_test", True) - boot("A_s0", "night_test", True))
        r["A_s0_day_minus_night"] = {"gap": d, "ci95": c}
        print(f"  A_s0 day - night = {fmt(d, c)}")
    for split in ("night_test", "day_test"):
        for s in ("s0", "s1"):
            if have(f"B_{s}", split) and have(f"C_{s}", split):
                d = point(f"B_{s}", split, True) - point(f"C_{s}", split, True)
                c = ci(boot(f"B_{s}", split, True) - boot(f"C_{s}", split, True))
                r[f"{split}_B_{s}_minus_C_{s}"] = {"gap": d, "ci95": c}
                print(f"  {split} B_{s} - C_{s} = {fmt(d, c)} (paired)")
    out["sensitivity_no_train"] = r


def test4(out):
    """Night mAP vs number of night images (seed 0), descriptive."""
    tag = {0: "A_s0", 500: "B_s0"}
    pts = {n: point(tag.get(n, f"dose{n}_s0"), "night_test") for n in DOSES
           if have(tag.get(n, f"dose{n}_s0"), "night_test")}
    if len(pts) < 2:
        return
    print("  night mAP50-95 by night images: " + ", ".join(f"{n}: {v:.2f}" for n, v in pts.items()))
    r = {"night_map_by_count": pts}
    if 0 in pts and 2000 in pts:
        half = pts[0] + (pts[2000] - pts[0]) / 2
        r["half_gain_target"] = half
        r["first_count_reaching_half"] = next((n for n in sorted(pts) if pts[n] >= half), None)
        print(f"  TEST 4: half of the A-to-2000 gain = {half:.2f}, first reached at "
              f"{r['first_count_reaching_half']} night images")
    else:
        print("  TEST 4: not judged yet (needs the dose runs)")
    out["test4"] = r


def test5(out):
    """INT8 loss on night vs day (B_s0 engines). Loss = reference - INT8, paired per split; the
    night-minus-day difference of losses is unpaired across splits, like test 1. Reference fixed in
    PLAN.md before any engine result: B_s0 torch (primary), FP16 engine (sensitivity)."""
    refs = {"torch": "B_s0", "fp16_engine": "B_s0_fp16"}
    r = {}
    for calib in ("int8_day", "int8_mixed"):
        q = f"B_s0_{calib}"
        for ref_name, ref in refs.items():
            if not all(have(t, s) for t in (q, ref) for s in ("day_test", "night_test")):
                continue
            loss = {s: point(ref, s) - point(q, s) for s in ("day_test", "night_test")}
            bl = {s: boot(ref, s) - boot(q, s) for s in ("day_test", "night_test")}
            d = loss["night_test"] - loss["day_test"]
            c = ci(bl["night_test"] - bl["day_test"])
            r[f"{calib}_vs_{ref_name}"] = {"loss_day": loss["day_test"], "loss_night": loss["night_test"],
                                           "loss_day_ci95": ci(bl["day_test"]), "loss_night_ci95": ci(bl["night_test"]),
                                           "night_minus_day_loss": d, "ci95": c}
            print(f"  {calib} vs {ref_name:11s} loss day {fmt(loss['day_test'], ci(bl['day_test']))}, "
                  f"night {fmt(loss['night_test'], ci(bl['night_test']))}; night - day {fmt(d, c)}")
    p = r.get("int8_day_vs_torch")
    if p:
        r["verdict"] = "PASS" if p["night_minus_day_loss"] > 0.5 else "FAIL"
        print(f"  TEST 5: {r['verdict']} (day-calibrated INT8 night loss - day loss "
              f"{p['night_minus_day_loss']:+.2f} vs 0.5 bar)")
        m = r.get("int8_mixed_vs_torch")
        if m:
            r["mixed_removes_it"] = m["night_minus_day_loss"] <= 0.5
            print(f"  TEST 5 follow-up: mixed calibration night - day loss {m['night_minus_day_loss']:+.2f} "
                  f"-> {'within' if r['mixed_removes_it'] else 'still above'} the 0.5 bar")
    else:
        print("  TEST 5: not judged yet (needs B_s0_int8_day and B_s0 evals)")
    out["test5"] = r


def main():
    out = {}
    print("Test 1: night gap (arm A, seed 0)")
    test1(out)
    print("Tests 2-3: night data vs equal extra day data")
    test2_3(out)
    print("Sensitivity: 9 classes, `train` dropped on both splits (not a verdict)")
    sensitivity(out)
    print("Test 4: dose")
    test4(out)
    print("Test 5: INT8 night penalty (B_s0 engines)")
    test5(out)
    (EVAL.parent / "compare.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
