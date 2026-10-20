"""Checks on the prepared data. Run after prepare.py and after any change to it.

Fails loudly on: wrong box conversion, split overlap or wrong counts, a test image inside any
training arm, a split list that drifted from the manifest, missing or malformed labels.
"""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import prepare as P

checks = []


def check(name, ok, detail=""):
    """Record one named check and print it."""
    checks.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail and not ok else ''}")


def read_list(path):
    """Lines of a split list, as a list of strings."""
    return Path(path).read_text().split()


def arm_train_lists(arm):
    """Image paths an arm yaml trains on (parses the simple yaml prepare.py writes)."""
    text = (P.ARMS / f"{arm}.yaml").read_text().splitlines()
    i = text.index("train:")
    paths = []
    for line in text[i + 1:]:
        if not line.startswith("  - "):
            break
        paths += read_list(line[4:])
    return paths


def main():
    """Run every check; exit 1 if any fails."""
    # 1. box conversion on hand-made boxes
    lines, drop = P.to_yolo_lines([
        {"label": "car", "bounding_box": [0.1, 0.2, 0.3, 0.4]},
        {"label": "traffic sign", "bounding_box": [0.9, -0.1, 0.2, 0.3]},  # spills off two edges
        {"label": "other vehicle", "bounding_box": [0.1, 0.1, 0.1, 0.1]},
        {"label": "pedestrian", "bounding_box": [0.5, 0.5, 0.0, 0.1]},
    ])
    check("box: top-left xywh -> centre xywh", lines[0] == "2 0.250000 0.400000 0.300000 0.400000", lines[0])
    check("box: clipped to the image", lines[1] == "9 0.950000 0.100000 0.100000 0.200000", lines[1])
    check("box: unknown class and zero-width dropped",
          len(lines) == 2 and drop == Counter({"other vehicle": 1, "degenerate": 1}), str(drop))

    samples = json.loads((P.RAW / "samples.json").read_text())["samples"]
    tod = {Path(s["filepath"]).stem: s["timeofday"]["label"] for s in samples}

    # 2. splits are deterministic and independent of input order
    shuffled = samples[:]
    random.Random(0).shuffle(shuffled)
    a, b = P.assign_splits(samples), P.assign_splits(shuffled)
    check("splits independent of input order",
          all([s["filepath"] for s in a[k]] == [s["filepath"] for s in b[k]] for k in a))

    # 3. split lists: counts, time of day, disjoint, manifest hashes
    man = json.loads((P.ROOT / "data" / "manifest.json").read_text())
    want = {"day_test": 1300, "day_val": 300, "day_extra": 500, "day_train": 3158,
            "night_test": 1300, "night_val": 300, "night_pool": 2329, "dusk_test": 778}
    lists = {k: read_list(P.SPLITS / f"{k}.txt") for k in want}
    check("split counts as planned", {k: len(v) for k, v in lists.items()} == want,
          str({k: len(v) for k, v in lists.items()}))
    expect_tod = {"day": "daytime", "night": "night", "dusk": "dawn/dusk"}
    check("every split holds only its time of day",
          all(tod[Path(p).stem] == expect_tod[k.split("_")[0]] for k, v in lists.items() for p in v))
    seen = Counter(p for v in lists.values() for p in v)
    check("base splits are disjoint", max(seen.values()) == 1)
    drift = [k for k, v in man["splits"].items()
             if hashlib.sha256((P.SPLITS / f"{k}.txt").read_bytes()).hexdigest() != v["sha256"]]
    check("split lists match manifest sha256", not drift, str(drift))

    # 4. nested night doses
    pool = lists["night_pool"]
    doses = [read_list(P.SPLITS / f"night_{n}.txt") for n in P.NIGHT_DOSES]
    check("night doses are nested prefixes of night_pool",
          all(d == pool[:n] for d, n in zip(doses, P.NIGHT_DOSES)))

    # 5. no test or val image in any training arm; arms have the planned sizes
    held_out = set(lists["day_test"] + lists["night_test"] + lists["dusk_test"]
                   + lists["day_val"] + lists["night_val"])
    sizes = {}
    for arm in ["A", "B", "C"] + [f"dose{n}" for n in P.NIGHT_DOSES]:
        tr = arm_train_lists(arm)
        sizes[arm] = len(tr)
        check(f"arm {arm}: no held-out image in training", not (set(tr) & held_out))
    check("arm sizes: B == C == A + 500",
          sizes["B"] == sizes["C"] == sizes["A"] + 500 == 3658, str(sizes))

    # 6. labels: every listed image has a parseable label file with valid class ids
    bad, n_boxes, per_class = [], 0, Counter()
    for p in set(p for v in lists.values() for p in v):
        lab = Path(p.replace("/images/", "/labels/")).with_suffix(".txt")
        if not Path(p).is_file() or not lab.is_file():
            bad.append(p)
            continue
        for line in lab.read_text().splitlines():
            c, *xywh = line.split()
            v = [float(x) for x in xywh]
            if not (0 <= int(c) < len(P.CLASSES) and len(v) == 4 and all(0 <= x <= 1 for x in v)):
                bad.append(p)
            per_class[P.CLASSES[int(c)]] += 1
            n_boxes += 1
    check("every image has a valid label file", not bad, f"{len(bad)} bad, e.g. {bad[:2]}")

    # 7. box count matches the source for the 10 kept classes
    src = Counter(d["label"] for s in samples if s["timeofday"]["label"] in ("daytime", "night", "dawn/dusk")
                  for d in (s.get("detections") or {}).get("detections", []) if d["label"] in P.CLASS_ID)
    check("box counts per class match the source", per_class == src,
          str({c: (per_class[c], src[c]) for c in P.CLASSES if per_class[c] != src[c]}))

    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
