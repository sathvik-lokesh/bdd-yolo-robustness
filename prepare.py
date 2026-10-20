"""Convert the BDD100K val 10k (FiftyOne export) into YOLO format plus fixed split lists.

Reads  data/raw/samples.json + data/raw/data/*.jpg
Writes data/yolo/images/*.jpg   (hard links to the raw files, no extra disk)
       data/yolo/labels/*.txt   (class cx cy w h, normalised)
       data/splits/<split>.txt  (absolute image paths, one per line)
       data/arms/<arm>.yaml     (Ultralytics dataset files, one per training arm)
       data/manifest.json       (split counts + sha256 of every split list)

Splits are decided by sha1(file name), never by RNG state, so they never move.
"""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
YOLO = ROOT / "data" / "yolo"
SPLITS = ROOT / "data" / "splits"
ARMS = ROOT / "data" / "arms"

# BDD100K detection classes, in the order BDD's own evaluation uses.
CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle", "traffic light", "traffic sign"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# (time of day, [(split, count), ...]); the remainder of each group goes to the last split.
PLAN = {
    "daytime": [("day_test", 1300), ("day_val", 300), ("day_extra", 500), ("day_train", None)],
    "night": [("night_test", 1300), ("night_val", 300), ("night_pool", None)],
    "dawn/dusk": [("dusk_test", None)],
}
NIGHT_DOSES = [100, 250, 500, 1000, 2000]


def name_hash(name):
    """Stable sort key for a file name, independent of Python's hash seed."""
    return hashlib.sha1(name.encode()).hexdigest()


def to_yolo_lines(dets):
    """FiftyOne boxes ([x, y, w, h], top-left, normalised) -> YOLO label lines; drops unknown classes."""
    lines, dropped = [], Counter()
    for d in dets:
        cid = CLASS_ID.get(d["label"])
        if cid is None:
            dropped[d["label"]] += 1
            continue
        x, y, w, h = d["bounding_box"]
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(1.0, x + w), min(1.0, y + h)
        if x1 <= x0 or y1 <= y0:
            dropped["degenerate"] += 1
            continue
        lines.append(f"{cid} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
    return lines, dropped


def assign_splits(samples):
    """Group samples by time of day and cut each group, in hash order, into the fixed splits."""
    groups = {}
    for s in samples:
        groups.setdefault(s["timeofday"]["label"], []).append(s)
    splits = {}
    for tod, cuts in PLAN.items():
        items = sorted(groups.get(tod, []), key=lambda s: name_hash(Path(s["filepath"]).name))
        pos = 0
        for split, n in cuts:
            end = len(items) if n is None else pos + n
            assert end <= len(items), f"not enough {tod} images for {split}"
            splits[split] = items[pos:end]
            pos = end
    return splits


def write_list(path, paths):
    """Write one absolute path per line and return the sha256 of the file."""
    text = "".join(f"{p}\n" for p in paths)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def write_arm(name, train_lists):
    """Write an Ultralytics dataset yaml; val is day_val for every arm (monitoring only)."""
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    train = "\n".join(f"  - {SPLITS / t}.txt" for t in train_lists)
    (ARMS / f"{name}.yaml").write_text(
        f"path: {ROOT / 'data'}\ntrain:\n{train}\nval: {SPLITS / 'day_val.txt'}\nnames:\n{names}\n")


def write_eval(split):
    """Dataset yaml that evaluates one test split (Ultralytics scores the 'val' entry)."""
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (ARMS / f"eval_{split}.yaml").write_text(
        f"path: {ROOT / 'data'}\ntrain: {SPLITS / 'day_train.txt'}\nval: {SPLITS / split}.txt\n"
        f"names:\n{names}\n")


def main():
    """Build the YOLO layout, split lists, arm yamls and manifest."""
    samples = json.loads((RAW / "samples.json").read_text())["samples"]
    for d in (YOLO / "images", YOLO / "labels", SPLITS, ARMS):
        d.mkdir(parents=True, exist_ok=True)

    dropped, n_boxes = Counter(), 0
    for s in samples:
        src = RAW / s["filepath"]
        assert src.is_file(), f"missing image {src}"
        name = src.stem
        dst = YOLO / "images" / f"{name}.jpg"
        if not dst.exists():
            os.link(src, dst)
        lines, drop = to_yolo_lines((s.get("detections") or {}).get("detections", []))
        (YOLO / "labels" / f"{name}.txt").write_text("".join(l + "\n" for l in lines))
        dropped.update(drop)
        n_boxes += len(lines)

    splits = assign_splits(samples)
    img = lambda s: str(YOLO / "images" / f"{Path(s['filepath']).stem}.jpg")
    manifest = {"source": "dgural/bdd100k@c2e7f266756bcd07b87f1a45a35937c8eac20241",
                "classes": CLASSES, "boxes_kept": n_boxes, "boxes_dropped": dict(dropped),
                "splits": {}}
    for split, items in splits.items():
        sha = write_list(SPLITS / f"{split}.txt", [img(s) for s in items])
        manifest["splits"][split] = {"images": len(items), "sha256": sha}
    pool = [img(s) for s in splits["night_pool"]]
    for n in NIGHT_DOSES:
        sha = write_list(SPLITS / f"night_{n}.txt", pool[:n])  # nested: night_100 is inside night_250
        manifest["splits"][f"night_{n}"] = {"images": n, "sha256": sha}

    write_arm("A", ["day_train"])
    write_arm("B", ["day_train", "night_500"])
    write_arm("C", ["day_train", "day_extra"])
    for n in NIGHT_DOSES:
        write_arm(f"dose{n}", ["day_train", f"night_{n}"])
    for split in ("day_test", "night_test", "dusk_test", "day_val", "night_val"):
        write_eval(split)

    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v["images"] for k, v in manifest["splits"].items()}))
    print(f"boxes kept {n_boxes}, dropped {dict(dropped)}")


if __name__ == "__main__":
    main()
