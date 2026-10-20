"""Dose curve figure for the README: day and night mAP50-95 vs night images added (seed 0), arm C for reference."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL = "runs/eval"
DOSES = [0, 100, 250, 500, 1000, 2000]
TAG = {0: "A_s0", 500: "B_s0"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"night_test": "#2a78d6", "day_test": "#eb6834"}
NAME = {"night_test": "night test", "day_test": "day test"}


def get(tag, split):
    r = json.load(open(f"{EVAL}/{tag}__{split}.json"))["main"]
    return 100 * r["mAP50_95"], [100 * c for c in r["ci95"]]


fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160, facecolor=SURF)
ax.set_facecolor(SURF)
x = list(range(len(DOSES)))
for split in ("day_test", "night_test"):
    v = [get(TAG.get(n, f"dose{n}_s0"), split) for n in DOSES]
    y = [p for p, _ in v]
    err = [[p - c[0] for p, c in v], [c[1] - p for p, c in v]]
    ax.errorbar(x, y, yerr=err, color=COL[split], lw=2, marker="o", ms=7, capsize=0, elinewidth=1,
                mec=SURF, mew=2, zorder=3)
    ax.annotate(NAME[split], (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points", va="center",
                color=INK, fontsize=10)
    c, _ = get("C_s0", split)
    ax.plot([x[3]], [c], marker="D", ms=7, mfc=SURF, mec=COL[split], mew=2, ls="none", zorder=4)
ax.annotate("hollow = +500 extra DAY images (arm C)", (x[3], get("C_s0", "night_test")[0]), xytext=(0, -26),
            textcoords="offset points", ha="center", color=INK2, fontsize=8.5)
ax.set_xticks(x, [str(n) for n in DOSES])
ax.set_xlabel("night images added to 3,158 day training images", color=INK2)
ax.set_ylabel("mAP50-95 (points)", color=INK2)
ax.set_title("Adding night images lifts night mAP, day stays flat (seed 0, 95% bootstrap CI)",
             color=INK, fontsize=10.5, loc="left")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, length=0)
ax.set_xlim(-0.3, len(DOSES) - 0.2)
fig.tight_layout()
fig.savefig("runs/dose.png", facecolor=SURF)
print("runs/dose.png")
