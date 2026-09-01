"""Graphical abstract — an illustration, not a chart. Drawn with matplotlib primitives.
Journal spec: min 531 x 1328 px (h x w). Rendered 13 x 5 cm @ 300 dpi.
Output: article/final/PExpo-Bench_graphical_abstract.{png,pdf}

One message: three ways to answer an exposure question, and what each buys and costs.
Cells shown are the manuscript headline cells (Section 3.2):
  GPT-5.4 A0 (unwrapped flagship) | GPT-5.4-mini A4 (wrapped small) | GPT-5.4 A3 (flagship best)
Drawing units are 260 x 100 so that x and y are physically equal on a 13 x 5 cm canvas.
"""
import os
import json, pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
df = pd.read_parquet(ROOT / "runs/v4_scored/all_scored_v4_main.parquet")
CANON = json.loads((ROOT / "article/final/V4_NUMBERS_20260818.json").read_text())
acc = lambda m, a: df[(df.model == m) & (df.arch == a)].score.mean() * 100
cost = lambda m, k: CANON["cost"][f"{m}|{k}"]

INK, MUTE, GOOD, WARN = "#2b2b2b", "#7d7a73", "#2f6b53", "#9c574b"
BIG, SMALL, TOOLBG, TOOLED = "#3d6b96", "#2f8f6b", "#efc85e", "#a8801c"

plt.rcParams.update({"font.family": "sans-serif"})
fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 260); ax.set_ylim(0, 100)

ax.text(130, 92.5, "Wrap a small model with tools, or buy a bigger one?",
        ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)
ax.text(130, 85, "PExpo-Bench · 1,027 curated personal-exposure questions · EPA · WHO · IRIS · ICRP · GBD",
        ha="center", va="center", fontsize=4.6, color=MUTE)


def model_chip(cx, cy, label, colour, w=62, h=12):
    for kw in (dict(facecolor=colour, edgecolor="none", alpha=0.14, zorder=2),
               dict(facecolor="none", edgecolor=colour, lw=1.1, zorder=3)):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=1.2", **kw))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=5.9,
            fontweight="bold", color=colour, zorder=4)


def icon_calc(x, y):
    ax.add_patch(Rectangle((x - 2.1, y - 2.6), 4.2, 5.2, facecolor="white",
                           edgecolor=TOOLED, lw=0.6, zorder=5))
    ax.add_patch(Rectangle((x - 1.5, y + 0.9), 3.0, 1.1, facecolor=TOOLED,
                           edgecolor="none", zorder=6))
    for r in range(2):
        for c in range(3):
            ax.add_patch(Circle((x - 1.3 + c * 1.3, y - 0.5 - r * 1.4), 0.32,
                                facecolor=TOOLED, edgecolor="none", zorder=6))


def icon_units(x, y):
    ax.add_patch(FancyArrowPatch((x - 2.4, y + 1.1), (x + 2.4, y + 1.1), arrowstyle="-|>",
                                 mutation_scale=3.5, color=TOOLED, lw=0.75, zorder=5))
    ax.add_patch(FancyArrowPatch((x + 2.4, y - 1.4), (x - 2.4, y - 1.4), arrowstyle="-|>",
                                 mutation_scale=3.5, color=TOOLED, lw=0.75, zorder=5))


def icon_doc(x, y):
    ax.add_patch(Rectangle((x - 1.6, y - 2.9), 3.6, 5.0, facecolor="white",
                           edgecolor=TOOLED, lw=0.55, zorder=5))
    ax.add_patch(Rectangle((x - 2.4, y - 2.2), 3.6, 5.0, facecolor="white",
                           edgecolor=TOOLED, lw=0.6, zorder=6))
    for i in range(3):
        ax.plot([x - 1.8, x + 0.6], [y + 1.4 - i * 1.3] * 2, color=TOOLED,
                lw=0.5, zorder=7, solid_capstyle="butt")


def tool_chips(cx, cy):
    for i, (lb, draw) in enumerate([("calc", icon_calc), ("units", icon_units),
                                    ("EFH / IRIS", icon_doc)]):
        x = cx - 23 + i * 23
        ax.add_patch(FancyBboxPatch((x - 10.5, cy - 4.2), 21, 8.4, boxstyle="round,pad=0.8",
                                    facecolor=TOOLBG, edgecolor=TOOLED, lw=0.7,
                                    alpha=0.92, zorder=3))
        draw(x - 6.4, cy)
        ax.text(x - 2.6, cy, lb, ha="left", va="center", fontsize=3.5,
                color="#4a3c0c", zorder=6)
    ax.text(cx, cy - 9.5, "callable tools", ha="center", va="center", fontsize=4.1,
            color=MUTE, style="italic", zorder=4)


def outcome(cx, pct, price, highlight=False):
    ax.text(cx, 33.5, f"{pct:.1f}%", ha="center", va="center",
            fontsize=13.5 if highlight else 10.5, fontweight="bold",
            color=GOOD if highlight else INK, zorder=4)
    ax.add_patch(FancyBboxPatch((cx - 28, 18.5), 56, 9, boxstyle="round,pad=0.9",
                                facecolor="#f2f0eb", edgecolor="#c9c5bc", lw=0.6, zorder=3))
    ax.text(cx, 23, f"${price:.2f} / 100 questions", ha="center", va="center",
            fontsize=4.7, color=INK, zorder=4)


COL = [46, 130, 214]

# recommended configuration, called out with a tinted panel
ax.add_patch(FancyBboxPatch((COL[1] - 38, 15.5), 76, 64.5, boxstyle="round,pad=1.5",
                            facecolor=GOOD, edgecolor=GOOD, lw=1.2, alpha=0.06, zorder=1))

model_chip(COL[0], 70, "GPT-5.4", BIG)
ax.text(COL[0], 56, "no tools", ha="center", va="center", fontsize=4.7,
        color=MUTE, style="italic")
outcome(COL[0], acc("gpt-5.4", "A0_naive"), cost("gpt-5.4", "A0"))

for cx, model, colour, arch, key, hi in [
        (COL[1], "gpt-5.4-mini", SMALL, "A4p_hybrid_constrained", "A4", True),
        (COL[2], "gpt-5.4", BIG, "A3_agent", "A3", False)]:
    model_chip(cx, 70, model.upper().replace("GPT", "GPT").replace("-MINI", "-mini"), colour)
    ax.add_patch(FancyArrowPatch((cx, 63.2), (cx, 59.0), arrowstyle="-|>",
                                 mutation_scale=6, color=MUTE, lw=0.9, zorder=4))
    tool_chips(cx, 53)
    outcome(cx, acc(model, arch), cost(model, key), highlight=hi)

ax.add_patch(FancyBboxPatch((14, 2.8), 232, 8.6, boxstyle="round,pad=0.9",
                            facecolor="#f2f0eb", edgecolor="#c9c5bc", lw=0.7, zorder=2))
ax.text(130, 7.1, "+16.0 points over the flagship used alone   ·   "
                  "the flagship's own accuracy at roughly a third of its price",
        ha="center", va="center", fontsize=5.3, fontweight="bold", color=INK, zorder=4)

OUT = ROOT / "article/final/PExpo-Bench_graphical_abstract"
fig.savefig(f"{OUT}.png", dpi=300, facecolor="white")
fig.savefig(f"{OUT}.pdf", facecolor="white")
from PIL import Image
w, h = Image.open(f"{OUT}.png").size
print(f"saved graphical abstract: ({w}, {h}) px (spec min 1328x531)")
