"""Graphical abstract, drawn entirely with matplotlib from the scored benchmark data
(no generative artwork; same declared scripting pipeline as the paper's data figures).
Journal spec: min 531 x 1328 px (h x w), readable at 5 x 13 cm. We render 13 x 5 cm @ 300 dpi.
Output: article/final/PExpo-Bench_graphical_abstract.{png,pdf}

Narrative: the wrapper beats the tier. A wrapped small model matches the flagship's best
configuration at a fraction of the cost; the accuracy-cost frontier is owned by
open-weight + tools.
"""
import os
import json, pathlib, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
sys.path.insert(0, str(ROOT / "pexpo_bench/analysis"))
from paper_palette import ARCH_FILL, ARCH_EDGE  # noqa: E402

df = pd.read_parquet(ROOT / "runs/v4_scored/all_scored_v4_main.parquet")
CANON = json.loads((ROOT / "article/final/V4_NUMBERS_20260818.json").read_text())
PAPER = ["A0_naive", "A1_context_eng", "A2p_rag_constrained", "A3_agent", "A4p_hybrid_constrained"]
LAB = dict(zip(PAPER, ["A0", "A1", "A2", "A3", "A4"]))
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4"]
SHORT = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4-mini",
         "gpt-5.4-nano": "GPT-5.4-nano", "deepseek-v4": "DeepSeek-V4"}
acc = lambda m, a: df[(df.model == m) & (df.arch == a)].score.mean() * 100
cost = lambda m, a: CANON["cost"][f"{m}|{LAB[a]}"]

fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54), dpi=300)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.12, 1.26], wspace=0.46,
                      left=0.035, right=0.975, top=0.78, bottom=0.17)
plt.rcParams.update({"font.family": "sans-serif"})
fig.suptitle("PExpo-Bench: how you wrap the model matters more than which model you buy",
             fontsize=7.2, fontweight="bold", y=0.965)

# ---------------- Panel 1: the benchmark and the five wrappers ----------------
ax = fig.add_subplot(gs[0]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.add_patch(FancyBboxPatch((0.03, 0.62), 0.94, 0.30, boxstyle="round,pad=0.02",
                            facecolor="#f2f0eb", edgecolor="#8a877e", lw=0.7))
ax.text(0.5, 0.83, "1,027 curated personal-exposure items", ha="center",
        fontsize=4.5, fontweight="bold")
ax.text(0.5, 0.69, "5 sub-domains · 3 question types\nEPA · WHO · IRIS · ICRP · GBD sources",
        ha="center", fontsize=4.6, va="center")
for i, (lab, desc) in enumerate([("A0", "naive"), ("A1", "context"), ("A2", "RAG"),
                                 ("A3", "tools"), ("A4", "tools\n+RAG")]):
    x = 0.03 + i * 0.194
    ax.add_patch(FancyBboxPatch((x, 0.18), 0.165, 0.24, boxstyle="round,pad=0.012",
                                facecolor=ARCH_FILL[lab], edgecolor=ARCH_EDGE[lab], lw=0.8))
    ax.text(x + 0.082, 0.345, lab, ha="center", fontsize=5.4, fontweight="bold")
    ax.text(x + 0.082, 0.245, desc, ha="center", fontsize=3.9, va="center", linespacing=0.95)
ax.add_patch(FancyArrowPatch((0.5, 0.60), (0.5, 0.47), arrowstyle="-|>", mutation_scale=6,
                             color="#8a877e", lw=0.8))
ax.text(0.5, 0.05, "5 wrappers × 4 base models", ha="center", fontsize=5.0)

# ---------------- Panel 2: wrapper beats tier ----------------
ax = fig.add_subplot(gs[1])
bars = [("GPT-5.4\nnaive", acc("gpt-5.4", "A0_naive"), ARCH_FILL["A0"], ARCH_EDGE["A0"]),
        ("GPT-5.4-mini\n+ tools", acc("gpt-5.4-mini", "A4p_hybrid_constrained"), ARCH_FILL["A4"], ARCH_EDGE["A4"]),
        ("GPT-5.4\n+ tools", acc("gpt-5.4", "A3_agent"), ARCH_FILL["A3"], ARCH_EDGE["A3"])]
xs = range(3)
ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars],
       edgecolor=[b[3] for b in bars], lw=0.9, width=0.64)
for i, b in enumerate(bars):
    ax.annotate(f"{b[1]:.1f}", (i, b[1] + 1.2), ha="center", fontsize=5.2, fontweight="bold")
ax.annotate("", xy=(1, 99.0), xytext=(0, 99.0),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="#2f6b53"))
ax.text(0.5, 100.6, "+16.0 pp", ha="center", fontsize=5.0, color="#2f6b53", fontweight="bold")
ax.text(2.0, 96.0, "same accuracy,\n9× the price", ha="center", va="center",
        fontsize=4.6, color="#6b6357", linespacing=1.1)
ax.set_xticks(list(xs), [b[0] for b in bars], fontsize=4.7, linespacing=1.1)
ax.set_ylim(60, 108); ax.set_yticks([60, 70, 80, 90])
ax.tick_params(axis="y", labelsize=4.6, width=0.5, length=2)
ax.set_ylabel("Accuracy (%)", fontsize=5)
ax.set_title("Wrapping a small model beats buying a big one", fontsize=5.4)
ax.spines[["top", "right"]].set_visible(False)
[s.set_linewidth(0.5) for s in ax.spines.values()]

# ---------------- Panel 3: accuracy-cost frontier ----------------
ax = fig.add_subplot(gs[2])
pts = [(m, a, cost(m, a), acc(m, a)) for m in MODELS for a in PAPER]
front = [p for p in pts if not any(q[2] <= p[2] and q[3] >= p[3] and (q[2] < p[2] or q[3] > p[3])
                                   for q in pts)]
fset = {(m, a) for m, a, _, _ in front}
for m, a, c, y in pts:
    on = (m, a) in fset
    ax.scatter([c], [y], s=26 if on else 15, color=ARCH_FILL[LAB[a]],
               edgecolor="#24272b" if on else ARCH_EDGE[LAB[a]],
               linewidth=1.1 if on else 0.5, alpha=1.0 if on else 0.45, zorder=3 if on else 2)
fs = sorted(front, key=lambda p: p[2])
ax.plot([p[2] for p in fs], [p[3] for p in fs], color="#2f6b53", lw=0.8, ls="-", zorder=1)
ax.annotate("every efficient choice:\nopen-weight + tools", (fs[-1][2], fs[-1][3]),
            xytext=(0.30, 70), fontsize=4.6, color="#2f6b53", ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.6, color="#2f6b53"))
ax.set_xscale("log")
ax.set_xlabel("Cost (USD / 100 questions)", fontsize=5)
ax.set_ylabel("Accuracy (%)", fontsize=5)
ax.tick_params(labelsize=4.6, width=0.5, length=2)
ax.set_ylim(48, 99)
ax.set_title("Accuracy–cost frontier (20 configurations)", fontsize=5.4)
ax.grid(alpha=0.18, ls=":")
ax.spines[["top", "right"]].set_visible(False)
[s.set_linewidth(0.5) for s in ax.spines.values()]

OUT = ROOT / "article/final/PExpo-Bench_graphical_abstract"
fig.savefig(f"{OUT}.png", dpi=300, facecolor="white")
fig.savefig(f"{OUT}.pdf", facecolor="white")
from PIL import Image
w, h = Image.open(f"{OUT}.png").size
print(f"saved graphical abstract: ({w}, {h}) px (spec min 1328x531)")
