"""Graphical abstract, drawn entirely with matplotlib from the corrected-rerun data
(no generative artwork; same declared scripting pipeline as the paper's data figures).
Journal spec: min 531 x 1328 px (h x w), readable at 5 x 13 cm. We render 13 x 5 cm @ 300 dpi.
Output: article/final/PExpo-Bench_graphical_abstract.{png,pdf}
"""
import os
import pathlib, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
sys.path.insert(0, str(ROOT / "pexpo_bench/analysis"))
from paper_palette import ARCH_FILL, ARCH_EDGE  # noqa: E402

current = pd.read_parquet(ROOT / "runs/v4_scored/all_scored_v4_main.parquet")
prior = pd.read_parquet(ROOT / "runs/v3_scored/all_scored_v2.parquet")
prior = prior[~prior.retired]
PAPER = ["A0_naive", "A1_context_eng", "A2p_rag_constrained", "A3_agent", "A4p_hybrid_constrained"]
LAB = dict(zip(PAPER, ["A0", "A1", "A2", "A3", "A4"]))
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4"]

fig = plt.figure(figsize=(13 / 2.54, 5 / 2.54), dpi=300)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.15], wspace=0.42,
                      left=0.035, right=0.975, top=0.78, bottom=0.16)
plt.rcParams.update({"font.family": "sans-serif"})
fig.suptitle("PExpo-Bench: validate the execution environment before trusting LLM architecture effects",
             fontsize=7.2, fontweight="bold", y=0.965)

# Panel 1: benchmark schematic (pure matplotlib shapes/text)
ax = fig.add_subplot(gs[0]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.add_patch(FancyBboxPatch((0.03, 0.62), 0.94, 0.3, boxstyle="round,pad=0.02",
                            facecolor="#f2f0eb", edgecolor="#8a877e", lw=0.7))
ax.text(0.5, 0.83, "1,027 curated personal-exposure items", ha="center", fontsize=5.0, fontweight="bold")
ax.text(0.5, 0.69, "5 sub-domains · 3 question types\ngold answers independently re-verified",
        ha="center", fontsize=4.8, va="center")
for i, (lab, desc) in enumerate([("A0", "naive"), ("A1", "context"), ("A2", "RAG"),
                                 ("A3", "tools"), ("A4", "tools+RAG")]):
    x = 0.03 + i * 0.194
    ax.add_patch(FancyBboxPatch((x, 0.18), 0.165, 0.24, boxstyle="round,pad=0.012",
                                facecolor=ARCH_FILL[lab], edgecolor=ARCH_EDGE[lab], lw=0.8))
    ax.text(x + 0.082, 0.345, lab, ha="center", fontsize=5.4, fontweight="bold")
    ax.text(x + 0.082, 0.235, desc, ha="center", fontsize=4.2)
ax.add_patch(FancyArrowPatch((0.5, 0.60), (0.5, 0.47), arrowstyle="-|>", mutation_scale=6,
                             color="#8a877e", lw=0.8))
ax.text(0.5, 0.05, "5 complete configurations × 4 LLMs", ha="center", fontsize=5.0)

# Panel 2: the artifact (before/after, nano A4 highlighted)
ax = fig.add_subplot(gs[1])
y2 = [prior[(prior.model == "gpt-5.4-nano") & (prior.arch == a)].score.mean() * 100 for a in PAPER]
y4 = [current[(current.model == "gpt-5.4-nano") & (current.arch == a)].score.mean() * 100 for a in PAPER]
xs = range(5)
ax.bar([x - 0.19 for x in xs], y2, width=0.36, facecolor="white",
       edgecolor=[ARCH_EDGE[LAB[a]] for a in PAPER], lw=0.9, label="original runs")
ax.bar([x + 0.19 for x in xs], y4, width=0.36,
       color=[ARCH_FILL[LAB[a]] for a in PAPER],
       edgecolor=[ARCH_EDGE[LAB[a]] for a in PAPER], lw=0.9, label="corrected rerun")
ax.annotate("defective sandbox\nmanufactured a\n−26 pp 'collapse'", xy=(4 - 0.19, 49),
            xytext=(1.95, 123), fontsize=4.6, color="#8c3a32", ha="left", va="top",
            arrowprops=dict(arrowstyle="->", lw=0.6, color="#8c3a32",
                            connectionstyle="arc3,rad=0.18"))
ax.set_xticks(list(xs), [LAB[a] for a in PAPER], fontsize=5)
ax.tick_params(axis="y", labelsize=4.6, width=0.5, length=2)
ax.set_ylabel("Accuracy (%)", fontsize=5)
ax.set_ylim(0, 125)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_title("Weakest model (GPT-5.4-nano)", fontsize=5.6)
ax.legend(fontsize=4.2, frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.2)
ax.spines[["top", "right"]].set_visible(False)
[s.set_linewidth(0.5) for s in ax.spines.values()]

# Panel 3: the corrected result (cross-model means)
ax = fig.add_subplot(gs[2])
means = [sum(current[(current.model == m) & (current.arch == a)].score.mean() for m in MODELS) / 4 * 100 for a in PAPER]
ax.bar(range(5), means, color=[ARCH_FILL[LAB[a]] for a in PAPER],
       edgecolor=[ARCH_EDGE[LAB[a]] for a in PAPER], lw=0.9, width=0.62)
for i, v in enumerate(means):
    ax.annotate(f"{v:.0f}", (i, v + 1.5), ha="center", fontsize=4.8)
ax.annotate("tool access is the intervention that matters;\nretrieval adds cost without accuracy",
            (2.0, 120), fontsize=4.6, ha="center", va="top", color="#5a5750")
ax.set_xticks(range(5), [LAB[a] for a in PAPER], fontsize=5)
ax.tick_params(axis="y", labelsize=4.6, width=0.5, length=2)
ax.set_ylim(0, 125)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel("Cross-model mean (%)", fontsize=5)
ax.set_title("Corrected rerun, all four models", fontsize=5.6)
ax.spines[["top", "right"]].set_visible(False)
[s.set_linewidth(0.5) for s in ax.spines.values()]

out = ROOT / "article/final"
fig.savefig(out / "PExpo-Bench_graphical_abstract.png", dpi=300)
fig.savefig(out / "PExpo-Bench_graphical_abstract.pdf")
import PIL.Image as I
im = I.open(out / "PExpo-Bench_graphical_abstract.png")
print("saved graphical abstract:", im.size, "px (spec min 1328x531)")
