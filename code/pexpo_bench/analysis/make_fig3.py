import os
"""New Figure 3 for the reframed paper: 'Execution environment manufactures architecture
effects.' Three panels from frozen data lines (see DATA_VERSIONS.md):
(a) accuracy (defective env) -> (corrected env) per model x arm — slope exhibit;
(b) GPT-5.4-nano open-ended answer-type collapse rate, vs ;
(c) A4−A3 across seeds () vs the main-run value, nano + DeepSeek.
Output: article/final/svg-fig-v4/Figure_3_environment_effect.{png,svg}
"""
import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
sys.path.insert(0, str(ROOT / "pexpo_bench/analysis"))
from paper_palette import ARCH_FILL, ARCH_EDGE, FRAME  # noqa: E402

OUT = ROOT / "article/final/svg-fig-v4"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333"})

current = pd.read_parquet(ROOT / "runs/v4_scored/all_scored_v4_main.parquet")
prior = pd.read_parquet(ROOT / "runs/v3_scored/all_scored_v2.parquet")
prior = prior[~prior.retired]
PAPER = ["A0_naive", "A1_context_eng", "A2p_rag_constrained", "A3_agent", "A4p_hybrid_constrained"]
LAB = dict(zip(PAPER, ["A0", "A1", "A2", "A3", "A4"]))
MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v4"]
MSHORT = {"gpt-5.4": "GPT-5.4", "gpt-5.4-mini": "GPT-5.4-mini", "gpt-5.4-nano": "GPT-5.4-nano", "deepseek-v4": "DeepSeek-V4"}
CODE = {"gpt-5.4": "F", "gpt-5.4-mini": "M", "gpt-5.4-nano": "N", "deepseek-v4": "D"}

fig = plt.figure(figsize=(10.6, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.0, 1.15], wspace=0.32,
                      left=0.06, right=0.985, top=0.86, bottom=0.15)

# (a) slope exhibit
ax = fig.add_subplot(gs[0])
xpos = {}
for i, ar in enumerate(PAPER):
    for j, m in enumerate(MODELS):
        xpos[(ar, m)] = i * 5 + j
for (ar, m), x in xpos.items():
    y2 = prior[(prior.model == m) & (prior.arch == ar)].score.mean() * 100
    y4 = current[(current.model == m) & (current.arch == ar)].score.mean() * 100
    col = ARCH_FILL[LAB[ar]] if isinstance(ARCH_FILL, dict) else "#888888"
    edge = ARCH_EDGE[LAB[ar]] if isinstance(ARCH_EDGE, dict) else "#333333"
    hl = (m == "gpt-5.4-nano" and ar == "A4p_hybrid_constrained")
    ax.plot([x, x], [y2, y4], color=edge, lw=2.4 if hl else 1.1,
            alpha=1.0 if hl else 0.65, zorder=3 if hl else 2)
    ax.scatter([x], [y2], marker="o", s=26, facecolor="white", edgecolor=edge,
               linewidth=1.1, zorder=4)
    ax.scatter([x], [y4], marker="o", s=30, facecolor=col, edgecolor=edge,
               linewidth=1.1, zorder=4)
    ax.annotate(CODE[m], (x, min(y2, y4) - 3.5), ha="center", va="top", fontsize=6.5, color="#555555")
    if hl:
        ax.annotate("+38.9 pp", (x + 0.35, (y2 + y4) / 2), fontsize=8, color=edge,
                    fontweight="bold", va="center")
for i, ar in enumerate(PAPER):
    ax.annotate(LAB[ar], (i * 5 + 1.5, 101.5), ha="center", fontsize=9, fontweight="bold",
                color=ARCH_EDGE[LAB[ar]] if isinstance(ARCH_EDGE, dict) else "#333333")
ax.set_ylim(38, 100)
ax.set_xlim(-1, 24)
ax.set_xticks([])
ax.set_ylabel("Accuracy (%)")
ax.set_title("(a) Original runs (open) → corrected rerun (filled)", loc="left", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

# (b) nano format collapse
ax = fig.add_subplot(gs[1])
vals = [9.3, 63.9, 8.7, 8.9]
labels = ["A3\norig.", "A4\norig.", "A3\ncorr.", "A4\ncorr."]
cols = ["#c9c5bc", "#CF8A7E", "#c9c5bc", "#E8B34F"]
bars = ax.bar(range(4), vals, color=cols, edgecolor="#333333", linewidth=0.8, width=0.62)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v + 1.5), ha="center", fontsize=8)
ax.set_xticks(range(4), labels, fontsize=8)
ax.set_ylabel("Tool-style answers to\nopen-ended items (%)")
ax.set_ylim(0, 72)
ax.set_title("(b) GPT-5.4-nano format collapse", loc="left", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

# (c) seeds
ax = fig.add_subplot(gs[2])
allp = pd.read_parquet(ROOT / "runs/v4_scored/all_scored_v4.parquet")
c = allp[(allp.phase == "C") & (allp.question_type != "open_ended")]
obj4 = current[current.question_type != "open_ended"]
obj2 = prior[prior.question_type != "open_ended"]
for k, (m, col) in enumerate([("gpt-5.4-nano", "#8a6d3b"), ("deepseek-v4", "#6F88B0")]):
    ds = []
    for s in [43, 44, 45]:
        pp = c[(c.model == m) & (c.seed == s)].pivot_table(index="qid", columns="arch", values="score")[
            ["A3_agent", "A4p_hybrid_constrained"]].dropna()
        ds.append((pp.A4p_hybrid_constrained - pp.A3_agent).mean() * 100)
    p4 = obj4[obj4.model == m].pivot_table(index="qid", columns="arch", values="score")[
        ["A3_agent", "A4p_hybrid_constrained"]].dropna()
    main4 = (p4.A4p_hybrid_constrained - p4.A3_agent).mean() * 100
    p2 = obj2[obj2.model == m].pivot_table(index="qid", columns="arch", values="score")[
        ["A3_agent", "A4p_hybrid_constrained"]].dropna()
    main2 = (p2.A4p_hybrid_constrained - p2.A3_agent).mean() * 100
    x = k * 1.0
    ax.scatter([x - 0.13] * 3 + [x], ds + [main4], s=32, facecolor=col, edgecolor="#333333",
               linewidth=0.8, zorder=4, label=None)
    ax.scatter([x + 0.16], [main2], s=46, marker="X", facecolor="white", edgecolor="#b3443c",
               linewidth=1.3, zorder=4)
    ax.annotate(MSHORT[m], (x, -30 if m == "gpt-5.4-nano" else -30), ha="center", fontsize=8,
                xytext=(x, -33), textcoords="data")
ax.axhline(0, color="#999999", lw=0.8, ls="--")
ax.annotate("original run\n(defective env.)", (1.18, -2.02), fontsize=7, color="#b3443c", va="top")
ax.annotate("original run: −26.0 ↓ (off scale)", (0.02, -27), fontsize=7, color="#b3443c")
ax.set_ylim(-31, 7)
ax.set_xlim(-0.5, 1.7)
ax.set_xticks([])
ax.set_ylabel("A4 − A3 (pp), objective items")
ax.set_title("(c) Seed replication, corrected rerun", loc="left", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

for ext in ["png", "svg"]:
    fig.savefig(OUT / f"Figure_3_environment_effect.{ext}", dpi=200)
print("saved", OUT / "Figure_3_environment_effect.png")
