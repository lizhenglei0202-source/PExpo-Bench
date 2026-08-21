"""SI Figure S3 (corrected campaign): factorial decomposition of the harness.
Replaces the original-campaign evidence-use ablation figure (its plain A2/A4 arms exist
only in the original runs). Design language matches the paper's dot/dumbbell figures.
Output: article/final/svg-fig-v4/Figure_S3_factorial.{png,svg}
"""
import os
import pathlib, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
sys.path.insert(0, str(ROOT / "pexpo_bench/analysis"))
from paper_palette import ARCH_FILL, ARCH_EDGE  # noqa: E402

allp = pd.read_parquet(ROOT / "runs/v4_scored/all_scored_v4.parquet")
fact = allp[allp.phase == "B"]
calcA = allp[(allp.phase == "A") & (allp.question_type == "calculation")]
ARMS = [("A3", "A", "A3_agent"), ("+R", "B", "fA3_R"), ("+P", "B", "fA3_P"), ("+B", "B", "fA3_B"),
        ("+R+P", "B", "fA3_RP"), ("+R+B", "B", "A4_hybrid"), ("+P+B", "B", "fA3_PB"),
        ("A4\n(+R+P+B)", "A", "A4p_hybrid_constrained")]
MODELS = [("gpt-5.4-nano", "GPT-5.4-nano", "#8a6d3b"), ("deepseek-v4", "DeepSeek-V4", "#6F88B0")]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9.5, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333"})
fig, ax = plt.subplots(figsize=(9.2, 4.0))
for k, (m, name, col) in enumerate(MODELS):
    qids = set(fact[fact.model == m].qid.unique())
    ys = []
    for label, ph, arch in ARMS:
        src = calcA if ph == "A" else fact
        ys.append(src[(src.model == m) & (src.arch == arch) & (src.qid.isin(qids))].score.mean() * 100)
    xs = [i + (-0.12 if k == 0 else 0.12) for i in range(len(ARMS))]
    ax.plot(xs, ys, color=col, lw=1.2, alpha=0.7, zorder=2)
    ax.scatter(xs, ys, s=46, facecolor=col, edgecolor="#333333", linewidth=0.9, zorder=3, label=name)
    for x, y, (label, _, arch) in zip(xs, ys, ARMS):
        if arch in ("A3_agent", "A4p_hybrid_constrained"):
            ax.annotate(f"{y:.1f}", (x, y + 1.1), ha="center", fontsize=7.5, color=col)
ax.axhspan(0, 0, color="none")
base = {m: (calcA if True else None) for m, _, _ in MODELS}
for k, (m, name, col) in enumerate(MODELS):
    qids = set(fact[fact.model == m].qid.unique())
    b = calcA[(calcA.model == m) & (calcA.arch == "A3_agent") & (calcA.qid.isin(qids))].score.mean() * 100
    ax.axhline(b, color=col, lw=0.7, ls=":", alpha=0.6)
ax.set_xticks(range(len(ARMS)), [a[0] for a in ARMS], fontsize=9)
ax.set_ylabel("Calculation accuracy (%)")
ax.set_ylim(80, 94)
ax.set_title("Factorial decomposition of the harness (corrected rerun, calculation stream, n = 361):\n"
             "R = retrieval tool, P = evidence-use rules, B = step budget 10", fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
OUT = ROOT / "article/final/svg-fig-v4"
for ext in ["png", "svg"]:
    fig.savefig(OUT / f"Figure_S3_factorial.{ext}", dpi=200)
print("saved Figure_S3_factorial")
