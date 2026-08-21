#!/usr/bin/env python3
"""Regenerate manuscript Figure 4 without the scored-Parquet dependency.

The figure uses raw trajectories for type-specific instruction following and
the frozen per-row atomic-claim judgments for the two grounding ratios. Panel
b is intentionally called adjudication coverage—not reference support—because
its numerator includes both SUPPORTED and CONTRADICTED claims.
"""

from __future__ import annotations

import os
import json
import math
import pathlib
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_code_markers import add_model_code_marker
from paper_palette import ARCH_EDGE, ARCH_FILL, A3_FOCUS_FILL, FRAME, GRID, MODEL_COLOR


ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
HR_FILE = ROOT / "runs/v4_rerun/_hr/per_row_hr.jsonl"
RUNS_DIR = ROOT / "runs/v4_rerun"
TOOL_RUNS_DIR = ROOT / "runs/_none_v4"
OUT = ROOT / "article/final/svg-fig-v4"

ARCHS = ["A0", "A1", "A2", "A3", "A4"]
DIR2ARCH = {
    "A0_naive": "A0",
    "A1_context_eng": "A1",
    "A2p_rag_constrained": "A2",
    "A3_agent": "A3",
    "A4p_hybrid_constrained": "A4",
}
MODELS = ["GPT-5.4", "GPT-5.4-mini", "GPT-5.4-nano", "DeepSeek-V4"]
MODEL_KEY = {
    "GPT-5.4": "gpt-5.4",
    "GPT-5.4-mini": "gpt-5.4-mini",
    "GPT-5.4-nano": "gpt-5.4-nano",
    "DeepSeek-V4": "deepseek-v4",
}
KEY2MODEL = {v: k for k, v in MODEL_KEY.items()}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#20242a",
        "text.color": "#20242a",
        "xtick.color": "#343a40",
        "ytick.color": "#343a40",
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


def _extractable_number(value) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return bool(re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value).replace(",", "")))


def _if_pass(row: dict) -> bool:
    if row.get("parse_error"):
        return False
    question_type = row.get("_question_type") or row.get("question_type")
    answer = row.get("answer")
    if question_type == "true_false":
        return isinstance(answer, bool) or str(answer).strip().lower() in {"true", "false"}
    if question_type == "calculation":
        unit = str(row.get("unit") or "").strip().lower()
        return _extractable_number(answer) and unit not in {"", "none", "null"}
    return answer is not None and bool(str(answer).strip())


def _load_instruction_following() -> dict:
    result = {}
    for model in MODELS:
        result[model] = {}
        for raw_arch, arch in DIR2ARCH.items():
            records = {}
            for run_root in (RUNS_DIR, TOOL_RUNS_DIR):
                path = run_root / MODEL_KEY[model] / raw_arch / "run_1.jsonl"
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    records[row["qid"]] = row
            flags = [_if_pass(row) for row in records.values()]
            result[model][arch] = (float(np.mean(flags)), len(flags)) if flags else (np.nan, 0)
    return result


def _wilson_ci(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or np.isnan(rate):
        return np.nan, np.nan
    den = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / den
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def _ratio_ci(frame, numerator, denominator, seed: int, n_boot: int = 1000):
    num = numerator(frame).to_numpy(dtype=float)
    den = denominator(frame).to_numpy(dtype=float)
    point = num.sum() / den.sum() if den.sum() else np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(frame), size=(n_boot, len(frame)))
    boot_num = num[idx].sum(axis=1)
    boot_den = den[idx].sum(axis=1)
    boot = np.divide(boot_num, boot_den, out=np.full(n_boot, np.nan), where=boot_den > 0)
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def build_summaries() -> dict:
    hr_df = pd.DataFrame(json.loads(line) for line in HR_FILE.read_text().splitlines())
    # grounding panels use the stratified 299-item reference-quoted subsample
    _sub = set((ROOT / "runs/v4_rerun/_hr/grounding_subsample_qids.txt").read_text().split())
    hr_df = hr_df[hr_df.qid.isin(_sub)]
    hr_df["arch"] = hr_df["arch"].map(DIR2ARCH)
    hr_df = hr_df.dropna(subset=["arch"])
    hr_df["model"] = hr_df["model"].map(KEY2MODEL)
    if_data = _load_instruction_following()

    summaries = {"if": {}, "coverage": {}, "contradiction": {}}
    for mi, model in enumerate(MODELS):
        for ai, arch in enumerate(ARCHS):
            rate, n = if_data[model][arch]
            lo, hi = _wilson_ci(rate, n)
            summaries["if"][(model, arch)] = (rate, lo, hi, n)
            sub = hr_df[(hr_df.model == model) & (hr_df.arch == arch)]
            summaries["coverage"][(model, arch)] = (
                *_ratio_ci(
                    sub,
                    lambda x: x["n_supported"] + x["n_contradicted"],
                    lambda x: x["n_claims"],
                    seed=100 + mi * 10 + ai,
                ),
                len(sub),
            )
            summaries["contradiction"][(model, arch)] = (
                *_ratio_ci(
                    sub,
                    lambda x: x["n_contradicted"],
                    lambda x: x["n_supported"] + x["n_contradicted"],
                    seed=200 + mi * 10 + ai,
                ),
                len(sub),
            )
    return summaries


def draw() -> None:
    summaries = build_summaries()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.6), sharex=True)
    metrics = [
        ("if", "Instruction following ↑", "Instruction-following rate", (0.68, 1.01)),
        ("coverage", "Adjudication coverage ↑", "Adjudicated claims", (0, 0.55)),
        ("contradiction", "Contradiction rate ↓", "Contradiction among\nadjudicated claims", (0, 0.20)),
    ]
    offsets = np.linspace(-0.18, 0.18, len(MODELS))
    xbase = np.arange(len(ARCHS))

    for pi, (key, title, ylabel, ylim) in enumerate(metrics):
        ax = axes[pi]
        ax.axvspan(2.77, 3.23, color=A3_FOCUS_FILL, alpha=0.62, lw=0, zorder=0)
        ax.text(
            3,
            1.018,
            "Focal A3",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=ARCH_EDGE["A3"],
            fontweight="bold",
            style="italic",
        )
        for mi, model in enumerate(MODELS):
            values = np.array([summaries[key][(model, arch)][0] for arch in ARCHS])
            lows = np.array([summaries[key][(model, arch)][1] for arch in ARCHS])
            highs = np.array([summaries[key][(model, arch)][2] for arch in ARCHS])
            xx = xbase + offsets[mi]
            if key == "if":
                ax.plot(xx, values, color=MODEL_COLOR[model], lw=1.9, alpha=0.88, zorder=2)
            ax.errorbar(
                xx,
                values,
                yerr=[np.maximum(values - lows, 0), np.maximum(highs - values, 0)],
                fmt="none",
                ecolor=mpl.colors.to_rgba(MODEL_COLOR[model], 0.42),
                elinewidth=0.9,
                capthick=0.9,
                capsize=2.0,
                zorder=3,
            )
            for x, y, arch in zip(xx, values, ARCHS):
                add_model_code_marker(
                    ax,
                    x,
                    y,
                    model,
                    ARCH_FILL[arch],
                    size=102,
                    zorder=4,
                )

        ax.set_xticks(xbase)
        ax.set_xticklabels(ARCHS, fontsize=9.5)
        for tick, arch in zip(ax.get_xticklabels(), ARCHS):
            tick.set_color(ARCH_EDGE[arch])
            tick.set_fontweight("bold")
        ax.set_ylabel(ylabel, fontsize=9.8)
        ax.set_ylim(*ylim)
        if key == "if":
            ax.set_yticks(np.arange(0.70, 1.001, 0.05))
        elif key == "coverage":
            ax.set_yticks(np.arange(0, 0.51, 0.10))
        else:
            ax.set_yticks(np.arange(0, 0.201, 0.05))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="y", color=GRID, alpha=0.22, lw=0.8, ls="--")
        ax.grid(axis="x", visible=False)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(FRAME)
            ax.spines[side].set_linewidth(0.85)
        ax.set_title(f"{chr(ord('a') + pi)}    {title}", loc="left", fontweight="bold", fontsize=12.5, pad=14)

    legend_ax = fig.add_axes([0.24, 0.865, 0.52, 0.10])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    positions = [0.03, 0.29, 0.57, 0.84]
    for x, model in zip(positions, MODELS):
        add_model_code_marker(
            legend_ax,
            x,
            0.52,
            model,
            MODEL_COLOR[model],
            size=105,
            zorder=10,
        )
        legend_ax.text(x + 0.045, 0.52, model, ha="left", va="center", fontsize=9.4)
    fig.supxlabel("Architecture", fontsize=10.5, fontweight="bold", y=0.055)
    fig.subplots_adjust(left=0.064, right=0.988, top=0.78, bottom=0.17, wspace=0.30)

    png = OUT / "Figure_4_reliability_diagnostics.png"
    svg = OUT / "Figure_4_reliability_diagnostics.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 4 saved: {png}")


if __name__ == "__main__":
    draw()
