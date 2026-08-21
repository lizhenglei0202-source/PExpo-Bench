#!/usr/bin/env python3
"""Regenerate Figure 2 with compact benchmark markers in the cost panel.

This standalone version reads the frozen figure-data workbook, avoiding the
optional Parquet engine required by ``v3_figures.py``.  In panel c, point colour
encodes architecture, F/M/N/D letters encode the base model, and a separate
black ring marks Pareto-efficient cells.
"""

from __future__ import annotations

import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_code_markers import add_model_code_marker
from paper_palette import (
    ACC_CMAP,
    ALERT,
    ARCH_EDGE as EDGE,
    ARCH_FILL as FILL,
    DIV_CMAP,
    FRAME,
    MODEL_COLOR,
    POSITIVE,
)


ROOT = pathlib.Path(os.environ.get("PEXPO_ROOT", "."))
DATA = ROOT / "article/final/PExpo-Bench_figure_data_v4.xlsx"
OUT = ROOT / "article/final/svg-fig-v4"

ARCHS = ["A0", "A1", "A2", "A3", "A4"]
ARCH_DESC = {
    "A0": "A0 naive",
    "A1": "A1 static context",
    "A2": "A2 RAG",
    "A3": "A3 tool agent",
    "A4": "A4 harness",
}
MODELS = ["GPT-5.4", "GPT-5.4-mini", "GPT-5.4-nano", "DeepSeek-V4"]
KEY2MODEL = {
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "gpt-5.4-nano": "GPT-5.4-nano",
    "deepseek-v4": "DeepSeek-V4",
}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


def _spines_box(ax) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color(FRAME)


def _load_data():
    acc = pd.read_excel(DATA, sheet_name="Fig2a_overall_accuracy")
    cost = pd.read_excel(DATA, sheet_name="Fig2b_accuracy_cost")
    subdomain = pd.read_excel(DATA, sheet_name="Fig3_subdomain_accuracy")
    for frame in (acc, cost, subdomain):
        frame["model"] = frame["model"].map(KEY2MODEL)
    acc_map = {
        m: {
            a: float(acc[(acc.model == m) & (acc.architecture == a)].accuracy_pct.iloc[0])
            for a in ARCHS
        }
        for m in MODELS
    }
    cost_map = {
        m: {
            a: float(cost[(cost.model == m) & (cost.architecture == a)].cost_usd_per_100q.iloc[0])
            for a in ARCHS
        }
        for m in MODELS
    }
    return acc_map, cost_map, subdomain


def _draw_legend_band(fig) -> None:
    ax = fig.add_axes([0.025, 0.006, 0.95, 0.105])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.0, 0.72, "Architecture (point colour)", fontsize=8.2, fontweight="bold", va="center")
    arch_x = [0.15, 0.27, 0.41, 0.53, 0.65]
    for x, arch in zip(arch_x, ARCHS):
        ax.scatter([x], [0.72], s=72, marker="s", facecolor=FILL[arch], edgecolor=EDGE[arch], linewidth=1.1)
        ax.text(x + 0.014, 0.72, ARCH_DESC[arch], fontsize=7.8, ha="left", va="center")
    ax.scatter([0.81], [0.72], s=118, facecolor="white", edgecolor="#24272b", linewidth=1.5)
    ax.text(0.826, 0.72, "Pareto-efficient", fontsize=7.8, ha="left", va="center")
    ax.scatter([0.915], [0.72], s=72, facecolor="#cfcfcf", edgecolor="none", alpha=0.45)
    ax.text(0.929, 0.72, "Dominated", fontsize=7.8, ha="left", va="center")

    ax.text(
        0.0,
        0.23,
        "Base model\n(F/M/N/D letter code)",
        fontsize=7.4,
        fontweight="bold",
        va="center",
        linespacing=1.05,
    )
    model_x = [0.30, 0.48, 0.67, 0.84]
    for x, model in zip(model_x, MODELS):
        add_model_code_marker(
            ax,
            x,
            0.23,
            model,
            "#65717c",
            size=92,
            zorder=10,
        )
        label = model + (" (open-weight)" if model == "DeepSeek-V4" else "")
        ax.text(x + 0.018, 0.23, label, fontsize=7.8, ha="left", va="center")


def draw() -> None:
    accuracy, cost, subdomain = _load_data()
    fig = plt.figure(figsize=(14.4, 9.0))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.95],
        wspace=0.28,
        left=0.115,
        right=0.955,
        top=0.92,
        bottom=0.16,
    )
    left = outer[0].subgridspec(2, 1, height_ratios=[4, 8], hspace=0.92)
    right = outer[1].subgridspec(2, 1, height_ratios=[0.84, 0.16], hspace=0)
    axa = fig.add_subplot(left[0])
    axb = fig.add_subplot(left[1])
    axc = fig.add_subplot(right[0])

    row_models = ["DeepSeek-V4", "GPT-5.4", "GPT-5.4-mini", "GPT-5.4-nano"]
    matrix_a = np.array([[accuracy[m][a] for a in ARCHS] for m in row_models])
    im_a = axa.imshow(matrix_a, cmap=ACC_CMAP, vmin=45, vmax=90, aspect="auto")
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_a.shape[1]):
            value = matrix_a[i, j]
            axa.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="white" if value > 72 else "#2a2a2a",
            )
    axa.set_xticks(range(len(ARCHS)))
    axa.set_xticklabels([ARCH_DESC[a] for a in ARCHS], fontsize=9)
    for tick, arch in zip(axa.get_xticklabels(), ARCHS):
        tick.set_color(EDGE[arch])
        tick.set_fontweight("bold")
    axa.xaxis.set_ticks_position("top")
    axa.set_yticks(range(len(row_models)))
    axa.set_yticklabels(row_models, fontsize=9.5, fontweight="bold")
    axa.tick_params(length=0)
    axa.set_xticks(np.arange(-0.5, len(ARCHS), 1), minor=True)
    axa.set_yticks(np.arange(-0.5, len(row_models), 1), minor=True)
    axa.grid(which="minor", color="white", linewidth=2.4)
    axa.tick_params(which="minor", length=0)
    axa.axvline(2.5, color="#bcbcbc", lw=1.0, zorder=4)
    transform = axa.get_xaxis_transform()
    for x0, x1, label in [(-0.44, 2.44, "No-tool architectures"), (2.56, 4.44, "Tool-enabled architectures")]:
        axa.plot([x0, x1], [1.135, 1.135], color="#999999", lw=1.0, transform=transform, clip_on=False)
        axa.plot([x0, x0], [1.135, 1.10], color="#999999", lw=1.0, transform=transform, clip_on=False)
        axa.plot([x1, x1], [1.135, 1.10], color="#999999", lw=1.0, transform=transform, clip_on=False)
        axa.text((x0 + x1) / 2, 1.165, label, transform=transform, ha="center", va="bottom", fontsize=8, color="#555555")
    _spines_box(axa)
    caxa = axa.inset_axes([0.20, -0.28, 0.60, 0.05])
    colourbar_a = fig.colorbar(im_a, cax=caxa, orientation="horizontal")
    colourbar_a.set_label("Mean accuracy (%)", fontsize=8)
    colourbar_a.ax.tick_params(length=2.5, labelsize=7.5)
    axa.set_title("a", loc="left", fontweight="bold", fontsize=12, pad=44)

    sd_map = {
        "S1_exposure_factors": "S1\nExposure\nfactors",
        "S2_microenv_conc": "S2\nMicroenvironment\nconcentration",
        "S3_trajectory_activity": "S3\nTrajectory/\nactivity",
        "S4_dosimetry": "S4\nDosimetry",
        "S5_health": "S5\nHealth\neffects",
    }
    rows = []
    row_labels = []
    for model in MODELS:
        pivot = subdomain[subdomain.model == model].pivot_table(
            index="subdomain", columns="architecture", values="accuracy_pct", aggfunc="first"
        ).reindex(list(sd_map)).reindex(ARCHS, axis=1)
        baseline = pivot[["A0", "A1", "A2"]].mean(axis=1)
        for arch in ("A3", "A4"):
            rows.append([(pivot.loc[sd, arch] - baseline.loc[sd]) for sd in sd_map])
            row_labels.append(arch)
    matrix_b = np.array(rows)
    im_b = axb.imshow(matrix_b, cmap=DIV_CMAP, vmin=-25, vmax=25, aspect="auto")
    for i in range(matrix_b.shape[0]):
        for j in range(matrix_b.shape[1]):
            value = matrix_b[i, j]
            axb.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white" if abs(value) > 15 else "#2a2a2a")
    axb.set_xticks(range(len(sd_map)))
    axb.set_xticklabels(list(sd_map.values()), fontsize=7.6)
    axb.xaxis.set_ticks_position("top")
    axb.set_yticks(range(len(row_labels)))
    axb.set_yticklabels(row_labels, fontsize=8.5)
    for tick, arch in zip(axb.get_yticklabels(), row_labels):
        tick.set_color(EDGE[arch])
        tick.set_fontweight("bold")
    axb.tick_params(length=0)
    axb.set_xticks(np.arange(-0.5, len(sd_map), 1), minor=True)
    axb.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    axb.grid(which="minor", color="white", linewidth=2.2)
    axb.tick_params(which="minor", length=0)
    for group in range(1, 4):
        axb.axhline(2 * group - 0.5, color="#333333", lw=1.3)
    for group, model in enumerate(MODELS):
        axb.plot([-0.62, -0.62], [2 * group - 0.30, 2 * group + 1.30], color=MODEL_COLOR[model], lw=1.8, clip_on=False, zorder=5, solid_capstyle="round")
        axb.text(-0.80, 2 * group + 0.5, model, fontsize=8.6, fontweight="bold", color=MODEL_COLOR[model], ha="right", va="center")
    _spines_box(axb)
    colourbar_b = fig.colorbar(im_b, ax=axb, fraction=0.022, pad=0.02)
    colourbar_b.set_label("Change vs\nno-tool baseline (pp)", fontsize=8)
    colourbar_b.ax.tick_params(length=2.5, labelsize=7.5)
    axb.set_title("b", loc="left", fontweight="bold", fontsize=12, pad=22)

    cells = [(cost[m][a], accuracy[m][a], m, a) for m in MODELS for a in ARCHS]

    def dominated(cell_cost, cell_accuracy):
        return any(
            c2 <= cell_cost and a2 >= cell_accuracy and (c2 < cell_cost or a2 > cell_accuracy)
            for c2, a2, _, _ in cells
        )

    frontier_xy = sorted([(c, acc) for c, acc, _, _ in cells if not dominated(c, acc)])
    frontier_x, frontier_y = zip(*frontier_xy)
    axc.plot(frontier_x, frontier_y, drawstyle="steps-post", color=POSITIVE, lw=1.8, zorder=2)
    for cell_cost, cell_accuracy, model, arch in cells:
        on_frontier = not dominated(cell_cost, cell_accuracy)
        add_model_code_marker(
            axc,
            cell_cost,
            cell_accuracy,
            model,
            FILL[arch],
            # Frontier glyphs carry the main visual conclusion and must remain
            # legible after the full figure is reduced to journal-column size.
            size=180 if on_frontier else 96,
            alpha=1.0 if on_frontier else 0.58,
            frontier=on_frontier,
            zorder=5 if on_frontier else 3,
        )
    axc.annotate(
        "open-weight model\nanchors the cheap frontier",
        (cost["DeepSeek-V4"]["A0"], accuracy["DeepSeek-V4"]["A0"]),
        (0.013, 79.5),
        fontsize=8.2,
        color=POSITIVE,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": POSITIVE, "lw": 0.8},
    )
    axc.annotate(
        "GPT-5.4 naive: dominated\nby GPT-5.4-mini + tools",
        (cost["GPT-5.4"]["A0"], accuracy["GPT-5.4"]["A0"]),
        (0.55, 67.5),
        fontsize=8.2,
        color="#9c574b",
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": ALERT, "lw": 0.8},
    )
    axc.set_xscale("log")
    axc.set_ylim(44, 91)
    axc.set_xlabel("Cost (USD / 100 questions, log scale)", fontsize=10)
    axc.set_ylabel("Overall accuracy (%)", fontsize=10.5)
    axc.grid(alpha=0.22, ls=":")
    _spines_box(axc)
    axc.set_title("c", loc="left", fontweight="bold", fontsize=12, pad=8)

    _draw_legend_band(fig)
    png = OUT / "Figure_2_results.png"
    svg = OUT / "Figure_2_results.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved: {png}")


if __name__ == "__main__":
    draw()
