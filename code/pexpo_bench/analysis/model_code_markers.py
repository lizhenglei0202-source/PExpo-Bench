"""Compact model-code markers for publication benchmark figures.

The marker uses a single-letter model code inside a solid circle.  This keeps
the statistical glyph readable at manuscript scale and leaves colour free to
encode the plot-specific grouping variable.
"""

from __future__ import annotations

import matplotlib.colors as mcolors


MODEL_CODE = {
    "GPT-5.4": "F",
    "GPT-5.4-mini": "M",
    "GPT-5.4-nano": "N",
    "DeepSeek-V4": "D",
}


def _contrast_text(colour: str) -> str:
    red, green, blue = mcolors.to_rgb(colour)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#20242a" if luminance > 0.60 else "white"


def add_model_code_marker(
    ax,
    x: float,
    y: float,
    model: str,
    colour: str,
    *,
    size: float = 105,
    alpha: float = 1.0,
    frontier: bool = False,
    zorder: float = 5,
):
    """Draw a solid circular point with an F/M/N/D model code."""
    scale = (size / 105.0) ** 0.5
    ax.scatter(
        [x],
        [y],
        s=size,
        facecolor=colour,
        edgecolor="white",
        linewidth=0.75,
        alpha=alpha,
        zorder=zorder,
    )
    if frontier:
        ax.scatter(
            [x],
            [y],
            s=size * 1.42,
            facecolor="none",
            edgecolor="#24272b",
            linewidth=1.45,
            alpha=alpha,
            zorder=zorder + 0.2,
        )
    ax.annotate(
        MODEL_CODE[model],
        (x, y),
        ha="center",
        va="center",
        fontsize=5.5 * scale,
        fontweight="bold",
        color=_contrast_text(colour),
        alpha=alpha,
        zorder=zorder + 0.5,
    )

