"""Shared visual language for the PExpo-Bench manuscript figures.

Colours are assigned by semantic role and remain stable across figures:
architectures use filled pastel colours, base models use darker categorical
colours, and signed numerical changes use one brick--cream--teal ramp.
"""

from matplotlib.colors import LinearSegmentedColormap


ARCH_FILL = {
    "A0": "#d8d6d0",
    "A1": "#a9a69d",
    "A2": "#6f88b0",
    "A3": "#e8b34f",
    "A4": "#cf8a7e",
}
ARCH_EDGE = {
    "A0": "#96938c",
    "A1": "#6a6760",
    "A2": "#3d5480",
    "A3": "#b3831f",
    "A4": "#9c574b",
}

MODEL_COLOR = {
    "GPT-5.4": "#34618f",
    "GPT-5.4-mini": "#3d8f78",
    "GPT-5.4-nano": "#7b5896",
    "DeepSeek-V4": "#b07a35",
}

TEXT = "#222222"
AXIS = "#333333"
FRAME = "#4f565d"
GRID = "#9aa5b1"
NEUTRAL = "#f7f3ec"
POSITIVE = "#2f6d54"
NEGATIVE = "#7f4d4d"
ALERT = "#c0392b"
A3_FOCUS_FILL = "#f4e8c9"
A3_FOCUS_TEXT = ARCH_EDGE["A3"]

# Figure 1 component colours inherit the simplest architecture that introduces
# each component: static context = A1, retrieval = A2, and tool calls = A3.
COMPONENT_STYLE = {
    "ctx": ("#efeeeb", ARCH_EDGE["A1"]),
    "ret": ("#e9edf4", ARCH_EDGE["A2"]),
    "tool": ("#f8f0dc", ARCH_EDGE["A3"]),
    "llm": ("#f4f4f4", AXIS),
    "io": ("#ffffff", "#9a9a9a"),
}

ACC_CMAP = LinearSegmentedColormap.from_list(
    "pexpo_seq",
    ["#fbf7ee", "#e8dcc0", "#b9c9a8", "#7aa88f", "#3d7d72", "#1f5148"],
    N=256,
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "pexpo_div",
    [NEGATIVE, "#cf8a7e", "#eec9b8", NEUTRAL, "#a9c9a9", "#6ba17f", POSITIVE],
    N=256,
)
ROSE_CMAP = LinearSegmentedColormap.from_list(
    "pexpo_rose",
    ["#fdf1f0", "#f5d8d6", "#dea3a2", "#b57979", NEGATIVE],
    N=256,
)
