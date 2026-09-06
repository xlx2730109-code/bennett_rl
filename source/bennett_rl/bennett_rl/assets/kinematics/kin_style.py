# Copyright (c) 2026, Bennett. All rights reserved.

"""Shared matplotlib styling for the kinematics chart set.

Same design tokens as scripts/analysis/plot_motor_report.py so the whole
Bennett report family reads as one system: off-white surface, ink text,
recessive warm-grey grid, blue for data, red reserved for limits/flags.
Every figure is saved as a 300-dpi PNG plus an editable-text SVG pair into
``generated/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"

OUT_DIR = Path(__file__).resolve().parent / "generated"

# -- tokens (mirror of plot_motor_report.py) --------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # categorical slot 1 / primary data
RED = "#d03b3b"      # status-critical (limits, closure-error flags)
PANEL = "#f2f1ec"    # inset / table ground

# categorical identity, same slot order as the motor report
LEG_COLORS = {"FL": "#2a78d6", "FR": "#eb6834", "RL": "#1baf7a", "RR": "#e87ba4"}

# single-hue blue ramp for magnitude (identical ramp to SEQ_CMAP in the report)
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def new_figure(figsize):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig


def style_axes(ax, x_grid=True, y_grid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    if x_grid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    if y_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def label(ax=None, *, xlabel=None, ylabel=None, title=None, subtitle=None):
    ax = ax or plt.gca()
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, pad=6, fontweight="bold")
    if subtitle:
        ax.text(0.0, 1.045, subtitle, transform=ax.transAxes,
                fontsize=8, color=INK2, va="bottom")


def fig_suptitle(fig, title, subtitle=None):
    """Bold title at the very top; optional (possibly multi-line) subtitle below."""
    h = fig.get_size_inches()[1]
    fig.suptitle(title, color=INK, fontsize=13.5, fontweight="bold", y=1 - 0.10 / h)
    if subtitle:
        lines = subtitle.split("\n")
        y = 1 - 0.55 / h
        for line in lines:
            fig.text(0.5, y, line, ha="center", color=INK2, fontsize=8.5)
            y -= 0.135 / h


def save(fig, name: str):
    """Write <stem>.png (300 dpi) + <stem>.svg pair into generated/."""
    OUT_DIR.mkdir(exist_ok=True)
    stem = name.rsplit(".", 1)[0]
    fig.savefig(OUT_DIR / f"{stem}.svg", facecolor=SURFACE)
    fig.savefig(OUT_DIR / name, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {stem}.png + {stem}.svg")


# --------------------------------------------------------------------------
# 3-D helpers (topology / workspace figures)
# --------------------------------------------------------------------------
def style_axes3d(ax):
    ax.set_facecolor(SURFACE)
    for pane_name in ax.panes:
        ax.panes[pane_name].set_visible(False)
    ax.xaxis.set_pane_color((SURFACE, 0.0))
    ax.yaxis.set_pane_color((SURFACE, 0.0))
    ax.zaxis.set_pane_color((SURFACE, 0.0))
    ax.grid(False)
    ax.tick_params(colors=INK2, labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        for lbl in axis.get_ticklabels():
            lbl.set_color(INK2)


def set_equal_aspect_3d(ax, zoom=1.0):
    """Equal-scale the three axes around the data midpoints (mpl <3.8-safe).

    ``zoom`` > 1 magnifies the drawn content inside the axes box.
    """
    mins, maxs = [], []
    for axis in (ax.get_xlim3d, ax.get_ylim3d, ax.get_zlim3d):
        lo, hi = axis()
        mins.append(lo)
        maxs.append(hi)
    centre = (np.array(mins) + np.array(maxs)) / 2
    half = (np.array(maxs) - np.array(mins)).max() / 2 * 1.05
    ax.set_xlim3d(centre[0] - half, centre[0] + half)
    ax.set_ylim3d(centre[1] - half, centre[1] + half)
    ax.set_zlim3d(centre[2] - half, centre[2] + half)
    if zoom != 1.0:
        try:
            ax.set_box_aspect((1, 1, 1), zoom=zoom)
        except TypeError:  # older mpl without zoom support
            pass


def draw_joint_axis(ax, origin, axis, length=0.05, color=RED, label_txt=None):
    """Double-headed arrow along a revolute axis + optional tag."""
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u) * length / 2
    a, b = np.asarray(origin) - u, np.asarray(origin) + u
    ax.quiver(*a, *(b - a), color=color, lw=1.6, arrow_length_ratio=0.18,
              pivot="middle")
    if label_txt:
        ax.text(*(b + 0.012 * u / max(np.linalg.norm(u), 1e-9)), label_txt,
                fontsize=7, color=color)
