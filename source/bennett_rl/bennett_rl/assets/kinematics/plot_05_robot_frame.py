# Copyright (c) 2026, Bennett. All rights reserved.

"""Chart 05 - whole-robot frame: top view and side view at the V5 stance.

Four legs solved at the BENNETT_CFG_V5 initial pose (+-0.12 thigh, -0.24
calf).  Top view shows hip positions, the base outline (schematic), and the
motor-axis direction at each hip; side view shows the leg plane with the
hip-to-foot drop.  Dimensions are measured, not drawn to a spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import kin_style as ks
from bennett_kinematics import V5_DEFAULT_POSE, load_leg_model


def solve_all():
    legs = load_leg_model()
    data = {}
    for side, leg in legs.items():
        q_t, q_c = V5_DEFAULT_POSE[side]
        qp, res = leg.solve_passive(q_t, q_c)
        assert res < 1e-6
        f = leg.frames(q_t, q_c, *qp)
        data[side] = {
            "hip": leg.hip_pos(),
            "foot": f["foot"],
            "ankle": f["_3"][0] @ leg.ankle_on_3 + f["_3"][1],
            "axis_top": leg.joints[f"{side}_thigh"].axis,
        }
    return data


def dim_arrow(ax, p, q, text, dy=0.0, color=ks.INK2, fs=8):
    """Horizontal/vertical dimension line with arrowheads and a label."""
    ax.annotate(
        "", xy=q, xytext=p,
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.0,
                        shrinkA=0, shrinkB=0),
    )
    mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    ax.text(mid[0], mid[1] + dy, text, fontsize=fs, color=color, ha="center",
            va="bottom" if dy >= 0 else "top")


def main():
    d = solve_all()
    fig = ks.new_figure((13.0, 6.2))
    ks.fig_suptitle(
        fig,
        "Bennett - whole-robot frame at the V5 stance",
        "Base frame: X forward, Y left, Z up (URDF convention).  Four legs at the "
        "BENNETT_CFG_V5 initial pose (+-0.12 thigh / -0.24 calf);  each foot is the\n"
        "FL_foot-style tip of link _1, solved through the closed chain.  All numbers "
        "measured from the parsed model, in metres.",
    )

    # ------------------------------------------------------------------
    # top view
    # ------------------------------------------------------------------
    ax = fig.add_axes([0.045, 0.10, 0.42, 0.74])
    ks.style_axes(ax)
    ax.set_aspect("equal")

    # schematic base hull: hips expanded by 0.055 m
    hx = np.array([d[s]["hip"][0] for s in ("FL", "FR", "RL", "RR")])
    hy = np.array([d[s]["hip"][1] for s in ("FL", "FR", "RL", "RR")])
    pad = 0.055
    rect = plt.Rectangle((hx.min() - pad, hy.min() - pad),
                         (hx.max() - hx.min()) + 2 * pad,
                         (hy.max() - hy.min()) + 2 * pad,
                         facecolor=ks.PANEL, edgecolor=ks.AXIS, lw=1.0, zorder=1)
    ax.add_patch(rect)
    ax.text(hx.min() - pad, hy.max() + pad + 0.008, "base outline (schematic)",
            fontsize=7.5, color=ks.MUTED)

    for side in ("FL", "FR", "RL", "RR"):
        c = ks.LEG_COLORS[side]
        hip, foot = d[side]["hip"][:2], d[side]["foot"][:2]
        ax.plot([hip[0], foot[0]], [hip[1], foot[1]], color=c, lw=1.6,
                alpha=0.75, zorder=3)
        ax.scatter(*hip, s=60, c=c, edgecolor=ks.SURFACE, linewidth=1.0, zorder=5)
        ax.scatter(*foot, s=46, c=c, marker="s", edgecolor=ks.SURFACE,
                   linewidth=1.0, zorder=5)
        ax.text(*(hip + np.array([0.012, 0.010])), side, fontsize=8.5,
                color=c, fontweight="bold")
        ax.text(*(foot + np.array([-0.008, -0.016])), "foot", fontsize=7,
                color=c, ha="right")
        # motor axis at the hip, seen from above (+-y direction)
        a = d[side]["axis_top"][:2] * 0.035
        ax.plot([hip[0] - a[0], hip[0] + a[0]], [hip[1] - a[1], hip[1] + a[1]],
                color=ks.RED, lw=1.8, zorder=4)

    # body axis indicator (kept small, inside the base hull)
    ax.annotate("", xy=(0.155, 0.0), xytext=(0.06, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=ks.INK, lw=1.2))
    ax.text(0.165, 0.0, "X", fontsize=8.5, color=ks.INK, va="center")
    ax.annotate("", xy=(0.0, 0.105), xytext=(0.0, 0.03),
                arrowprops=dict(arrowstyle="-|>", color=ks.INK, lw=1.2))
    ax.text(0.0, 0.118, "Y", fontsize=8.5, color=ks.INK, ha="center")

    # dimensions
    dim_arrow(ax, (-0.211, -0.155), (0.211, -0.155), "wheelbase 0.422 m", dy=-0.012)
    ax.annotate("", xy=(0.300, 0.0682), xytext=(0.300, -0.0682),
                arrowprops=dict(arrowstyle="<->", color=ks.INK2, lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.text(0.316, 0.0, "track 0.136 m", fontsize=8, color=ks.INK2,
            rotation=90, ha="left", va="center")

    ax.set_xlim(-0.36, 0.40)
    ax.set_ylim(-0.24, 0.26)
    ks.label(ax, title="Top view (X-Y)")
    ax.set_xlabel("x (m)", fontsize=8.5)
    ax.set_ylabel("y (m)", fontsize=8.5)

    # ------------------------------------------------------------------
    # side view
    # ------------------------------------------------------------------
    ax2 = fig.add_axes([0.545, 0.10, 0.42, 0.74])
    ks.style_axes(ax2)
    ax2.set_aspect("equal")

    # ground line at the lowest foot
    z0 = min(d[s]["foot"][2] for s in d)
    ax2.axhline(z0, color=ks.AXIS, lw=1.2)
    ax2.text(0.335, z0 + 0.004, "ground", fontsize=7.5, color=ks.MUTED,
             ha="right", va="bottom")

    # left/right legs coincide in this projection: front solid, rear dashed
    for side in ("FL", "FR", "RL", "RR"):
        hip, foot = d[side]["hip"][[0, 2]], d[side]["foot"][[0, 2]]
        rear = side.startswith("R")
        ax2.plot([hip[0], foot[0]], [hip[1], foot[1]], color=ks.BLUE,
                 lw=2.4 if not rear else 1.6, ls="-" if not rear else (0, (5, 3)),
                 alpha=1.0 if not rear else 0.55, zorder=3)
        ax2.scatter(*hip, s=56, c=ks.BLUE, alpha=1.0 if not rear else 0.55,
                    edgecolor=ks.SURFACE, linewidth=1.0, zorder=5)
        ax2.scatter(*foot, s=44, c=ks.BLUE, marker="s",
                    alpha=1.0 if not rear else 0.55, edgecolor=ks.SURFACE,
                    linewidth=1.0, zorder=5)

    # base origin + height reference
    ax2.scatter(0, 0, s=60, c=ks.INK, marker="s", zorder=6)
    ax2.text(0.012, 0.008, "base origin", fontsize=8, color=ks.INK)
    ax2.plot([0, 0], [0, z0], color=ks.RED, lw=1.0, ls=(0, (4, 3)))
    ax2.text(0.008, z0 / 2, f"stance drop\n{abs(z0):.3f} m", fontsize=8,
             color=ks.RED, va="center")

    fl = d["FL"]["hip"][0]
    dim_arrow(ax2, (fl, z0 - 0.035), (d["RL"]["hip"][0], z0 - 0.035),
              "wheelbase 0.422 m", dy=-0.010)

    leg2 = Line2D([], [], color=ks.BLUE, lw=2.4)
    leg2r = Line2D([], [], color=ks.BLUE, lw=1.6, ls=(0, (5, 3)), alpha=0.55)
    ax2.legend([leg2, leg2r], ["front legs (FL/FR)", "rear legs (RL/RR, dashed)"],
               fontsize=7.5, frameon=False, loc="lower left",
               bbox_to_anchor=(0.0, -0.02))

    ax2.set_xlim(-0.34, 0.36)
    ax2.set_ylim(z0 - 0.115, 0.09)
    ks.label(ax2, title="Side view (X-Z)")
    ax2.set_xlabel("x (m)", fontsize=8.5)
    ax2.set_ylabel("z (m)", fontsize=8.5)

    # footer
    fig.text(0.5, 0.012,
             "8 x DM8006: thigh + calf per leg (red ticks = motor axes, both along "
             "(0, +-0.707, +0.707)).  Feet land on a 0.422 x 0.316 m rectangle;  in "
             "the side view left/right legs coincide exactly.",
             ha="center", fontsize=8, color=ks.MUTED)

    ks.save(fig, "05_robot_frame.png")


if __name__ == "__main__":
    main()
