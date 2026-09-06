# Copyright (c) 2026, Bennett. All rights reserved.

"""Chart 02 - foot-tip workspace of one leg (FL) over the commanded envelope.

Every point is a solved pose: q_thigh in [-0.8, 0.8], q_calf in [-0.9, 0.55]
(the task-side joint-target limits), passive angles from the closed-chain
solve.  Left: sagittal (x-z) cloud coloured by knee bend; right: frontal
(y-z) view showing the sideways offset the tilted axes produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import kin_style as ks
from bennett_kinematics import LEG_JOINT_LIMITS, V5_DEFAULT_POSE, load_leg_model, sweep_workspace

QT = np.linspace(*LEG_JOINT_LIMITS["thigh"], 81)
QC = np.linspace(*LEG_JOINT_LIMITS["calf"], 81)


def main():
    legs = load_leg_model()
    leg = legs["FL"]
    sw = sweep_workspace(leg, QT, QC)

    X, Z, Y = sw["foot_x"], sw["foot_z"], sw["foot_y"]
    hip = leg.hip_pos()
    q_t, q_c = V5_DEFAULT_POSE["FL"]
    qp, _ = leg.solve_passive(q_t, q_c)
    foot0 = leg.frames(q_t, q_c, *qp)["foot"]

    reach = np.hypot(X - hip[0], Z - hip[2])
    i_max = np.unravel_index(np.nanargmax(reach), reach.shape)
    i_low = np.unravel_index(np.nanargmin(Z), Z.shape)
    i_high = np.unravel_index(np.nanargmax(Z), Z.shape)

    print(f"reach max  {reach[i_max]:.4f} m at q=({QT[i_max[0]]:+.2f},{QC[i_max[1]]:+.2f})")
    print(f"foot z min {Z[i_low]:.4f} m at q=({QT[i_low[0]]:+.2f},{QC[i_low[1]]:+.2f})")
    print(f"foot z max {Z[i_high]:.4f} m at q=({QT[i_high[0]]:+.2f},{QC[i_high[1]]:+.2f})")
    print(f"y range    [{np.nanmin(Y):.4f}, {np.nanmax(Y):.4f}] m")

    fig = ks.new_figure((12.8, 5.8))
    ks.fig_suptitle(
        fig,
        "FL foot-tip workspace over the commanded joint envelope",
        "81 x 81 solved poses, q_thigh in [-0.80, 0.80] rad, q_calf in [-0.90, +0.55] rad "
        "(the tasks' joint-target clamp).  Colour = knee bend q_calf;\n"
        "closed-chain passive angles solved at every point.  Red dot = V5 stance "
        "(+0.12, -0.24);  black square = hip axis crossing.",
    )

    QCa, QTh = np.meshgrid(QC, QT)

    # ------------------------------------------------------------------
    # sagittal view
    # ------------------------------------------------------------------
    ax = fig.add_axes([0.07, 0.13, 0.38, 0.66])
    ks.style_axes(ax)
    ax.set_aspect("equal")
    h = ax.scatter(X, Z, c=QCa, cmap=ks.SEQ_CMAP, s=7, linewidths=0)
    cb = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("q_calf (rad)", fontsize=8, color=ks.INK2)
    cb.ax.tick_params(labelsize=7, colors=ks.INK2)
    cb.outline.set_edgecolor(ks.AXIS)

    ax.scatter(*hip[[0, 2]], s=70, c=ks.INK, marker="s", zorder=6)
    ax.text(*(hip[[0, 2]] + [-0.010, 0.004]), "hip", fontsize=8, color=ks.INK,
            ha="right")
    ax.scatter(*foot0[[0, 2]], s=90, c=ks.RED, zorder=6, edgecolor=ks.SURFACE,
               linewidth=1.0)
    ax.text(*(foot0[[0, 2]] + [0.016, -0.046]), "V5 stance", fontsize=8, color=ks.RED)
    ax.annotate(
        f"max reach {reach[i_max]:.3f} m\nat q=({QT[i_max[0]]:+.2f}, {QC[i_max[1]]:+.2f})",
        xy=(X[i_max], Z[i_max]), xytext=(-0.145, 0.018),
        fontsize=8, color=ks.INK2,
        arrowprops=dict(arrowstyle="-", color=ks.INK2, lw=0.8),
    )
    ax.annotate(
        f"lowest {Z[i_low]:.3f} m\nat q=({QT[i_low[0]]:+.2f}, {QC[i_low[1]]:+.2f})",
        xy=(X[i_low], Z[i_low]), xytext=(-0.14, -0.325),
        fontsize=8, color=ks.INK2,
        arrowprops=dict(arrowstyle="-", color=ks.INK2, lw=0.8),
    )
    ax.set_xlim(-0.17, 0.44)
    ax.set_ylim(-0.34, -0.01)
    ax.set_xlabel("x, forward from base origin (m)", fontsize=8.5)
    ax.set_ylabel("z, up (m)", fontsize=8.5)
    ks.label(ax, title="Sagittal view (x-z)")

    # ------------------------------------------------------------------
    # frontal view
    # ------------------------------------------------------------------
    ax2 = fig.add_axes([0.535, 0.13, 0.30, 0.66])
    ks.style_axes(ax2)
    ax2.set_aspect("equal")
    h2 = ax2.scatter(Y, Z, c=QCa, cmap=ks.SEQ_CMAP, s=7, linewidths=0)
    cb2 = fig.colorbar(h2, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label("q_calf (rad)", fontsize=8, color=ks.INK2)
    cb2.ax.tick_params(labelsize=7, colors=ks.INK2)
    cb2.outline.set_edgecolor(ks.AXIS)

    ax2.scatter(*hip[[1, 2]], s=70, c=ks.INK, marker="s", zorder=6)
    ax2.text(*(hip[[1, 2]] + [0.006, 0.010]), "hip", fontsize=8, color=ks.INK)
    ax2.scatter(*foot0[[1, 2]], s=90, c=ks.RED, zorder=6, edgecolor=ks.SURFACE,
                linewidth=1.0)
    dy = Y - hip[1]
    print(f"lateral offset range [{np.nanmin(dy):.4f}, {np.nanmax(dy):.4f}] m")
    iy = np.unravel_index(np.nanargmin(Y), Y.shape)
    ax2.annotate(
        f"lateral sweep\n{np.nanmax(dy) - np.nanmin(dy):.3f} m",
        xy=(Y[iy], Z[iy]), xytext=(0.095, -0.062),
        fontsize=8, color=ks.INK2,
        arrowprops=dict(arrowstyle="-", color=ks.INK2, lw=0.8),
    )
    ax2.set_xlabel("y, left (m)", fontsize=8.5)
    ax2.set_ylabel("z, up (m)", fontsize=8.5)
    ks.label(ax2, title="Frontal view (y-z)")

    # side note panel
    ax3 = fig.add_axes([0.885, 0.13, 0.105, 0.66])
    ax3.set_axis_off()
    notes = [
        "READING",
        "",
        "Knee bend (q_calf)",
        "maps near-linearly",
        "to foot height;",
        "thigh angle sweeps",
        "the fore-aft arc.",
        "",
        "The frontal spread",
        "is the tilted-axis",
        "(0, .707, .707)",
        "signature - the leg",
        "is not a planar",
        "two-link arm.",
    ]
    y = 0.97
    for i, line in enumerate(notes):
        ax3.text(0, y, line, fontsize=8 if i else 8.5,
                 color=ks.INK if i == 0 else ks.INK2,
                 fontweight="bold" if i == 0 else "normal", va="top")
        y -= 0.052

    ks.save(fig, "02_foot_workspace.png")


if __name__ == "__main__":
    main()
