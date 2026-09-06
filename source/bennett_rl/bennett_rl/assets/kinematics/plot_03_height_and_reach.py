# Copyright (c) 2026, Bennett. All rights reserved.

"""Chart 03 - foot height / reach / lateral offset maps over the joint grid.

Three heatmaps of the same solved grid (q_thigh x q_calf, FL leg):
foot height below the base origin, radial reach from the hip axis, and the
lateral (y) offset produced by the tilted axes.  The V5 stance and the zero
pose are marked on every panel.
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

PANELS = [
    ("foot_z", "Foot height below base origin (m)", "heights"),
    ("reach", "Hip-to-foot radial reach (m)", "reach"),
    ("foot_y", "Lateral offset y (m)", "lateral"),
]


def main():
    legs = load_leg_model()
    leg = legs["FL"]
    sw = sweep_workspace(leg, QT, QC)
    hip = leg.hip_pos()
    Z = sw["foot_z"]
    reach = np.hypot(sw["foot_x"] - hip[0], sw["foot_z"] - hip[2])

    q_t, q_c = V5_DEFAULT_POSE["FL"]
    ti = int(np.argmin(np.abs(QT - q_t)))
    ci = int(np.argmin(np.abs(QC - q_c)))
    zi = (int(np.argmin(np.abs(QT))), int(np.argmin(np.abs(QC))))

    print(f"V5 stance: height {Z[ti, ci]:.4f} m, reach {reach[ti, ci]:.4f} m, "
          f"y {sw['foot_y'][ti, ci]:.4f} m")
    print(f"zero pose: height {Z[zi]:.4f} m, reach {reach[zi]:.4f} m, "
          f"y {sw['foot_y'][zi]:.4f} m")
    print(f"height range [{np.nanmin(Z):.4f}, {np.nanmax(Z):.4f}] m, "
          f"reach range [{np.nanmin(reach):.4f}, {np.nanmax(reach):.4f}] m")

    fig = ks.new_figure((13.2, 5.2))
    ks.fig_suptitle(
        fig,
        "FL foot placement maps over the commanded joint grid",
        "Same solved 81 x 81 grid as the workspace chart.  Red star = V5 stance "
        "(+0.12, -0.24);  white dot = zero pose.  Axes are the commanded angles in rad.",
    )

    QCa, QTh = np.meshgrid(QC, QT)
    fields = {"foot_z": Z, "reach": reach, "foot_y": sw["foot_y"]}

    for k, (key, ttl, _) in enumerate(PANELS):
        ax = fig.add_axes([0.055 + k * 0.315, 0.14, 0.245, 0.64])
        ks.style_axes(ax)
        C = fields[key]
        h = ax.pcolormesh(QCa, QTh, C, cmap=ks.SEQ_CMAP, shading="nearest")
        cb = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7, colors=ks.INK2)
        cb.outline.set_edgecolor(ks.AXIS)
        ax.scatter(QC[ci], QT[ti], s=150, marker="*", c=ks.RED,
                   edgecolor=ks.SURFACE, linewidth=0.8, zorder=6)
        ax.scatter(0, 0, s=40, facecolor="none", edgecolor=ks.SURFACE,
                   linewidth=1.6, zorder=6)
        ax.set_xlabel("q_calf (rad)", fontsize=8.5)
        ax.set_ylabel("q_thigh (rad)", fontsize=8.5)
        ks.label(ax, title=ttl)

    ks.save(fig, "03_height_and_reach.png")


if __name__ == "__main__":
    main()
