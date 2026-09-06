# Copyright (c) 2026, Bennett. All rights reserved.

"""Chart 01 - Bennett leg linkage topology.

One large 3-D skeleton (FL leg, V5 stance) with every joint labelled and its
rotation axis drawn, next to an information panel (joint table, link masses,
closure note).  Geometry is read live from Urdf_Bennett_3.urdf; the passive
angles come from the Gauss-Newton ankle-pin solve.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python plot_01_...py` directly
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import kin_style as ks
from bennett_kinematics import V5_DEFAULT_POSE, load_leg_model

GRAY = "#898781"

JOINT_TABLE = [
    # label, type, axis (URDF), drive
    ("FL_thigh", "revolute", "(0, +0.707, +0.707)", "DM8006 motor"),
    ("FL_calf", "revolute", "(0, +0.707, +0.707)", "DM8006 motor"),
    ("FL_1", "revolute", "(+0.836, +0.512, +0.195)", "passive"),
    ("FL_2", "revolute", "(0, -0.998, +0.057)", "passive"),
    ("FL_3", "revolute", "(-0.836, +0.512, +0.195)", "passive"),
    ("FL_foot", "fixed", "-", "foot tip (body-fixed)"),
]

MASS_TABLE = [  # kg, Urdf_Bennett_3 inertials
    ("base", "4.221"), ("thigh", "0.994"), ("calf", "0.310"),
    ("_1", "0.162"), ("_2", "0.191"), ("_3", "0.191"), ("foot", "0.072"),
]


def main():
    legs = load_leg_model()
    leg = legs["FL"]
    q_t, q_c = V5_DEFAULT_POSE["FL"]

    qp, res = leg.solve_passive(q_t, q_c)
    assert res < 1e-6, f"closure failed ({res:.1e} m)"
    f = leg.frames(q_t, q_c, *qp)

    hip, knee = f["thigh"][1], f["calf"][1]
    p1, p2, p3 = f["_1"][1], f["_2"][1], f["_3"][1]
    ankle = f["_3"][0] @ leg.ankle_on_3 + f["_3"][1]
    foot = f["foot"]

    print("key points (base frame, m):")
    for nm, p in (("hip", hip), ("knee", knee), ("_1", p1), ("_2", p2),
                  ("_3", p3), ("ankle", ankle), ("foot", foot)):
        print(f"  {nm:6s} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    print(f"  passive q1/q2/q3 = {np.round(qp, 4)}  residual {res:.1e} m")
    print(f"  hip->foot drop   = {hip[2] - foot[2]:.4f} m")

    fig = ks.new_figure((13.2, 7.4))
    ks.fig_suptitle(
        fig,
        "Bennett closed-chain leg - linkage topology (front-left, V5 stance)",
        "URDF tree: base - thigh - {calf - _3,  _1 - {_2, foot}}.  The hardware closes the "
        "loop with the ankle pin: _3 tip meets _2 tip.\nGeometry parsed live from "
        "Urdf_Bennett_3.urdf; passive angles solved to < 1 um residual.",
    )

    # ------------------------------------------------------------------
    # left: large 3-D skeleton
    # ------------------------------------------------------------------
    ax = fig.add_axes([0.015, 0.01, 0.60, 0.845], projection="3d")
    ax.set_axis_off()
    ax.set_proj_type("ortho")

    def seg(a, b, color, lw):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=color, lw=lw, solid_capstyle="round")

    # rigid bodies: driven links blue, passive links grey
    seg((0, 0, 0), hip, ks.INK2, 2.4)          # base mount
    seg(hip, knee, ks.BLUE, 5.0)               # thigh body
    seg(hip, p1, ks.BLUE, 5.0)
    seg(knee, p3, ks.BLUE, 5.0)                # calf body
    seg(p1, p2, GRAY, 3.6)                     # _1 body
    seg(p1, foot, GRAY, 3.6)
    seg(p2, ankle, GRAY, 3.0)                  # _2 body
    seg(p3, ankle, GRAY, 3.0)                  # _3 body

    # joints: blue motor dots w/ red axes, grey passive dots w/ grey axes
    for pos, axis, Rpar, motor in (
        (hip, leg.joints["FL_thigh"].axis, f["thigh"][0], True),
        (knee, leg.joints["FL_calf"].axis, f["thigh"][0], True),
        (p1, leg.joints["FL_1"].axis, f["thigh"][0], False),
        (p2, leg.joints["FL_2"].axis, f["_1"][0], False),
        (p3, leg.joints["FL_3"].axis, f["calf"][0], False),
    ):
        ax.scatter(*pos, s=95 if motor else 55, c=ks.BLUE if motor else GRAY,
                   edgecolor=ks.SURFACE, linewidth=1.2, depthshade=False, zorder=6)
        ks.draw_joint_axis(ax, pos, Rpar @ axis, length=0.052 if motor else 0.045,
                           color=ks.RED if motor else GRAY)

    ax.scatter(*ankle, s=340, marker="*", facecolor="none", edgecolor=ks.RED,
               linewidth=2.0, depthshade=False, zorder=7)
    ax.scatter(*foot, s=90, c=ks.RED, edgecolor=ks.SURFACE, linewidth=1.2,
               depthshade=False, zorder=7)
    ax.scatter(0, 0, 0, s=70, c=ks.INK, marker="s", depthshade=False, zorder=7)

    # labels (offsets hand-tuned in data coords, metres)
    ax.text(*(hip + [-0.030, 0, 0.000]), "hip - thigh joint (DM8006)",
            fontsize=9, color=ks.INK, ha="right", zorder=8)
    ax.text(*(knee + [0.010, 0, 0.014]), "knee - calf joint (DM8006)",
            fontsize=9, color=ks.INK, ha="left", zorder=8)
    ax.text(*(p1 + [-0.010, 0, -0.012]), "_1 joint (passive)", fontsize=8.5,
            color=ks.INK2, ha="right", zorder=8)
    ax.text(*(p2 + [0.020, 0, -0.010]), "_2 joint (passive)", fontsize=8.5,
            color=ks.INK2, ha="left", zorder=8)
    ax.text(*(p3 + [0.012, 0, 0.006]), "_3 joint (passive)", fontsize=8.5,
            color=ks.INK2, ha="left", zorder=8)
    ax.text(*(ankle + [-0.018, 0, 0.010]), "ankle pin  (_3 tip = _2 tip)",
            fontsize=9, color=ks.RED, ha="right", zorder=8)
    ax.text(*(foot + [0.008, 0, -0.016]), "foot tip", fontsize=9, color=ks.RED,
            zorder=8)
    ax.text(*(np.array([0.0, 0, 0]) + [0.004, 0, 0.014]), "base origin",
            fontsize=7.5, color=ks.INK2, zorder=8)

    # link name tag (thigh is already named by the hip joint label)
    ax.text(*((knee + p3) / 2 + [0.016, 0, 0.010]), "calf", fontsize=8.5,
            fontweight="bold", color=ks.BLUE, zorder=8)

    # light drop line: hip -> foot vertical reference
    ax.plot([hip[0], foot[0]], [hip[1], foot[1]], [hip[2], foot[2]],
            color=ks.RED, lw=1.0, ls=(0, (4, 3)), alpha=0.75)

    allpts = np.vstack([[0, 0, 0], hip, knee, p1, p2, p3, ankle, foot])
    lo_pt, hi_pt = allpts.min(0), allpts.max(0)
    pad = 0.035
    ax.set_xlim(lo_pt[0] - pad, hi_pt[0] + pad)
    ax.set_ylim(lo_pt[1] - pad, hi_pt[1] + pad)
    ax.set_zlim(lo_pt[2] - pad, hi_pt[2] + pad)
    ks.set_equal_aspect_3d(ax, zoom=1.35)
    ax.view_init(elev=16, azim=-72)

    # corner axes indicator (X fwd, Y left, Z up) from the bounding-box corner
    o = lo_pt.copy()
    for vec, name in (([1, 0, 0], "X"), ([0, 1, 0], "Y"), ([0, 0, 1], "Z")):
        v = np.array(vec, dtype=float) * 0.075
        ax.quiver(*o, *v, color=ks.INK2, lw=1.4, arrow_length_ratio=0.22)
        ax.text(*(o + v * 1.30), name, fontsize=8, color=ks.INK2, ha="center")

    # ------------------------------------------------------------------
    # right: information panel
    # ------------------------------------------------------------------
    px = fig.add_axes([0.635, 0.01, 0.355, 0.845])
    px.set_axis_off()

    y = 0.97
    px.text(0, y, "JOINTS  (FL leg, Urdf_Bennett_3)", fontsize=9.5,
            fontweight="bold", color=ks.INK)
    y -= 0.040
    cols = (0.0, 0.155, 0.42, 0.80)
    for cx, h in zip(cols, ("joint", "type", "axis (URDF)", "drive")):
        px.text(cx, y, h, fontsize=8, color=ks.INK2)
    y -= 0.014
    px.plot([0, 1], [y, y], color=ks.GRID, lw=1.0)
    for name, typ, axis, drive in JOINT_TABLE:
        y -= 0.036
        c = ks.BLUE if "motor" in drive else (ks.RED if "foot" in drive else GRAY)
        px.text(cols[0], y, name, fontsize=8.5, color=ks.INK)
        px.text(cols[1], y, typ, fontsize=8, color=ks.INK2)
        px.text(cols[2], y, axis, fontsize=8, color=ks.INK2)
        px.text(cols[3], y, drive, fontsize=8, color=c)

    y -= 0.050
    px.text(0, y, "LINK MASSES  (kg, Urdf_Bennett_3 inertials)", fontsize=9.5,
            fontweight="bold", color=ks.INK)
    y -= 0.036
    px.text(0, y, "base 4.221 | thigh 0.994 | calf 0.310 | _1 0.162",
            fontsize=7.8, color=ks.INK2)
    y -= 0.032
    px.text(0, y, "_2 0.191 | _3 0.191 | foot 0.072   (leg total 1.921)",
            fontsize=7.8, color=ks.INK2)

    y -= 0.050
    px.text(0, y, "KEY NUMBERS  (V5 stance, q = +0.12 / -0.24)", fontsize=9.5,
            fontweight="bold", color=ks.INK)
    key = [
        ("passive q1 / q2 / q3", f"{qp[0]:+.3f} / {qp[1]:+.3f} / {qp[2]:+.3f} rad"),
        ("closure residual", f"{res:.1e} m"),
        ("hip-to-foot drop", f"{hip[2] - foot[2]:.3f} m"),
        ("foot tip (base frame)",
         f"({foot[0]:+.3f}, {foot[1]:+.3f}, {foot[2]:+.3f}) m"),
    ]
    for nm, v in key:
        y -= 0.036
        px.text(0, y, nm, fontsize=8.5, color=ks.INK2)
        px.text(1, y, v, fontsize=8.5, color=ks.INK, ha="right")

    y -= 0.048
    px.plot([0, 1], [y, y], color=ks.GRID, lw=1.0)
    notes = [
        "Driven: thigh + calf only (8 DM8006 motors on the robot).  The",
        "parallelogram four-bar (_1, _2, _3) keeps the foot branch oriented;",
        "its single closure is the ankle pin, solved as a 3-eq Gauss-Newton",
        "on the passive angles (q1, q2, q3).",
    ]
    for line in notes:
        y -= 0.026
        px.text(0, y, line, fontsize=8, color=ks.INK2)

    y -= 0.044
    px.text(0, y, "LEGEND", fontsize=9.5, fontweight="bold", color=ks.INK)
    chips = [
        (ks.BLUE, "o", "driven joint (DM8006)"),
        (GRAY, "o", "passive four-bar joint"),
        (ks.RED, "*", "ankle pin = closure constraint"),
    ]
    for c, mk, txt in chips:
        y -= 0.032
        px.scatter([0.010], [y], s=46 if mk == "o" else 150, marker=mk,
                   c=c if mk == "o" else "none", edgecolor=c, linewidth=1.4,
                   clip_on=False)
        px.text(0.05, y, txt, fontsize=8.5, color=ks.INK2, va="center")

    ks.save(fig, "01_linkage_topology.png")


if __name__ == "__main__":
    main()
