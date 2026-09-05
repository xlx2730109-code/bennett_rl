# Copyright (c) 2026, Bennett. All rights reserved.

"""Compare the production DCMotor baseline with the DM-J8006-2EC envelope.

Runs without the simulation app (numpy + torch only).  Renders the four-quadrant
joint-side envelope of the training baseline (Isaac Lab DCMotor 8 / 20 / 19.8968)
against the datasheet envelope built from the digitized 24 V performance sweep,
plus every digitized sweep point and the manual anchor points.

Output: generated/dm_j8006_envelope_model_comparison.png (+ .svg) next to this
script.  ``--validate-only`` just prints the key numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

try:
    from bennett_rl.assets.motor.dm8006_envelope import (
        NO_LOAD_SPEED_RAD_S,
        PEAK_TORQUE_NM,
        RATED_SPEED_RAD_S,
        RATED_TORQUE_NM,
        build_envelope,
        dc_motor_envelope,
        load_sweep,
    )
except ModuleNotFoundError:
    # standalone run without the simulation env: the ``bennett_rl`` package
    # __init__ pulls in isaaclab (pxr), so load this sibling module by path.
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "dm8006_envelope", Path(__file__).resolve().parent / "dm8006_envelope.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    NO_LOAD_SPEED_RAD_S = _mod.NO_LOAD_SPEED_RAD_S
    PEAK_TORQUE_NM = _mod.PEAK_TORQUE_NM
    RATED_SPEED_RAD_S = _mod.RATED_SPEED_RAD_S
    RATED_TORQUE_NM = _mod.RATED_TORQUE_NM
    build_envelope = _mod.build_envelope
    dc_motor_envelope = _mod.dc_motor_envelope
    load_sweep = _mod.load_sweep

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "dm_j8006_envelope_model_comparison.png"

# palette tokens shared with scripts/analysis/plot_motor_report.py
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
BLUE = "#2a78d6"   # categorical slot 1: current training model
RED = "#d03b3b"    # status-critical: datasheet envelope / warning color
RPM = 60.0 / (2.0 * np.pi)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=10, length=4, width=1.0)


def report(w_grid, tau_dm) -> None:
    for name, w in [
        ("at 0 rad/s (stall)", 0.0),
        ("at rated 120 rpm", RATED_SPEED_RAD_S),
        ("at 15 rad/s", 15.0),
    ]:
        dm = float(np.interp(w, w_grid, tau_dm))
        dc = float(dc_motor_envelope(np.array([w]))[0])
        print(f"[ENVELOPE] tau {name}: damiao={dm:.2f}  dc_motor_baseline={dc:.2f} N-m")
    print(f"[ENVELOPE] no-load corner: {NO_LOAD_SPEED_RAD_S:.4f} rad/s (=190 rpm @ 24 V)")
    print("[ENVELOPE] dc_motor baseline: 8 N-m flat to 11.94 rad/s, then linear to 0 @ 19.90")


def plot(w_grid, tau_dm, sweep, output: Path):
    tau_dc = dc_motor_envelope(w_grid)
    w_max = NO_LOAD_SPEED_RAD_S * 1.08
    t_max = PEAK_TORQUE_NM * 1.12

    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    # datasheet envelope, four-quadrant symmetric (|tau| <= tau_max(|speed|))
    for k, (sgn_w, sgn_t) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        ax.plot(sgn_w * w_grid, sgn_t * tau_dm, color=RED, lw=2.0, ls=(0, (6, 3)), zorder=3,
                label="DM-J8006-2EC datasheet envelope (24 V sweep + manual anchors)" if k == 0 else None)
    # production training model, categorical blue
    for k, (sgn_w, sgn_t) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        ax.plot(sgn_w * w_grid, sgn_t * tau_dc, color=BLUE, lw=2.0, zorder=3,
                label="Training baseline: Isaac Lab DCMotor 8 / 20 / 19.90" if k == 0 else None)

    # every digitized sweep point (plateau rows included: they re-measure the knee)
    sw_w = sweep["speed_rpm"] * (2.0 * np.pi / 60.0)
    sw_t = sweep["torque_nm"]
    valid = sw_t > 0.0
    for k, (sgn_w, sgn_t) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        ax.scatter(sgn_w * sw_w[valid], sgn_t * sw_t[valid], s=26, facecolors="none",
                   edgecolors=INK2, linewidths=1.0, zorder=4,
                   label="Digitized 24 V sweep points" if k == 0 else None)

    # anchors
    ax.scatter([RATED_SPEED_RAD_S], [RATED_TORQUE_NM], marker="*", s=200, color=INK,
               zorder=5, label=f"Manual rated point: 8 N-m @ 120 rpm")
    ax.scatter([NO_LOAD_SPEED_RAD_S], [0.0], marker="o", s=60, facecolor=SURFACE,
               edgecolor=INK, zorder=5, label="Manual no-load: 190 rpm @ 24 V")

    style_axes(ax)
    ax.set_xlim(-w_max, w_max)
    ax.set_ylim(-t_max, t_max)
    ax.set_xticks(np.arange(-20.0, 20.1, 5.0))
    ax.set_yticks(np.arange(-20.0, 20.1, 5.0))
    ax.set_xlabel("Joint speed (rad/s)", fontsize=11.5, color=INK)
    ax.set_ylabel("Joint torque (N-m)", fontsize=11.5, color=INK)
    ax.axhline(0.0, color=GRID, linewidth=0.8, zorder=1)

    sec = ax.secondary_xaxis("top", functions=(lambda v: v * RPM, lambda r: r / RPM))
    sec.set_xlabel("Joint speed (rpm)", fontsize=11.5, color=INK)
    sec.tick_params(labelcolor=INK, color=INK, labelsize=10, length=4, width=1.0)
    sec.spines["top"].set_color(INK)
    sec.spines["top"].set_linewidth(1.0)
    # keep edge gridlines from painting over the secondary top spine (same fix
    # as scripts/analysis/plot_motor_report.py chart 05)
    ylim = ax.get_ylim()
    for gl, ypos in zip(ax.yaxis.get_gridlines(), ax.get_yticks()):
        if ypos >= ylim[1] - 1e-9 or ypos <= ylim[0] + 1e-9:
            gl.set_visible(False)

    ax.annotate("peak 20 N-m (stall,\nunvalidated by the sweep)", xy=(0.0, PEAK_TORQUE_NM),
                xytext=(1.6, 18.2), fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9})
    i_end = int(np.argmax(sw_t))
    ax.annotate(f"sweep end: {sw_t[i_end]:.0f} N-m @ {sweep['speed_rpm'][i_end]:.0f} rpm",
                xy=(sw_w[i_end], sw_t[i_end]), xytext=(2.6, 15.4),
                fontsize=8.5, color=INK, arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9})
    ax.annotate("current-limit clip (effort_limit)\nacts below this envelope", xy=(0.02, 0.03),
                xycoords="axes fraction", ha="left", va="bottom", fontsize=8.5, color=INK2)

    ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor=SURFACE,
              edgecolor="none", framealpha=0.9, labelcolor=INK)
    fig.suptitle("DM-J8006-2EC envelope: datasheet LUT vs training DCMotor baseline (joint side, 4-quadrant)",
                 color=INK, fontsize=13, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=SURFACE)
    fig.savefig(output.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    print(f"[OUTPUT] {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    w_grid, tau_dm = build_envelope()
    sweep = load_sweep()
    print(f"[SOURCE] digitized sweep loaded ({sweep['torque_nm'].size} points)")
    report(w_grid, tau_dm)
    if not args.validate_only:
        plot(w_grid, tau_dm, sweep, OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
