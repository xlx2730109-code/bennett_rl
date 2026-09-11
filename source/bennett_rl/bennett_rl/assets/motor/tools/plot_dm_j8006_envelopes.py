# Copyright (c) 2026, Bennett. All rights reserved.

"""Compare the production DCMotor baseline with the DM-J8006-2EC envelope.

Runs without the simulation app (numpy + torch only).  Reads the digitized
24 V sweep CSV and renders the manufacturer's chart -- torque on the
horizontal axis, one performance metric per row (speed / efficiency /
power) -- next to the max torque-speed envelope of the datasheet LUT
against the training baseline (Isaac Lab DCMotor 8 / 20 / 19.8968).

Output: ``generated/`` next to this script (tools/), i.e.
``dm_j8006_envelope_model_comparison.png`` (+ .svg).  ``--validate-only`` just
prints the key numbers.
"""

from __future__ import annotations

import argparse
import csv
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
    # __init__ pulls in isaaclab (pxr), so load the envelope core by path --
    # this script lives in tools/, one level below the motor package root.
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "dm8006_envelope", Path(__file__).resolve().parent.parent / "dm8006_envelope.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CURVE_CSV = _mod.CURVE_CSV
    NO_LOAD_SPEED_RAD_S = _mod.NO_LOAD_SPEED_RAD_S
    PEAK_TORQUE_NM = _mod.PEAK_TORQUE_NM
    RATED_SPEED_RAD_S = _mod.RATED_SPEED_RAD_S
    RATED_TORQUE_NM = _mod.RATED_TORQUE_NM
    build_envelope = _mod.build_envelope
    dc_motor_envelope = _mod.dc_motor_envelope
    load_sweep = _mod.load_sweep

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated" / "dm_j8006_envelope_model_comparison.png"

# palette tokens shared with scripts/analysis/plot_motor_report.py; the left
# column reuses the OFFICIAL datasheet curve colors so the reproduction can be
# checked against docs/ "24V 120RPM 8006电机性能曲线图.png" line by line
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
BLUE = "#2a78d6"   # training baseline: DCMotor (right panel only)
RED = "#d03b3b"    # datasheet envelope + official Eff(%) curve color
NAVY = "#1c4587"   # official Speed(rpm) curve color (deep blue)
MOSS = "#aeb500"   # official Pout(W) curve color (yellow-leaning green)
PURPLE = "#9a3fc2" # official Pin(W) curve color
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
    """Manufacturer's reading direction: TORQUE on the horizontal axis.

    Left column reproduces the official datasheet chart from the digitized
    CSV (speed / efficiency / power vs torque, one metric per row).  The
    right panel keeps the same torque axis and compares the max
    torque-speed envelope with the training DCMotor baseline.
    """
    with Path(CURVE_CSV).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    tq = np.array([float(r["torque_nm"]) for r in rows])
    rpm = np.array([float(r["speed_rpm"]) for r in rows])
    eff = np.array([float(r["efficiency_pct"]) for r in rows])
    pin = np.array([float(r["input_power_w"]) for r in rows])
    pout = np.array([float(r["output_power_w"]) for r in rows])

    fig = plt.figure(figsize=(14.0, 8.0), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_env = fig.add_subplot(gs[:, 1])

    # -- left column: the datasheet chart, one metric per row -------------
    # colors follow the official chart: Speed deep blue, Eff red,
    # Pout yellow-green, Pin purple
    metrics = [
        (rpm, NAVY, "Speed (rpm)", None),
        (eff, RED, "Efficiency (%)", None),
        (pout, MOSS, "Output power (W)", (pin, PURPLE, "input power (W)")),
    ]
    for i, (ax, (y, color, ylab, extra)) in enumerate(zip(axes, metrics)):
        ax.plot(tq, y, color=color, lw=1.6, zorder=3)
        ax.scatter(tq, y, s=15, color=color, zorder=4)
        if extra is not None:
            ax.plot(tq, extra[0], color=extra[1], lw=1.1, ls=(0, (4, 2)), zorder=2)
            ax.text(0.985, 0.92, extra[2], transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.0, color=extra[1])
        ax.set_ylim(*{0: (0.0, 145.0), 1: (0.0, 80.0), 2: (0.0, 340.0)}[i])
        ax.set_xlim(0.0, 13.8)
        ax.set_xticks(np.arange(0.0, 13.1, 2.0))
        ax.set_ylabel(ylab, fontsize=10.5, color=color)
        style_axes(ax)
        if i < 2:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Torque (N-m)", fontsize=11.0, color=INK)

    # -- right panel: max envelope vs baseline, torque on X ---------------
    # invert the (speed -> max torque) LUT: max speed available at each torque
    tau_hi = np.linspace(0.0, PEAK_TORQUE_NM, 241)
    omega_hi_rpm = np.interp(tau_hi, tau_dm[::-1], w_grid[::-1]) * RPM
    ax_env.plot(tau_hi, omega_hi_rpm, color=RED, lw=2.4, zorder=3,
                label="DM-J8006-2EC datasheet envelope")
    falling = sweep["torque_nm"] >= RATED_TORQUE_NM + 1.0
    ax_env.scatter(sweep["torque_nm"][falling], sweep["speed_rpm"][falling], s=30,
                   facecolors="none", edgecolors=RED, linewidths=1.3, zorder=4,
                   label="digitized sweep (falling branch)")
    dc_edge_tau = [0.0, RATED_TORQUE_NM, RATED_TORQUE_NM]
    dc_edge_rpm = [NO_LOAD_SPEED_RAD_S * RPM,
                   NO_LOAD_SPEED_RAD_S * (1.0 - RATED_TORQUE_NM / PEAK_TORQUE_NM) * RPM,
                   0.0]
    ax_env.plot(dc_edge_tau, dc_edge_rpm, color=BLUE, lw=2.0, zorder=3,
                label="training baseline: DCMotor 8 / 20 / 19.9")

    ax_env.scatter([RATED_TORQUE_NM], [RATED_SPEED_RAD_S * RPM], marker="*", s=210,
                   color=INK, zorder=5)
    ax_env.text(RATED_TORQUE_NM + 0.35, RATED_SPEED_RAD_S * RPM + 5.0,
                "rated: 8 N-m @ 120 rpm", fontsize=8.5, color=INK, ha="left")
    ax_env.scatter([0.0], [NO_LOAD_SPEED_RAD_S * RPM], s=46, facecolor=SURFACE,
                   edgecolor=INK, zorder=5)
    ax_env.text(0.4, NO_LOAD_SPEED_RAD_S * RPM + 5.0,
                "no-load: 190 rpm (= 19.897 rad/s)", fontsize=8.5, color=INK, ha="left")

    ax_env.text(11.4, 150.0, "DM-J8006-2EC envelope", color=RED, fontsize=9.0, ha="left")
    ax_env.text(8.55, 30.0, "training baseline:\nDCMotor 8 / 20 / 19.9\n(effort_limit clips at 8 N-m)",
                color=BLUE, fontsize=8.5, va="bottom")
    ax_env.text(0.018, 0.02, "four-quadrant symmetric (mirror on |torque|, |speed|); motoring quadrant shown",
                transform=ax_env.transAxes, fontsize=8.0, color=INK2, ha="left", va="bottom")

    ax_env.set_xlim(0.0, PEAK_TORQUE_NM * 1.05)
    ax_env.set_ylim(0.0, 208.0)
    ax_env.set_xticks(np.arange(0.0, 20.1, 4.0))
    ax_env.set_yticks(np.arange(0.0, 201.0, 50.0))
    ax_env.set_xlabel("Torque (N-m)", fontsize=11.0, color=INK)
    ax_env.set_ylabel("Speed (rpm)", fontsize=11.0, color=INK)
    style_axes(ax_env)
    ax_env.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor=SURFACE,
                  edgecolor="none", framealpha=0.9, labelcolor=INK)

    fig.suptitle("DM-J8006-2EC @ 24 V: digitized datasheet curves and max torque-speed envelope (joint side, after 6:1)",
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
