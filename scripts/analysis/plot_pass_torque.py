"""Pass-through torque demand per terrain.

Answers the sizing question: how much torque does each motor actually need
while the robot *passes through* each of the six training terrains?
Uses only the forward (fwd mid / fwd fast) segments of the rough diagnostics
CSV -- the segments where the robot walks across the terrain features.

Figures
  06_pass_torque_envelope : per-terrain time envelope of max-over-joints |tau|
                            (L2 fast / L2 mid / L0 fast) with DM8006 rated 8 /
                            peak 20 reference lines
  07_torque_demand_hist   : per-terrain |tau| histograms (L0 vs L2) with the
                            fraction above rated / at peak
  pass_torque_summary.csv : one row per terrain x level x speed with distance,
                            peak, RMS, %>8, #sat20

Usage:
  python scripts/analysis/plot_pass_torque.py --csv <rough csv> [--out_dir DIR]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_terrain_report import (  # noqa: E402
    DM_PEAK_TORQUE,
    DM_RATED_TORQUE,
    INK,
    INK2,
    LIMIT_RED,
    MUTED,
    SURFACE,
    TERRAIN_COLORS,
    TERRAIN_LABEL,
    TERRAIN_NOTE,
    TERRAINS,
    seg,
    save,
    style_axes,
)

TAU_COLS = None  # filled in main() from the CSV header

NOTE_BBOX = dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=1.5)


# ------------------------------------------------------------------ helpers --
def tau_matrix(g):
    """(|tau|, n_steps, 8) for one command-phase segment."""
    return np.abs(g[TAU_COLS].to_numpy())


def max_over_joints(g):
    return tau_matrix(g).max(axis=1)


def pass_distance(g):
    t = g["phase_time_s"].to_numpy()
    vx = np.abs(g["base_lin_vel_x"].to_numpy())
    return float(np.trapz(vx, t)) if len(t) > 1 else 0.0


def stats_row(terr, lvl, scen, g):
    tau = tau_matrix(g)
    rms = float(np.sqrt((tau ** 2).mean()))
    return dict(
        terrain=terr,
        level=lvl,
        scenario=scen,
        passed_m=round(pass_distance(g), 2),
        peak_nm=round(float(tau.max()), 1),
        peak_joint=TAU_COLS[int(np.unravel_index(tau.argmax(), tau.shape)[1])].replace("joint_torque_nm_", ""),
        rms_nm=round(rms, 2),
        over8_pct=round(100 * float((tau > DM_RATED_TORQUE).mean()), 1),
        sat20=int((tau >= DM_PEAK_TORQUE - 0.05).sum()),
    )


# ------------------------------------------------------------------- figures --
def fig_envelope(df, out_dir):
    """Per-terrain envelope of the worst joint torque while passing."""
    fig, axes = plt.subplots(
        2, 3, figsize=(13.5, 7.4), sharex=True, sharey=True,
        constrained_layout=True, facecolor=SURFACE,
    )
    t0 = None
    for ax, terr in zip(axes.ravel(), TERRAINS):
        color = TERRAIN_COLORS[terr]
        series = {}
        for lvl in (0, 2):
            for scen, ls, lw in (("forward_fast", "-", 1.9), ("forward_mid", "--", 1.1)):
                g = seg(df, terr, lvl, scen)
                if len(g) < 50:
                    continue
                if t0 is None:
                    t0 = g["phase_time_s"].to_numpy()
                series[(lvl, scen)] = (max_over_joints(g), color[f"L{lvl}"], ls, lw)
        for (lvl, scen), (y, c, ls, lw) in series.items():
            ax.plot(t0[: len(y)], y, color=c, linestyle=ls, linewidth=lw)
        ax.axhline(DM_RATED_TORQUE, color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)))
        ax.axhline(DM_PEAK_TORQUE, color=LIMIT_RED, linewidth=1.1)
        ax.set_title(f"{TERRAIN_LABEL[terr]}  ({TERRAIN_NOTE[terr]})", fontsize=9.5, color=INK)
        style_axes(ax)
        ax.set_ylim(0, 22)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_xlim(0, 5)
        # annotation: peaks / RMS / saturation, from both speeds and both levels
        rows = []
        for lvl in (0, 2):
            taus = [tau_matrix(seg(df, terr, lvl, s)) for s in ("forward_mid", "forward_fast")]
            taus = [x for x in taus if x.size]
            if not taus:
                continue
            all_t = np.concatenate([x.ravel() for x in taus])
            rows.append((lvl, all_t.max(), np.sqrt((all_t ** 2).mean()), 100 * (all_t > DM_RATED_TORQUE).mean(),
                         int((all_t >= DM_PEAK_TORQUE - 0.05).sum())))
        if rows:
            lines = []
            for lvl, peak, rms, o8, s20 in rows:
                lines.append(f"L{lvl}: pk {peak:.1f}  rms {rms:.1f}  >8 {o8:.0f}%" + (f"  sat20 x{s20}" if s20 else ""))
            ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.4, color=INK2, linespacing=1.45, bbox=NOTE_BBOX, zorder=5)
    for ax in axes[-1]:
        ax.set_xlabel("time in command phase (s)", fontsize=8.5, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("worst-joint |torque| (N·m)", fontsize=8.5, color=INK)
    # line-type legend in the first panel (its upper right stays empty)
    handles = [
        Line2D([], [], color=INK2, linewidth=1.9, linestyle="-", label="L2 fast"),
        Line2D([], [], color=INK2, linewidth=1.1, linestyle="--", label="L2 mid"),
        Line2D([], [], color=INK2, linewidth=1.4, linestyle="-", alpha=0.45, label="L0 fast"),
        Line2D([], [], color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)), label="DM8006 rated 8"),
        Line2D([], [], color=LIMIT_RED, linewidth=1.1, label="DM8006 peak 20"),
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=7.3, loc="upper right",
                      handlelength=2.2, borderaxespad=0.4)
    fig.suptitle(
        "Torque needed to PASS each terrain  (forward mid + fast, 24 pass segments, all completed)",
        fontsize=12, color=INK,
    )
    save(fig, out_dir, "06_pass_torque_envelope.png")


def fig_hist(df, out_dir):
    """Per-terrain distribution of |tau| samples while passing."""
    fig, axes = plt.subplots(
        2, 3, figsize=(13.5, 7.0), sharex=True, sharey="row",
        constrained_layout=True, facecolor=SURFACE,
    )
    bins = np.arange(0, 21.75, 0.5)
    for ax, terr in zip(axes.ravel(), TERRAINS):
        color = TERRAIN_COLORS[terr]
        notes, legend_handles = [], []
        for lvl in (0, 2):
            taus = [tau_matrix(seg(df, terr, lvl, s)) for s in ("forward_mid", "forward_fast")]
            taus = [x for x in taus if x.size]
            if not taus:
                continue
            all_t = np.concatenate([x.ravel() for x in taus])
            if lvl == 0:  # pale step outline so both levels stay readable when overlaid
                ax.hist(all_t, bins=bins, histtype="step", color=color["L0"],
                        linewidth=1.4, label=f"Level 0 (n={len(all_t)})")
            else:
                ax.hist(all_t, bins=bins, color=color["L2"], alpha=0.72,
                        edgecolor=SURFACE, linewidth=0.3, label=f"Level 2 (n={len(all_t)})")
            o8 = 100 * (all_t > DM_RATED_TORQUE).mean()
            s20 = int((all_t >= DM_PEAK_TORQUE - 0.05).sum())
            notes.append(f"L{lvl}: {o8:.1f}% > 8 N·m" + (f", {s20} at 20" if s20 else ""))
        ax.axvline(DM_RATED_TORQUE, color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)))
        ax.axvline(DM_PEAK_TORQUE, color=LIMIT_RED, linewidth=1.1)
        ax.set_title(TERRAIN_LABEL[terr], fontsize=9.5, color=INK)
        style_axes(ax)
        ax.set_xlim(0, 22)
        ax.set_xticks([0, 5, 8, 10, 15, 20])
        if notes:
            ax.text(0.97, 0.97, "\n".join(notes), transform=ax.transAxes, va="top", ha="right",
                    fontsize=7.6, color=INK2, linespacing=1.5, bbox=NOTE_BBOX, zorder=5)
    for ax in axes[-1]:
        ax.set_xlabel("|torque| (N·m)   ·   dashed = rated 8, red = peak 20", fontsize=8.5, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("samples", fontsize=8.5, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=7.6, loc="center right")
    fig.suptitle(
        "Torque-sample distribution while passing each terrain  (fwd mid + fast, all 8 joints)",
        fontsize=12, color=INK,
    )
    save(fig, out_dir, "07_torque_demand_hist.png")


# ---------------------------------------------------------------------- main --
def main():
    global TAU_COLS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    csv = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    TAU_COLS = [c for c in df.columns if c.startswith("joint_torque_nm_")]
    assert len(TAU_COLS) == 8, f"expected 8 torque columns, got {len(TAU_COLS)}"

    rows = []
    for terr in TERRAINS:
        for lvl in (0, 2):
            for scen in ("forward_mid", "forward_fast"):
                g = seg(df, terr, lvl, scen)
                if len(g) >= 50:
                    rows.append(stats_row(terr, lvl, scen, g))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "pass_torque_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(f"\n  wrote pass_torque_summary.csv ({len(summary)} rows)")

    fig_envelope(df, out_dir)
    fig_hist(df, out_dir)


if __name__ == "__main__":
    main()
