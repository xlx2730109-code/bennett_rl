"""Motor-report charts for the Bennett free_gait3 command matrix (PNG atlas).

Reads the CSV written by ``scripts/rsl_rl/collect_bennett_policy_diagnostics.py``
(the full omnidirectional command matrix) and writes a set of 300-dpi PNGs into
the CSV's own directory (or ``--out_dir``), one file per chart:

  01  joint torque time series      (6 directions x 3 speeds, 8 joints)
  02  joint power time series       (P = tau*omega, drive positive / brake negative)
  03  torque heatmap                (joint x scenario: peak & mean |tau|)
  04  power heatmap                 (joint x scenario: peak & mean |P|)
  05  torque-speed operating points (8 joints + DM8006 8/20 N-m rating lines)
  06  Hildebrand gait diagram       (contact raster from measured foot force)
  07  vertical foot force           (mid speed, all directions)
  08  cost of transport             (sum |P| / (m g v), per scenario)
  09  velocity tracking             (commanded vs achieved, dominant component)

Palette follows the validated default: legs are categorical slots 1-4
(thigh solid / calf dashed = composite encoding), heatmaps are a single-hue
sequential ramp, chart chrome uses the muted ink tokens.  English labels, 300 dpi.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Keep SVG text as real, editable text (Inkscape / PowerPoint / Illustrator),
# instead of converting glyphs to paths.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------- palette ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

LEG_COLORS = {  # categorical slots 1-4, one per leg (identity is fixed)
    "FL": "#2a78d6",
    "FR": "#eb6834",
    "RL": "#1baf7a",
    "RR": "#e87ba4",
}
LEGEND_LINE = Line2D  # alias for style keys below

DIR_COLORS = {  # 6 direction identities for the CoT bars (slots 1-4 + 6)
    "forward": "#2a78d6",
    "backward": "#eb6834",
    "lateral_left": "#1baf7a",
    "lateral_right": "#eda100",
    "yaw_left": "#e87ba4",
    "yaw_right": "#008300",
}

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

JOINTS = ("FL_thigh", "FL_calf", "FR_thigh", "FR_calf",
          "RL_thigh", "RL_calf", "RR_thigh", "RR_calf")
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

DIRECTIONS = ("forward", "backward", "lateral_left", "lateral_right", "yaw_left", "yaw_right")
SPEEDS = ("slow", "mid", "fast")

# Command magnitudes shown in row labels.  Must match the fraction table in
# scripts/rsl_rl/collect_bennett_policy_diagnostics.py (_matrix()).
SPEED_FRAC = {"slow": 0.25, "mid": 0.57, "fast": 0.91}
CMD_RANGE = {
    "forward": ("lin_x", 0.35), "backward": ("lin_x", 0.35),
    "lateral_left": ("lin_y", 0.25), "lateral_right": ("lin_y", 0.25),
    "yaw_left": ("ang_z", 0.60), "yaw_right": ("ang_z", 0.60),
}
CMD_UNIT = {"lin_x": "m/s", "lin_y": "m/s", "ang_z": "rad/s"}


def speed_triplet(direction):
    """Exact commanded speeds of one direction, slow/mid/fast, e.g. '0.088/0.200/0.318 m/s'."""
    kind, vmax = CMD_RANGE[direction]
    vals = [vmax * SPEED_FRAC[s] for s in SPEEDS]
    return f"cmd {vals[0]:.3f}/{vals[1]:.3f}/{vals[2]:.3f} {CMD_UNIT[kind]}"


def speed_value(direction, speed):
    """Exact commanded speed of one scenario, e.g. '0.199 m/s'."""
    kind, vmax = CMD_RANGE[direction]
    return f"{vmax * SPEED_FRAC[speed]:.3f} {CMD_UNIT[kind]}"

# Robot mass [kg]: base 4.22141 + 4 legs x (0.99385+0.30971+0.19132+0.16224+0.19132+0.072)
#   = 4.22141 + 4 x 1.92044  (Urdf_Bennett_3 inertials)
ROBOT_MASS_KG = 11.90
GRAVITY = 9.81
DM_RATED_TORQUE = 8.0    # DM-J8006-2EC continuous rating
DM_PEAK_TORQUE = 20.0    # peak
CONTACT_FORCE_N = 5.0    # foot force threshold for stance detection


# ------------------------------------------------------------------ style ---
def style_axes(ax, y_grid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.0)
    if y_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def new_fig(w, h, nrows, ncols, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), constrained_layout=True, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, np.atleast_1d(axes).ravel()


def save(fig, out_dir, name):
    stem = name.rsplit(".", 1)[0]
    fig.savefig(out_dir / f"{stem}.svg", facecolor=SURFACE)  # editable text
    fig.savefig(out_dir / name, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name} + {stem}.svg")


def scenario_list(df):
    """All command scenarios in matrix order (stand first, then dirs x speeds)."""
    order = [("stand", "stand")]
    for d in DIRECTIONS:
        for s in SPEEDS:
            order.append((f"{d}_{s}", d))
    seen = set(df["scenario"].unique())
    return [(name, d) for name, d in order if name in seen]


def speed_label(scenario):
    for s in SPEEDS:
        if scenario.endswith(s):
            return s.capitalize()
    return "Stand"


def dir_label(direction):
    return {
        "stand": "Stand",
        "forward": "Forward",
        "backward": "Backward",
        "lateral_left": "Left",
        "lateral_right": "Right",
        "yaw_left": "Yaw left",
        "yaw_right": "Yaw right",
    }[direction]


def leg_of(joint):
    return joint.split("_")[0]


def command_cols(df, scen_rows, phase):
    """Group rows by scenario, keeping only the given phase."""
    frames = []
    for name, _ in scen_rows:
        sub = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
        if len(sub):
            frames.append(sub)
    return frames


# ------------------------------------------------------------------ charts ---
def torque_timeseries(df, out_dir, phase="command"):
    _series_grid(df, out_dir, "01_torque_timeseries.png",
                 "Joint torque per scenario  (command phase)",
                 lambda f, j: f[f"joint_torque_nm_{j}"].to_numpy(), "N-m", phase)


def power_timeseries(df, out_dir, phase="command"):
    _series_grid(df, out_dir, "02_power_timeseries.png",
                 "Joint mechanical power  P = tau*omega  (positive = drive, negative = brake)",
                 lambda f, j: (f[f"joint_torque_nm_{j}"] * f[f"joint_vel_rad_s_{j}"]).to_numpy(),
                 "W", phase)


def _series_grid(df, out_dir, fname, title, col_fn, unit, phase):
    """6 directions (rows) x 3 speeds (cols); one line per joint, leg colours."""
    available = set(df["scenario"].unique())
    grid = [(f"{d}_{s}", d, s) for d in DIRECTIONS for s in SPEEDS
            if f"{d}_{s}" in available]
    if not grid:
        print(f"  [skip] {fname}: no command scenarios")
        return
    fig, axes = plt.subplots(len(DIRECTIONS), len(SPEEDS),
                             figsize=(3.2 * len(SPEEDS), 1.95 * len(DIRECTIONS)),
                             sharex=True, sharey=True, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)
    values = []
    for name, direction, speed in grid:
        frame = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
        if not len(frame):
            continue
        for joint in JOINTS:
            values.append(np.abs(col_fn(frame, joint)).max())
    vmax = max(values) * 1.05 if values else 1.0
    for r, direction in enumerate(DIRECTIONS):
        for c, speed in enumerate(SPEEDS):
            ax = axes[r][c]
            name = f"{direction}_{speed}"
            if name not in available:
                ax.set_visible(False)
                continue
            frame = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
            if not len(frame):
                ax.set_visible(False)
                continue
            t0 = frame["scenario_time_s"].to_numpy()
            t0 = t0 - t0[0]
            for joint in JOINTS:
                leg = leg_of(joint)
                is_thigh = joint.endswith("thigh")
                ax.plot(t0, col_fn(frame, joint), color=LEG_COLORS[leg],
                        linestyle="-" if is_thigh else "--",
                        linewidth=1.7, solid_capstyle="round")
            style_axes(ax)
            ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=0)
            ax.set_ylim(-vmax, vmax)
            ax.yaxis.set_major_locator(MultipleLocator(2.5))
            # column label on the top row, exact command speed on every panel
            if r == 0:
                ax.set_title(f"{speed.capitalize()}  {speed_value(direction, speed)}",
                             fontsize=9.5, color=INK, pad=4)
            else:
                ax.set_title(speed_value(direction, speed), fontsize=7.5, color=INK, pad=4)
            if r == len(DIRECTIONS) - 1:
                ax.set_xlabel("Time (s)", fontsize=8.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{dir_label(direction)} ({unit})", fontsize=8.5, color=INK)
    handles = [Line2D([], [], color=LEG_COLORS[l], linestyle="-" if k == "thigh" else "--",
                      linewidth=1.8, label=f"{l} {k}")
               for l in LEG_COLORS for k in ("thigh", "calf")]
    fig.suptitle(title, color=INK, fontsize=13, fontweight="bold")
    fig.legend(handles=handles, ncol=8, loc="outside lower center", frameon=False,
               fontsize=8, labelcolor=INK)
    save(fig, out_dir, fname)


def _heat(ax, matrix, title, cmap_norm, cbar_label):
    im = ax.imshow(matrix, aspect="auto", **cmap_norm)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in matrix.columns], fontsize=6, color=MUTED)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    cbar.outline.set_visible(False)
    cbar.set_label(cbar_label, fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10, color=INK2, pad=6)


def torque_heatmap(df, out_dir, phase="command"):
    scen_rows = [(n, d) for n, d in scenario_list(df) if n != "stand"]
    if not scen_rows:
        print("  [skip] 03: no scenarios")
        return
    peak = pd.DataFrame(index=JOINTS, columns=[n for n, _ in scen_rows], dtype=float)
    mean = pd.DataFrame(index=JOINTS, columns=[n for n, _ in scen_rows], dtype=float)
    for name, _ in scen_rows:
        sub = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
        if not len(sub):
            continue
        for joint in JOINTS:
            vals = sub[f"joint_torque_nm_{joint}"].abs().to_numpy()
            peak.loc[joint, name] = vals.max()
            mean.loc[joint, name] = vals.mean()
    fig, axes = new_fig(16.5, 4.6, 1, 2)
    _heat(axes[0], peak, "Peak |torque| per joint and scenario",
          {"cmap": SEQ_CMAP, "vmin": 0, "vmax": float(np.nanmax(peak.to_numpy()))}, "N-m")
    _heat(axes[1], mean, "Mean |torque| per joint and scenario",
          {"cmap": SEQ_CMAP, "vmin": 0, "vmax": float(np.nanmax(mean.to_numpy()))}, "N-m")
    fig.suptitle("Torque load by joint x scenario  (command phase)", color=INK,
                 fontsize=13, fontweight="bold")
    save(fig, out_dir, "03_torque_heatmap.png")


def power_heatmap(df, out_dir, phase="command"):
    scen_rows = [(n, d) for n, d in scenario_list(df) if n != "stand"]
    if not scen_rows:
        print("  [skip] 04: no scenarios")
        return
    peak = pd.DataFrame(index=JOINTS, columns=[n for n, _ in scen_rows], dtype=float)
    mean = pd.DataFrame(index=JOINTS, columns=[n for n, _ in scen_rows], dtype=float)
    for name, _ in scen_rows:
        sub = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
        if not len(sub):
            continue
        for joint in JOINTS:
            power = (sub[f"joint_torque_nm_{joint}"] * sub[f"joint_vel_rad_s_{joint}"]).abs().to_numpy()
            peak.loc[joint, name] = power.max()
            mean.loc[joint, name] = power.mean()
    fig, axes = new_fig(16.5, 4.6, 1, 2)
    _heat(axes[0], peak, "Peak |P| per joint and scenario",
          {"cmap": SEQ_CMAP, "vmin": 0, "vmax": float(np.nanmax(peak.to_numpy()))}, "W")
    _heat(axes[1], mean, "Mean |P| per joint and scenario",
          {"cmap": SEQ_CMAP, "vmin": 0, "vmax": float(np.nanmax(mean.to_numpy()))}, "W")
    fig.suptitle("Mechanical power load by joint x scenario  (command phase)",
                 color=INK, fontsize=13, fontweight="bold")
    save(fig, out_dir, "04_power_heatmap.png")


def torque_speed_scatter(df, out_dir, phase="command"):
    fig, axes = new_fig(15.0, 7.2, 2, 4)
    for ax, joint in zip(axes, JOINTS):
        leg = leg_of(joint)
        taus, omegas = [], []
        for name in df["scenario"].unique():
            sub = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
            if not len(sub):
                continue
            taus.append(sub[f"joint_torque_nm_{joint}"].to_numpy())
            omegas.append(sub[f"joint_vel_rad_s_{joint}"].to_numpy())
        if not taus:
            continue
        tau = np.concatenate(taus)
        om = np.concatenate(omegas)
        ax.scatter(om, tau, s=6, color=LEG_COLORS[leg], alpha=0.25, linewidths=0,
                   edgecolors="none")
        for level, style, text in ((DM_RATED_TORQUE, "-", "rated 8"),
                                   (DM_PEAK_TORQUE, "-", "peak 20")):
            for sign in (1, -1):
                ax.axhline(sign * level, color=AXIS, linewidth=1.0, linestyle=style, zorder=0)
        ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=0)
        style_axes(ax)
        ax.set_title(joint, fontsize=9, color=INK2, pad=4)
        ax.set_ylim(-DM_PEAK_TORQUE - 2, DM_PEAK_TORQUE + 2)
    fig.suptitle("Torque-speed operating points vs DM-J8006-2EC limits "
                 f"(gray lines = +-{DM_RATED_TORQUE:.0f} rated / +-{DM_PEAK_TORQUE:.0f} peak N-m)",
                 color=INK, fontsize=13, fontweight="bold")
    save(fig, out_dir, "05_torque_speed_operating_points.png")


def hildebrand(df, out_dir, phase="command"):
    """Contact raster: one column of subplots per direction (mid speed)."""
    scen = [f"{d}_mid" for d in DIRECTIONS]
    scen = [s for s in scen if s in set(df["scenario"].unique())]
    if not scen:
        print("  [skip] 06: no mid-speed scenarios")
        return
    fig, axes = new_fig(15.0, 2.4 * len(scen), len(scen), 1)
    for ax, name in zip(axes, scen):
        sub = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
        t = sub["scenario_time_s"].to_numpy()
        t0 = t - t[0]
        for row, foot in enumerate(FEET):
            contact = (sub[f"foot_contact_force_n_{foot}"].to_numpy() > CONTACT_FORCE_N).astype(int)
            y = len(FEET) - 1 - row  # FL on top
            starts = np.flatnonzero(np.diff(np.concatenate(([0], contact)) ) == 1)
            ends = np.flatnonzero(np.diff(np.concatenate((contact, [0]))) == -1)
            for s0, e0 in zip(starts, ends):
                ax.fill_between([t0[s0], t0[e0]], y, y + 0.82,
                                color=LEG_COLORS[foot.split("_")[0]], linewidth=0)
            duty = contact.mean()
            ax.text(t0[-1] + 0.05, y + 0.4, f"duty {duty:.2f}", va="center",
                    fontsize=7.5, color=INK2)
        ax.set_yticks([len(FEET) - 1 - i for i in range(len(FEET))])
        ax.set_yticklabels(FEET, fontsize=8, color=MUTED)
        ax.set_xlim(0.0, t0[-1] * 1.12)
        ax.set_ylim(-0.2, len(FEET) - 0.6)
        style_axes(ax, y_grid=False)
        ax.set_title(f"{dir_label(name.rsplit('_', 1)[0])} (mid speed)",
                     fontsize=10, color=INK2, pad=4, loc="left")
    fig.suptitle("Hildebrand gait diagram - measured foot contact "
                 f"(force > {CONTACT_FORCE_N:.0f} N, command phase)",
                 color=INK, fontsize=13, fontweight="bold")
    save(fig, out_dir, "06_hildebrand_gait_diagram.png")


def foot_force(df, out_dir, phase="command"):
    scen = [f"{d}_mid" for d in DIRECTIONS]
    scen = [s for s in scen if s in set(df["scenario"].unique())]
    if not scen:
        print("  [skip] 07: no mid-speed scenarios")
        return
    fig, axes = new_fig(15.0, 2.4 * len(scen), len(scen), 1)
    for ax, name in zip(axes, scen):
        sub = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
        t0 = sub["scenario_time_s"].to_numpy()
        t0 = t0 - t0[0]
        for foot in FEET:
            ax.plot(t0, sub[f"foot_contact_force_n_{foot}"].to_numpy(),
                    color=LEG_COLORS[foot.split("_")[0]], linewidth=1.8, solid_capstyle="round")
        style_axes(ax)
        ax.set_title(f"{dir_label(name.rsplit('_', 1)[0])} (mid speed)",
                     fontsize=10, color=INK2, pad=4, loc="left")
    handles = [Line2D([], [], color=LEG_COLORS[f.split("_")[0]], linewidth=1.8, label=f)
               for f in FEET]
    fig.suptitle("Vertical foot contact force (mid speed, command phase)",
                 color=INK, fontsize=13, fontweight="bold")
    fig.legend(handles=handles, ncol=4, loc="outside lower center", frameon=False,
               fontsize=8, labelcolor=INK2)
    save(fig, out_dir, "07_foot_contact_force.png")


def cost_of_transport(df, out_dir, phase="command"):
    scen_rows = [(n, d) for n, d in scenario_list(df) if n != "stand"]
    if not scen_rows:
        print("  [skip] 08: no scenarios")
        return
    names, cots, colors = [], [], []
    for name, direction in scen_rows:
        sub = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
        if not len(sub):
            continue
        vx = sub["base_lin_vel_x"].to_numpy()
        vy = sub["base_lin_vel_y"].to_numpy()
        v = np.hypot(vx, vy).mean()
        if v < 1e-3:
            continue
        p = sub["mechanical_power_abs_w"].mean()
        cot = p / (ROBOT_MASS_KG * GRAVITY * v)
        names.append(name)
        cots.append(cot)
        colors.append(DIR_COLORS[direction])
    if not cots:
        print("  [skip] 08: all speeds ~zero")
        return
    fig, ax = plt.subplots(figsize=(15.0, 4.8), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(cots))
    ax.bar(x, cots, width=0.62, color=colors, edgecolor=SURFACE, linewidth=1.0)
    style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7, color=MUTED)
    for xi, c in zip(x, cots):
        ax.text(xi, c + max(cots) * 0.015, f"{c:.2f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_ylabel("CoT  (dimensionless)", fontsize=9, color=INK2)
    ax.set_ylim(0, max(cots) * 1.12)
    handles = [Line2D([], [], marker="s", linestyle="", markersize=9,
                      markerfacecolor=c, markeredgecolor="none", label=dir_label(d))
               for d, c in DIR_COLORS.items()]
    fig.suptitle("Cost of transport  CoT = mean(sum|P|) / (m g v)   "
                 f"(m = {ROBOT_MASS_KG:.1f} kg)  -  mini-cheetah trot ~ 0.9",
                 color=INK, fontsize=13, fontweight="bold")
    fig.legend(handles=handles, ncol=6, loc="outside lower center", frameon=False,
               fontsize=8, labelcolor=INK2)
    save(fig, out_dir, "08_cost_of_transport.png")


def velocity_tracking(df, out_dir, phase="command"):
    """6 directions (rows) x 3 speeds (cols); actual vs commanded dominant component."""
    available = set(df["scenario"].unique())
    grid = [(d, s) for d in DIRECTIONS for s in SPEEDS if f"{d}_{s}" in available]
    if not grid:
        print("  [skip] 09: no command scenarios")
        return
    fig, axes = plt.subplots(len(DIRECTIONS), len(SPEEDS),
                             figsize=(3.2 * len(SPEEDS), 1.9 * len(DIRECTIONS)),
                             sharex=True, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)
    comp_of = {"forward": "base_lin_vel_x", "backward": "base_lin_vel_x",
               "lateral_left": "base_lin_vel_y", "lateral_right": "base_lin_vel_y",
               "yaw_left": "base_ang_vel_z", "yaw_right": "base_ang_vel_z"}
    cmd_of = {"base_lin_vel_x": "command_x", "base_lin_vel_y": "command_y",
              "base_ang_vel_z": "command_yaw"}
    for r, direction in enumerate(DIRECTIONS):
        comp = comp_of[direction]
        cmd_col = cmd_of[comp]
        for c, speed in enumerate(SPEEDS):
            ax = axes[r][c]
            name = f"{direction}_{speed}"
            frame = df[(df["scenario"] == name) & (df["scenario_phase"] == phase)]
            if name not in available or not len(frame):
                ax.set_visible(False)
                continue
            t0 = frame["scenario_time_s"].to_numpy()
            t0 = t0 - t0[0]
            ax.plot(t0, frame[comp].to_numpy(), color="#2a78d6", linewidth=1.8,
                    solid_capstyle="round")
            ax.plot(t0, frame[cmd_col].to_numpy(), color=AXIS, linewidth=1.6,
                    linestyle="--")
            style_axes(ax)
            ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=0)
            if r == 0:
                ax.set_title(speed.capitalize(), fontsize=10, color=INK, pad=4)
            if r == len(DIRECTIONS) - 1:
                ax.set_xlabel("Time (s)", fontsize=8.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{dir_label(direction)} ({comp.split('_')[-1]})\n{speed_triplet(direction)}",
                              fontsize=8.5, color=INK2)
    handles = [Line2D([], [], color="#2a78d6", linewidth=1.8, label="actual"),
               Line2D([], [], color=AXIS, linewidth=1.6, linestyle="--", label="command")]
    fig.suptitle("Velocity tracking (dominant component per direction)",
                 color=INK, fontsize=13, fontweight="bold")
    fig.legend(handles=handles, ncol=2, loc="outside lower center", frameon=False,
               fontsize=8, labelcolor=INK2)
    save(fig, out_dir, "09_velocity_tracking.png")


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True, help="diagnostics CSV from collect_bennett_policy_diagnostics.py")
    ap.add_argument("--out_dir", type=Path, default=None, help="default: the CSV's own directory")
    args = ap.parse_args()

    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"[motor-report] csv={args.csv} rows={len(df)} scenarios={df['scenario'].nunique()}")
    print(f"[motor-report] out_dir={out_dir}")

    torque_timeseries(df, out_dir)
    power_timeseries(df, out_dir)
    torque_heatmap(df, out_dir)
    power_heatmap(df, out_dir)
    torque_speed_scatter(df, out_dir)
    hildebrand(df, out_dir)
    foot_force(df, out_dir)
    cost_of_transport(df, out_dir)
    velocity_tracking(df, out_dir)
    print("[motor-report] done.")


if __name__ == "__main__":
    main()
