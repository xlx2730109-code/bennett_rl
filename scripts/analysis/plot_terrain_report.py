"""Terrain-report charts for the Bennett go2 rough-terrain policy (PNG atlas).

Reads the rough-mode CSV written by
``scripts/rsl_rl/collect_bennett_policy_diagnostics.py --terrain_mode rough``
(terrain x terrain_level x scenario rows on the six training terrain types)
and writes a set of 300-dpi PNGs into the CSV's own directory (or ``--out_dir``):

  00  terrain x command overview      (speed tracking grid: 6 commands x 6 terrains,
                                       L0 solid-pale / L2 solid-dark vs grey command)
  01  torque heatmap by terrain       (joint x terrain/level, forward mid & fast)
  02  torque-speed operating points   (all scenarios, coloured by terrain, DM8006 envelope)
  03  Hildebrand gait diagram         (terrain rows x forward mid/fast cols)
  04  cost of transport by terrain    (grouped bars: 4 moving commands x 6 terrains x L0/L2)
  05  body attitude by terrain        (pitch/roll proxies, forward fast, L0 vs L2)

Terrain identity uses six categorical hues; difficulty level is a lightness
step of the same hue (L0 pale, L2 dark) so identity and level never collide.
Palette/chrome follow plot_motor_report.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- palette ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LIMIT_RED = "#d03b3b"
FELL_RED = "#d03b3b"
CMD_GREY = "#b9b8ae"

# Six terrain hues = categorical slots 1-6; L0 is a pale step of the same hue,
# L2 the full step (lightness encodes difficulty, hue encodes terrain).
TERRAINS = ("stairs", "stairs_down", "boxes", "random_rough", "slope_up", "slope_down")
TERRAIN_COLORS = {
    "stairs":       {"L0": "#8fb9ea", "L2": "#2a78d6"},
    "stairs_down":  {"L0": "#f3a886", "L2": "#eb6834"},
    "boxes":        {"L0": "#84d5b6", "L2": "#1baf7a"},
    "random_rough": {"L0": "#f4cf85", "L2": "#eda100"},
    "slope_up":     {"L0": "#f3bacc", "L2": "#e87ba4"},
    "slope_down":   {"L0": "#85c787", "L2": "#008300"},
}
LEVELS = (0, 2)
LEVEL_LABEL = {0: "Level 0", 2: "Level 2"}

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

JOINTS = ("FL_thigh", "FL_calf", "FR_thigh", "FR_calf",
          "RL_thigh", "RL_calf", "RR_thigh", "RR_calf")
FEET = ("FL_1", "FR_1", "RL_1", "RR_1")

# Commands in the reduced rough matrix, in plot order (stand handled separately).
COMMANDS = ("stand", "forward_mid", "forward_fast", "lateral_left_mid",
            "yaw_left_mid", "yaw_right_mid")
# Dominant velocity component per moving command (what the policy must track).
COMP_OF = {
    "forward_mid": ("base_lin_vel_x", "command_x", "vx"),
    "forward_fast": ("base_lin_vel_x", "command_x", "vx"),
    "lateral_left_mid": ("base_lin_vel_y", "command_y", "vy"),
    "yaw_left_mid": ("base_ang_vel_z", "command_yaw", "wz"),
    "yaw_right_mid": ("base_ang_vel_z", "command_yaw", "wz"),
}
CMD_LABEL = {
    "stand": "Stand",
    "forward_mid": "Fwd mid",
    "forward_fast": "Fwd fast",
    "lateral_left_mid": "Left mid",
    "yaw_left_mid": "Yaw L mid",
    "yaw_right_mid": "Yaw R mid",
}
TERRAIN_LABEL = {
    "stairs": "Stairs up",
    "stairs_down": "Stairs down",
    "boxes": "Boxes",
    "random_rough": "Random rough",
    "slope_up": "Slope up",
    "slope_down": "Slope down",
}
TERRAIN_NOTE = {
    "stairs": "step 0.05-0.23 m / width 0.3",
    "stairs_down": "step 0.05-0.23 m / width 0.3",
    "boxes": "grid 0.45 m, h 0.025-0.1 m",
    "random_rough": "noise 0.01-0.06 m, step 0.01",
    "slope_up": "slope 0-0.4 rad",
    "slope_down": "slope 0-0.4 rad",
}

ROBOT_MASS_KG = 12.2629  # V1 asset total mass
GRAVITY = 9.81
DM_RATED_TORQUE = 8.0
DM_PEAK_TORQUE = 20.0
CONTACT_FORCE_N = 5.0


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
    fig.savefig(out_dir / f"{stem}.svg", facecolor=SURFACE)
    fig.savefig(out_dir / name, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name} + {stem}.svg")


def leg_of(joint):
    return joint.split("_")[0]


def seg(df, terrain, level, scenario, phase="command"):
    """One scenario segment; empty frame if the run terminated before it."""
    sub = df[
        (df["terrain"] == terrain)
        & (df["terrain_level"] == level)
        & (df["scenario"] == scenario)
        & (df["scenario_phase"] == phase)
    ]
    return sub


def fell(df, terrain, level, scenario):
    """True if the scenario terminated early (any row marked done)."""
    sub = df[(df["terrain"] == terrain) & (df["terrain_level"] == level)
             & (df["scenario"] == scenario)]
    return bool(sub["done"].max() > 0) if len(sub) else False


def rel_time(frame):
    t = frame["scenario_time_s"].to_numpy()
    return t - t[0]


def line_color(terrain, level):
    return TERRAIN_COLORS[terrain]["L0" if level == 0 else "L2"]


# ------------------------------------------------------------------ charts ---
def overview(df, out_dir):
    """00 - commands (rows) x terrains (cols); L0/L2 lines vs grey command."""
    fig, axes = plt.subplots(len(COMMANDS), len(TERRAINS),
                             figsize=(2.9 * len(TERRAINS), 1.85 * len(COMMANDS)),
                             sharex=True, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)
    for r, cmd in enumerate(COMMANDS):
        for c, terrain in enumerate(TERRAINS):
            ax = axes[r][c]
            style_axes(ax)
            if cmd == "stand":
                # Standing: base height in world frame.  Stair patches spawn the
                # robot on the platform, so the level line itself shows the
                # patch height (stairs up climb, stairs down descend).
                for level in LEVELS:
                    f = seg(df, terrain, level, "stand")
                    if not len(f):
                        continue
                    ax.plot(rel_time(f), f["root_height_m"].to_numpy(),
                            color=line_color(terrain, level), linewidth=1.5)
                if r == 0:
                    ax.set_title(TERRAIN_LABEL[terrain], fontsize=9.5, color=INK, pad=4)
                if r == len(COMMANDS) - 1:
                    ax.set_xlabel("Time (s)", fontsize=8, color=INK)
                if c == 0:
                    ax.set_ylabel("Stand\nbase z, world (m)", fontsize=8.5, color=INK)
                continue
            comp, cmd_col, _ = COMP_OF[cmd]
            cmd_val = seg(df, terrain, LEVELS[0], cmd)[cmd_col]
            cmd_line = cmd_val.to_numpy() if len(cmd_val) else None
            rms = {}
            for level in LEVELS:
                f = seg(df, terrain, level, cmd)
                if not len(f):
                    continue
                t0 = rel_time(f)
                actual = f[comp].to_numpy()
                ax.plot(t0, actual, color=line_color(terrain, level), linewidth=1.5)
                if cmd_line is not None and len(cmd_line) == len(actual):
                    rms[level] = float(np.sqrt(np.mean((actual - cmd_line) ** 2)))
            if cmd_line is not None:
                ax.plot(rel_time(seg(df, terrain, LEVELS[0], cmd)), cmd_line,
                        color=CMD_GREY, linewidth=1.2, linestyle=(0, (3, 2)))
                ax.axhline(0.0, color=AXIS, linewidth=0.6, zorder=0)
            # per-cell verdict: RMS + fell marker
            notes = []
            for level in LEVELS:
                txt = f"{rms[level]:.3f}" if level in rms else "n/a"
                if fell(df, terrain, level, cmd):
                    txt += " fell"
                notes.append(f"L{level} {txt}")
            ax.text(0.985, 0.955, "  ·  ".join(notes), transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.2, color=INK2)
            if r == 0:
                ax.set_title(TERRAIN_LABEL[terrain], fontsize=9.5, color=INK, pad=4)
            if r == len(COMMANDS) - 1:
                ax.set_xlabel("Time (s)", fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(f"{CMD_LABEL[cmd]}\n({comp.split('_')[-1]})", fontsize=8.5, color=INK)
    handles = [
        *[Line2D([], [], color=line_color(t, lv), linewidth=1.8,
                 label=f"{TERRAIN_LABEL[t]} {LEVEL_LABEL[lv]}")
          for t in TERRAINS for lv in (2, 0)],  # dark before pale reads better
        Line2D([], [], color=CMD_GREY, linewidth=1.4, linestyle=(0, (3, 2)), label="command"),
    ]
    fig.suptitle("Rough-terrain overview - actual vs commanded on the six training "
                 "terrains (numbers: command-phase RMS error, fell = terminated early)",
                 color=INK, fontsize=13, fontweight="bold")
    fig.legend(handles=handles, ncol=7, loc="outside lower center", frameon=False,
               fontsize=7.5, labelcolor=INK2)
    save(fig, out_dir, "00_terrain_command_overview.png")


def torque_heatmap_terrain(df, out_dir):
    """01 - peak/mean |torque| per joint across terrain x level (forward)."""
    cols = [(t, lv) for t in TERRAINS for lv in LEVELS]
    col_names = [f"{TERRAIN_LABEL[t]}\n{LEVEL_LABEL[lv]}" for t, lv in cols]
    frames = {}
    for t, lv in cols:
        for cmd in ("forward_mid", "forward_fast"):
            f = seg(df, t, lv, cmd)
            if len(f):
                frames[(t, lv, cmd)] = f
    if not frames:
        print("  [skip] 01: no forward segments")
        return
    # one shared colour scale across both panels so magnitudes are comparable
    grids = {}
    for cmd in ("forward_mid", "forward_fast"):
        grid = np.full((len(JOINTS), len(cols)), np.nan)
        for j, (t, lv) in enumerate(cols):
            f = frames.get((t, lv, cmd))
            if f is None:
                continue
            for i, joint in enumerate(JOINTS):
                grid[i, j] = f[f"joint_torque_nm_{joint}"].abs().max()
        grids[cmd] = grid
    vmax = np.nanmax([np.nanmax(g) for g in grids.values()]) * 1.05
    fig, axes = new_fig(13.5, 5.4, 1, 2)
    for ax, cmd in zip(axes, ("forward_mid", "forward_fast")):
        peak = grids[cmd]
        im = ax.imshow(peak, aspect="auto", cmap=SEQ_CMAP, vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(col_names, fontsize=5.6, color=MUTED, rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(JOINTS)))
        ax.set_yticklabels(JOINTS, fontsize=7, color=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        # direct cell values keep the heatmap readable at a glance
        for i in range(len(JOINTS)):
            for j in range(len(cols)):
                if not np.isnan(peak[i, j]):
                    v = peak[i, j]
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.4,
                            color=INK if v < vmax * 0.5 else SURFACE)
        ax.set_title(f"{CMD_LABEL[cmd]}  peak |torque| (N-m)", fontsize=10, color=INK2, pad=6)
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(labelsize=7, colors=MUTED)
        cbar.outline.set_visible(False)
    fig.suptitle("Torque load per joint across terrain and difficulty (command phase)",
                 color=INK, fontsize=13, fontweight="bold")
    save(fig, out_dir, "01_torque_heatmap_terrain.png")


def torque_speed_terrain(df, out_dir):
    """02 - operating points coloured by terrain (L2 solid, L0 pale)."""
    fig, axes = new_fig(11.8, 7.2, 1, 1)
    ax = axes[0]
    vlim = 19.896753
    rated, peak_t = DM_RATED_TORQUE, DM_PEAK_TORQUE
    w = np.linspace(0.0, vlim, 400)
    env = np.clip(peak_t * (1.0 - w / vlim), 0.0, rated)
    ax.plot(w, env, color=LIMIT_RED, lw=2.0, ls=(0, (6, 3)), zorder=3)
    ax.plot(-w, -env, color=LIMIT_RED, lw=2.0, ls=(0, (6, 3)), zorder=3)
    knee = vlim * (1.0 - rated / peak_t)

    om_max = 1.5
    for terrain in TERRAINS:
        for level, alpha in ((2, 0.30), (0, 0.14)):
            pts_t, pts_o = [], []
            for cmd in COMMANDS:
                f = seg(df, terrain, level, cmd)
                if not len(f):
                    continue
                for joint in JOINTS:
                    pts_t.append(f[f"joint_torque_nm_{joint}"].to_numpy())
                    pts_o.append(f[f"joint_vel_rad_s_{joint}"].to_numpy())
            if not pts_t:
                continue
            ax.scatter(np.concatenate(pts_o), np.concatenate(pts_t), s=4,
                       color=line_color(terrain, level), alpha=alpha, linewidths=0,
                       edgecolors="none", rasterized=True, zorder=2)
            om_max = max(om_max, float(np.percentile(np.abs(np.concatenate(pts_o)), 99.9)))
    style_axes(ax)
    ax.set_xlabel("Joint speed (rad/s)", fontsize=11.5, color=INK)
    ax.set_ylabel("Joint torque (N-m)", fontsize=11.5, color=INK)
    ax.tick_params(colors=INK, labelsize=10, length=4, width=1.0)
    xmax = max(om_max * 1.3, knee + 2.5)  # always show the envelope knee (11.9 rad/s)
    ax.set_xlim(-xmax, xmax)
    # show the full +-20 N-m effort clip: terrain hits the limit where flat does not
    ax.set_ylim(-peak_t - 1.5, peak_t + 1.5)
    ax.set_yticks([-20, -15, -10, -5, 0, 5, 10, 15, 20])
    ylim = ax.get_ylim()
    for gl, ypos in zip(ax.yaxis.get_gridlines(), ax.get_yticks()):
        if ypos >= ylim[1] - 1e-9 or ypos <= ylim[0] + 1e-9:
            gl.set_visible(False)
    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda v: v * 60.0 / (2.0 * np.pi), lambda r: r * 2.0 * np.pi / 60.0),
    )
    sec.set_xlabel("Joint speed (rpm)", fontsize=11.5, color=INK)
    sec.tick_params(labelcolor=INK, color=INK, labelsize=10, length=4, width=1.0)
    sec.spines["top"].set_color(INK)
    sec.spines["top"].set_linewidth(1.0)
    ax.text(0.985, 0.975,
            f"red dashed = limit: flat +-{rated:.0f} N-m up to {knee:.1f} rad/s,\n"
            f"then linear to 0 at {vlim:.1f} rad/s (short-time peak +-{peak_t:.0f} N-m)\n"
            f"pale points = {LEVEL_LABEL[0]}  ·  solid points = {LEVEL_LABEL[2]}\n"
            "bands at +-20 N-m = joints saturated on the effort clip",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=INK, linespacing=1.4)
    handles = [
        Line2D([], [], color=LIMIT_RED, lw=2.0, ls=(0, (6, 3)),
               label=f"DM-J8006-2EC continuous limit (+-{rated:.0f} N-m)"),
        *[Line2D([], [], marker="o", ls="none", markersize=6,
                 color=TERRAIN_COLORS[t]["L2"], label=TERRAIN_LABEL[t])
          for t in TERRAINS],
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True,
              facecolor=SURFACE, edgecolor="none", framealpha=0.9,
              labelcolor=INK, ncols=1, borderaxespad=0.6)
    fig.suptitle("Torque-speed operating points on the six training terrains (all commands)",
                 color=INK, fontsize=13, fontweight="bold")
    save(fig, out_dir, "02_torque_speed_terrain.png")


def hildebrand_terrain(df, out_dir, cmd="forward_mid"):
    """03 - contact rasters: 12 rows (terrain x level) x 2 columns (mid/fast)."""
    cmds = [c for c in ("forward_mid", "forward_fast") if len(seg(df, TERRAINS[0], LEVELS[0], c))]
    if not cmds:
        print("  [skip] 03: no forward segments")
        return
    rows = [(t, lv) for t in TERRAINS for lv in LEVELS]
    fig, axes = plt.subplots(len(rows), len(cmds),
                             figsize=(11.5, 1.05 * len(rows) + 0.8),
                             squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    duty_all = {}
    for r, (terrain, level) in enumerate(rows):
        for c, cmd_name in enumerate(cmds):
            ax = axes[r][c]
            f = seg(df, terrain, level, cmd_name)
            if not len(f):
                ax.set_visible(False)
                continue
            t0 = rel_time(f)
            duty_txt = []
            for k, foot in enumerate(FEET):
                contact = (f[f"foot_contact_force_n_{foot}"].to_numpy() > CONTACT_FORCE_N).astype(int)
                y = len(FEET) - 1 - k
                starts = np.flatnonzero(np.diff(np.concatenate(([0], contact))) == 1)
                ends = np.flatnonzero(np.diff(np.concatenate((contact, [0]))) == -1)
                for s0, e0 in zip(starts, ends):
                    ax.fill_between([t0[s0], t0[e0]], y, y + 0.82,
                                    color=line_color(terrain, level), linewidth=0)
                duty = contact.mean()
                duty_all[(terrain, level, foot)] = duty
                duty_txt.append(f"{duty:.2f}")
            ax.text(t0[-1] * 1.005, 0.42, " ".join(duty_txt), va="center", fontsize=5.8,
                    color=INK2, ha="left")
            ax.set_yticks([len(FEET) - 1 - i for i in range(len(FEET))])
            ax.set_ylim(-0.2, len(FEET) - 0.6)
            style_axes(ax, y_grid=False)
            ax.set_xlim(0.0, t0[-1] * 1.16)
            if c == 0:
                ax.set_yticklabels([ft.split("_")[0] for ft in FEET], fontsize=5.6, color=MUTED)
                ax.set_ylabel(f"{TERRAIN_LABEL[terrain]}\n{LEVEL_LABEL[level]}", fontsize=7.2,
                              color=INK, rotation=0, ha="right", va="center", labelpad=6)
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(CMD_LABEL[cmd_name], fontsize=9.5, color=INK, pad=3)
            if r == len(rows) - 1:
                ax.set_xlabel("Time (s)", fontsize=8, color=INK)
            else:
                ax.set_xticklabels([])
    fig.suptitle("Gait rhythm per terrain - measured foot contact "
                 f"(force > {CONTACT_FORCE_N:.0f} N);  right numbers = per-foot duty",
                 color=INK, fontsize=13, fontweight="bold")
    save(fig, out_dir, "03_gait_rhythm_terrain.png")


def cot_by_terrain(df, out_dir):
    """04 - grouped bars: CoT per terrain (L0 pale / L2 dark), one group per command."""
    groups, labels = [], []
    for cmd in COMMANDS:
        if cmd == "stand":
            continue
        vals = []
        for terrain in TERRAINS:
            for level in LEVELS:
                f = seg(df, terrain, level, cmd)
                if not len(f):
                    vals.append(np.nan)
                    continue
                v = float(np.hypot(f["base_lin_vel_x"], f["base_lin_vel_y"]).mean())
                p = f["mechanical_power_abs_w"].mean()
                vals.append(p / (ROBOT_MASS_KG * GRAVITY * v) if v > 1e-3 else np.nan)
        groups.append(vals)
        labels.append(CMD_LABEL[cmd])
    if not groups:
        print("  [skip] 04: no moving segments")
        return
    fig, ax = plt.subplots(figsize=(13.5, 5.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    n_cmd, n_bar = len(groups), len(TERRAINS) * len(LEVELS)
    x = np.arange(n_cmd) * (n_bar + 1.4)
    all_vals = [v for g in groups for v in g if not np.isnan(v)]
    vmax = max(all_vals) * 1.14
    for gi, vals in enumerate(groups):
        for bi, (terrain, level) in enumerate([(t, lv) for t in TERRAINS for lv in LEVELS]):
            v = vals[bi]
            xi = x[gi] + bi
            color = line_color(terrain, level)
            if np.isnan(v):
                # distinguish a real fall (segment exists, terminated) from a
                # segment that simply is not in the CSV
                label = "fell" if fell(df, terrain, level, COMMANDS[gi + 1]) else "n/a"
                ax.text(xi, vmax * 0.02, label, rotation=90, ha="center", va="bottom",
                        fontsize=6.0, color=FELL_RED if label == "fell" else MUTED)
                continue
            ax.bar(xi, v, width=0.82, color=color, edgecolor=SURFACE, linewidth=0.8)
            ax.text(xi, v + vmax * 0.012, f"{v:.2f}", ha="center", fontsize=5.6, color=INK2)
    style_axes(ax)
    ax.set_xticks(x + (n_bar - 1) / 2)
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylabel("CoT  (dimensionless)", fontsize=9, color=INK2)
    ax.set_ylim(0, vmax)
    handles = [
        *[Line2D([], [], marker="s", linestyle="", markersize=8,
                 markerfacecolor=line_color(t, lv), markeredgecolor="none",
                 label=f"{TERRAIN_LABEL[t]} {LEVEL_LABEL[lv]}")
          for t in TERRAINS for lv in (2, 0)],
    ]
    fig.suptitle(f"Cost of transport  CoT = mean(sum|P|) / (m g v)   (m = {ROBOT_MASS_KG:.2f} kg, V1)",
                 color=INK, fontsize=13, fontweight="bold")
    fig.legend(handles=handles, ncol=7, loc="outside lower center", frameon=False,
               fontsize=7.5, labelcolor=INK2)
    save(fig, out_dir, "04_cot_by_terrain.png")


def attitude_terrain(df, out_dir):
    """05 - pitch/roll proxies (projected gravity x/y), forward fast, L0 vs L2."""
    comps = (("projected_gravity_x", "pitch ~ pg_x"), ("projected_gravity_y", "roll ~ pg_y"))
    rows = [(t, lv) for t in TERRAINS for lv in LEVELS]
    frames = {(t, lv): seg(df, t, lv, "forward_fast") for t, lv in rows}
    if not any(len(f) for f in frames.values()):
        print("  [skip] 05: no forward_fast segments")
        return
    # one y-scale per column so amplitudes are comparable across terrains
    ylims = []
    for col, _ in comps:
        vals = np.concatenate([f[col].to_numpy() for f in frames.values() if len(f)])
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.08 * (hi - lo) + 0.01
        ylims.append((lo - pad, hi + pad))
    fig, axes = plt.subplots(len(rows), len(comps),
                             figsize=(9.5, 1.05 * len(rows) + 0.6),
                             sharex=True, squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for r, (terrain, level) in enumerate(rows):
        for c, (col, comp_label) in enumerate(comps):
            ax = axes[r][c]
            f = frames[(terrain, level)]
            if not len(f):
                ax.set_visible(False)
                continue
            t0 = rel_time(f)
            ax.plot(t0, f[col].to_numpy(), color=line_color(terrain, level), linewidth=1.4)
            ax.axhline(0.0, color=AXIS, linewidth=0.7, zorder=0)
            style_axes(ax)
            ax.set_ylim(*ylims[c])
            rms = float(np.sqrt(np.mean(f[col].to_numpy() ** 2)))
            ax.text(0.985, 0.92, f"rms {rms:.3f}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=6.2, color=INK2)
            if r == 0:
                ax.set_title(comp_label, fontsize=9.5, color=INK, pad=3)
            if r == len(rows) - 1:
                ax.set_xlabel("Time (s)", fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(f"{TERRAIN_LABEL[terrain]}\n{LEVEL_LABEL[level]}", fontsize=7.5,
                              color=INK)
    fig.suptitle("Body attitude on forward fast  (pg_x = pitch, pg_y = roll; 0 = level)",
                 color=INK, fontsize=13, fontweight="bold")
    save(fig, out_dir, "05_body_attitude_terrain.png")


# -------------------------------------------------------------------- main ---
def main():
    global ROBOT_MASS_KG, FEET
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True, help="rough-mode diagnostics CSV")
    ap.add_argument("--out_dir", type=Path, default=None, help="default: the CSV's own directory")
    ap.add_argument("--mass", type=float, default=ROBOT_MASS_KG, help="Robot mass in kg (V1 total by default).")
    ap.add_argument("--feet", type=str, nargs=4, default=None, help="Foot column suffixes. Default FL_1-style.")
    args = ap.parse_args()

    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    ROBOT_MASS_KG = args.mass
    if args.feet is not None:
        FEET = tuple(args.feet)
    missing = [c for c in ("terrain", "terrain_level") if c not in df.columns]
    if missing:
        raise SystemExit(f"[terrain-report] CSV lacks {missing}; use --terrain_mode rough when collecting")
    print(f"[terrain-report] csv={args.csv} rows={len(df)} "
          f"terrains={sorted(df['terrain'].unique())} levels={sorted(df['terrain_level'].unique())}")
    for t in TERRAINS:
        print(f"[terrain-report]   {t}: {TERRAIN_NOTE[t]}")
    print(f"[terrain-report] out_dir={out_dir}")

    overview(df, out_dir)
    torque_heatmap_terrain(df, out_dir)
    torque_speed_terrain(df, out_dir)
    hildebrand_terrain(df, out_dir)
    cot_by_terrain(df, out_dir)
    attitude_terrain(df, out_dir)
    print("[terrain-report] done.")


if __name__ == "__main__":
    main()
