"""Natural half workspace of the Bennett leg from the one-sided knee
(user's construction, replaces the manual half-cut-away):

  * the model's q2 IS the thigh-calf bar angle (calf is a child of thigh:
    calf absolute rotation = q1 + q2, verified against bennett_1.xml tree);
  * with the bar angle q2 fixed, advancing q1 rotates the WHOLE leg (closed
    chain shape unchanged) rigidly about the coaxial drive axis -- exact
    identity, passives unchanged (checked to 0.0e0 rad);
  * so each fixed q2 sweeps an exact horizontal circle; the workspace is the
    surface of revolution of the branch-tracked generator g(q2);
  * keeping the chirality (thigh left / calf right: q2 in [-pi, 0], the
    resting side) confines the generator to HALF the branch meridian -> the
    revolution is naturally half the workspace, no manual cutting.  The two
    boundary circles (q2 = 0 straight leg, q2 = -pi folded flat) are the
    natural cut edges.

Generator: continuation from the straight assembly (q2 = 0, passives 0)
along negative q2 -- the physical assembly branch (identity q5 = q2 holds;
the full-branch extent is q2 in [-358 deg, +68.5 deg], folding at both ends).

Display orientation identical to plot_bennett_workspace.py: FR leg (y mirror),
drive axis (0,-1,1)/sqrt2.  Axis limits match fig5_workspace (450 mm) so the
two figures compare directly.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bennett_leg_fk import rot_axis, solve_passive  # noqa: E402
from plot_bennett_workspace import (  # noqa: E402
    INK, MIR, OUT_DIR, SURFACE, U_PLOT,
    add_axis3d, add_axis_side, add_triad, finish_ortho, nice_lim, ortho_fill,
    style3d, three_view_fig,
)

GEN = OUT_DIR / "workspace_generator_branch.npz"
LIM = 450.0            # same as fig5_workspace for direct comparison
N_ROT = 280            # revolution steps over the full q1 cycle


def build_surface(g_mm, n_rot=N_ROT):
    """tips(q1, q2) = R(U_PLOT, q1) g(q2); q1 seam closed, q2 arc open."""
    q1s = np.linspace(0.0, 2.0 * np.pi, n_rot + 1)
    G = g_mm / 1000.0
    tips = np.empty((len(q1s), len(G), 3))
    for i, q1 in enumerate(q1s):
        R = rot_axis(U_PLOT, q1)[:3, :3]
        tips[i] = (R @ G.T).T
    return q1s, tips * 1000.0


def main():
    d = np.load(GEN)
    q2s, g = d["q2s"], d["g_tips"]                     # descending q2: 0 -> -pi
    order = np.argsort(q2s)                            # ascending: -pi -> 0
    q2s, g = q2s[order], g[order]
    g = g * MIR                                        # FR display orientation

    q1s, tips = build_surface(g)
    lim = LIM

    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(np.min(tips[:, :, 2]), np.max(tips[:, :, 2]))

    # boundary circles: q2 = -pi (row 0) and q2 = 0 (row -1), fully revolved
    edge_lo = tips[:, 0, :]                            # q2 = -pi
    edge_hi = tips[:, -1, :]                           # q2 = 0

    # rest pose (standing defaults q2 = -0.16), on the mirrored FR leg
    _, _, rest = solve_passive(0.0, -0.16, np.zeros(3))
    rest = rest * 1000.0 * MIR

    fig, ax3d, ax_side, ax_front = three_view_fig()
    add_axis3d(ax3d, lim)
    surf = ax3d.plot_surface(tips[:, :, 0], tips[:, :, 1], tips[:, :, 2],
                             cmap=cmap, norm=norm, rstride=2, cstride=2,
                             linewidth=0.1, edgecolor=(0, 0, 0, 0.14),
                             shade=False, antialiased=True, rasterized=True)
    for edge in (edge_lo, edge_hi):
        ax3d.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=INK, linewidth=1.8,
                  zorder=5)
    ax3d.scatter(*rest, color=INK, marker="*", s=150, depthshade=False,
                 edgecolors=SURFACE, linewidths=0.7, zorder=6)
    add_triad(ax3d, 0.22 * lim)
    style3d(ax3d)
    ax3d.set_xlabel("x (mm)", fontsize=8.5, color=INK)
    ax3d.set_ylabel("y (mm)", fontsize=8.5, color=INK)
    ax3d.set_zlabel("z (mm)", fontsize=8.5, color=INK)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=22, azim=-60)
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)

    ortho_fill(ax_side, tips, cmap, norm, xy=(1, 2))
    add_axis_side(ax_side, lim)
    for edge in (edge_lo, edge_hi):
        ax_side.plot(edge[:, 1], edge[:, 2], color=INK, linewidth=1.3)
    ax_side.scatter(rest[1], rest[2], color=INK, marker="*", s=150,
                    edgecolors=SURFACE, linewidths=0.7, zorder=6)
    finish_ortho(ax_side, "Side view (y-z)", lim)

    ortho_fill(ax_front, tips, cmap, norm, xy=(0, 2))
    for edge in (edge_lo, edge_hi):
        ax_front.plot(edge[:, 0], edge[:, 2], color=INK, linewidth=1.3)
    ax_front.scatter(rest[0], rest[2], color=INK, marker="*", s=150,
                     edgecolors=SURFACE, linewidths=0.7, zorder=6)
    finish_ortho(ax_front, "Front view (x-z)", lim)

    cb = fig.colorbar(surf, ax=ax3d, shrink=0.6, pad=0.06)
    cb.set_label("tip z (mm)", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK)
    cb.outline.set_edgecolor("#898781")

    fig.suptitle("Natural half workspace from the one-sided knee "
                 "(bar angle $q_2$ kept on the resting side, thigh left / calf right)\n"
                 "surface = full revolution of the branch generator $g(q_2)$, "
                 "$q_2 \\in [-\\pi, 0]$; boundary circles at $q_2 = 0$ and "
                 "$q_2 = -\\pi$; $\\bigstar$ = rest pose (FR)",
                 fontsize=11, color=INK)

    for name in ("fig5_workspace_natural.png", "fig5_workspace_natural.svg"):
        fig.savefig(OUT_DIR / name, dpi=300, facecolor=SURFACE)
        print(f"  wrote {name}")
    plt.close(fig)

    # quick stats
    r = np.linalg.norm(g - (g @ U_PLOT)[:, None] * U_PLOT, axis=1)
    z_ax = g @ U_PLOT
    print(f"meridian: axis radius {r.min():.1f}..{r.max():.1f} mm, "
          f"axial {z_ax.min():.1f}..{z_ax.max():.1f} mm")
    print(f"surface: tip |p| max {np.linalg.norm(tips, axis=2).max():.1f} mm, "
          f"z range [{tips[:,:,2].min():.1f}, {tips[:,:,2].max():.1f}] mm")
    print(f"rest star: {np.round(rest, 1)} mm")


if __name__ == "__main__":
    main()
