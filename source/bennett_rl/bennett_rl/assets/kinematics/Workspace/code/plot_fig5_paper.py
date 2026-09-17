"""Paper-exact reproduction of Fig. 5 (Gu et al., MMT 176, 2022) -- the
theoretical workspace of the Bennett leg, generated the way the paper and the
user's Bennett_5R_Workspace_Fixed.m generate it:

  * the tip lies on a 1-D generator curve in the (r, w) plane about the drive
    axis (w = axial coordinate, r = axis radius) -- Appendix A, Eq. (A.1);
  * explicit parameterisation obtained from A.1:
        w(t) = b sin(beta) sin t
        r(t) = sqrt( (b cos(beta) sin t)^2 + (a + b cos t)^2 ),  t in [-pi/2, pi/2]
    (beta = 90 deg reduces this to the user's .m form r = a + b cos d,
    z = -b sin d sin alpha -- the "semicircular line" of the paper);
  * the workspace surface is the full revolution of that generator about the
    drive axis z1, tilted by the hip frame assembly angle sigma;

Module parameters come from the user's URDF (bennett_1.xml): thigh bar
a = |L1h - K| = 180.6 mm, shank bar b = |foot| = 215.2 mm, twist beta = 60 deg
(exact URDF angle U1^U1B), assembly sigma = -45 deg (drive axis (0,1,1)/sqrt2,
tilted 45 deg from gravity, shown as-is in the hip frame). One-sided knee per
the user's annotation: the generator keeps only the resting chirality
(t in [-pi/2, 0], axial w <= 0) -- the band toward +z1 (w > 0) is dropped --
and the kept half revolves the FULL 360 deg; where the straight-leg rim rises
above z = 0 it is kept (it still belongs to the axial lower half).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_bennett_workspace as pbw  # noqa: E402
from plot_bennett_workspace import (  # noqa: E402
    INK,
    OUT_DIR,
    SURFACE,
    add_triad,
    finish_ortho,
    nice_lim,
    ortho_fill,
    style3d,
)

# --- module parameters from the user's URDF (bennett_1.xml geometry) ---------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bennett_leg_fk import CALF_POS, FOOT_POS, L1_POS  # noqa: E402

A = float(np.linalg.norm(L1_POS - CALF_POS) * 1000.0)   # thigh bar (mm)
B = float(np.linalg.norm(FOOT_POS) * 1000.0)            # shank bar (mm)
BETA = np.radians(60.0)     # twist U1 vs U1B in the URDF (exactly 60 deg)
SIGMA = np.radians(-45.0)   # real hip-frame assembly: drive axis (0,1,1)/sqrt2,
#                            i.e. tilted 45 deg from gravity -- shown as-is.
# One-sided knee (per the user's annotation): "upper half" means the part of
# the band toward +z1 (axial upper half, w > 0 -- the opposite knee
# chirality). Keep ONLY the resting chirality w <= 0 (the real robot stands
# at axial -119 mm) and revolve the FULL 360 deg: the straight-leg rim then
# swings above z = 0 on part of the revolution and that is CORRECT and kept
# (it belongs to the lower-axial half). No world-z clipping.

N_GEN = 181        # generator samples over the semicircular line
N_ROT = 360        # revolution steps

# drive axis z1 = (0, -sin sigma, cos sigma); displayed pointing upward
U_AXIS = np.array([0.0, -np.sin(SIGMA), np.cos(SIGMA)])
if U_AXIS[2] < 0:
    U_DISP = -U_AXIS
else:
    U_DISP = U_AXIS
# orthonormal frame: e1 = x, e2 = (0, cos sigma, sin sigma), axis = z1
E2 = np.array([0.0, np.cos(SIGMA), np.sin(SIGMA)])


def generator(n=N_GEN):
    """(r, w) generator samples, mm -- the paper's semicircular line, kept on
    the resting chirality only (t in [-pi/2, 0]) so the revolved surface is
    the LOWER half of the torus (tip never rises above the straight-leg
    plane, like the real robot's one-sided knee)."""
    t = np.linspace(-np.pi / 2.0, 0.0, n)
    w = B * np.sin(BETA) * np.sin(t)
    r = np.sqrt((B * np.cos(BETA) * np.sin(t)) ** 2 + (A + B * np.cos(t)) ** 2)
    return r, w, t


PAPER_PURPLE = "#7b2d8b"   # triad + axis-label purple, paper Fig. 5 style


def ortho_mesh(ax, tips, xy=(1, 2), d_th=5, d_t=6):
    """Fine parameter-line mesh over an orthographic projection (paper style:
    the surface grid stays visible inside the filled band)."""
    i, j = xy
    for col in range(0, tips.shape[1], d_t):          # circle family (fixed t)
        ax.plot(tips[:, col, i], tips[:, col, j], color=INK,
                lw=0.22, alpha=0.28, rasterized=True, zorder=3)
    for row in range(0, tips.shape[0], d_th):         # generator family (fixed theta)
        ax.plot(tips[row, :, i], tips[row, :, j], color=INK,
                lw=0.22, alpha=0.28, rasterized=True, zorder=3)


def revolve(r, w, n_rot=N_ROT):
    """tips[theta, t, :] (mm): p = r (cosT e1 + sinT e2) + w z1, y-mirrored
    (FR display), real assembly: the axis is tilted 45 deg from gravity."""
    th = np.linspace(0.0, 2.0 * np.pi, n_rot + 1)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]   # (n_rot+1, 1)
    R, W = r[None, :], w[None, :]                     # (1, n_gen)
    x = R * c
    y = -(R * s * E2[1] + W * U_DISP[1])
    z = R * s * E2[2] + W * U_DISP[2]
    return np.stack([x, y, z], axis=2)


def main():
    global SIGMA, U_AXIS, U_DISP, E2
    if "--sigma" in sys.argv:  # scratch comparison renders
        SIGMA = np.radians(float(sys.argv[sys.argv.index("--sigma") + 1]))
        U_AXIS = np.array([0.0, -np.sin(SIGMA), np.cos(SIGMA)])
        U_DISP = U_AXIS if U_AXIS[2] >= 0 else -U_AXIS
        E2 = np.array([0.0, np.cos(SIGMA), np.sin(SIGMA)])
    r, w, t = generator()
    tips_full = revolve(r, w)  # full revolution about the tilted axis
    lim = nice_lim(tips_full)
    cmap = plt.get_cmap("turbo")     # user preference: back to turbo
    norm = plt.Normalize(tips_full[:, :, 2].min(), tips_full[:, :, 2].max())

    # paper layout: one large square-ish isometric panel on the left, exactly
    # as tall as the two stacked ortho panels on the right. Manual placement,
    # then a second pass aligns the TOP of the right column flush with the
    # topmost point actually drawn by the 3D panel (user request).
    fig = plt.figure(figsize=(10.4, 6.3), facecolor=SURFACE)
    ax3d = fig.add_axes([0.012, 0.040, 0.630, 0.835], projection="3d")
    RIGHT_X, RIGHT_W = 0.665, 0.330   # right column rect (fig fraction)
    RIGHT_BOT = 0.085                 # room for the front panel's x labels
    RIGHT_GAP = 0.035
    ax_side = fig.add_axes([RIGHT_X, 0.5, RIGHT_W, 0.35])    # fixed in pass 2
    ax_front = fig.add_axes([RIGHT_X, 0.05, RIGHT_W, 0.35])

    # ---- isometric (paper style: viridis surface + fine mesh, purple
    #      coordinate triad drawn on top of the surface) ----
    style3d(ax3d)
    ax3d.computed_zorder = False   # manual layering: axis < surface < triad
    ax3d.tick_params(labelsize=8)
    surf = ax3d.plot_surface(tips_full[:, :, 0], tips_full[:, :, 1],
                             tips_full[:, :, 2], cmap=cmap, norm=norm,
                             rstride=2, cstride=2, linewidth=0.15,
                             edgecolor=(0.02, 0.02, 0.08, 0.16), shade=False,
                             antialiased=True, rasterized=True, zorder=2,
                             alpha=0.82)
    # drive axis z1 (long, on top of the surface) + body triad
    # (no black rim/envelope lines -- user: only the light mesh on the surface)
    pbw.U_PLOT = U_DISP * np.array([1.0, -1.0, 1.0])  # mirrored, matches FR display
    Lz = 1.38 * lim                        # well past the surface on both ends
    uz = pbw.U_PLOT
    ax3d.plot([-Lz * uz[0], Lz * uz[0]], [-Lz * uz[1], Lz * uz[1]],
              [-Lz * uz[2], Lz * uz[2]], color=pbw.INK2, linewidth=0.9,
              linestyle=(0, (5, 4)), alpha=0.95, zorder=10)
    ax3d.text(1.03 * Lz * uz[0], 1.03 * Lz * uz[1], 1.03 * Lz * uz[2],
              "$z_1$", color=pbw.INK2, fontsize=8, ha="center", va="bottom",
              zorder=10)
    for d in np.eye(3):                    # slim shaft, prominent arrowhead
        ax3d.quiver(0, 0, 0, *(d * 0.34 * lim), color=PAPER_PURPLE,
                    linewidth=1.4, arrow_length_ratio=0.28, zorder=10)
    ax3d.set_box_aspect((1, 1, 1), zoom=1.02)   # cube fills its panel (paper)
    ax3d.set_proj_type("ortho")            # true isometric, no perspective
    ax3d.view_init(elev=30, azim=-45)
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    # z tick labels on the LEFT vertical edge (paper layout, like MATLAB)
    _zp = ax3d.zaxis._PLANES
    ax3d.zaxis._PLANES = (_zp[2], _zp[3], _zp[0], _zp[1], _zp[4], _zp[5])
    ax3d.set_xlabel("x (mm)", fontsize=11, color=PAPER_PURPLE,
                    fontstyle="italic", fontweight="bold")
    ax3d.set_ylabel("y (mm)", fontsize=11, color=PAPER_PURPLE,
                    fontstyle="italic", fontweight="bold")
    ax3d.set_zlabel("z (mm)", fontsize=11, color=PAPER_PURPLE,
                    fontstyle="italic", fontweight="bold")
    ax3d.text2D(0.0, 0.87, "Isometric view", transform=ax3d.transAxes,
                fontsize=11, color=INK, style="italic", zorder=10)

    # ---- orthographic projections (paper style: fill + fine mesh) ----
    ortho_fill(ax_side, tips_full, cmap, norm, xy=(1, 2))
    ax_side.collections[-1].set_alpha(0.82)   # same lightening as isometric
    ortho_mesh(ax_side, tips_full, xy=(1, 2))
    pbw.add_axis_side(ax_side, lim)
    # keep the z1 tag clear of the in-panel title: tuck it under the line end
    _z1 = ax_side.texts[-1]
    _z1.set_position((_z1.get_position()[0] + 8, _z1.get_position()[1] - 5))
    _z1.set_va("top")
    if abs(U_DISP[2]) > 0.99:  # vertical axis: label beside the line, not above
        ax_side.texts[-1].set_position((16, 0.42 * lim))
        ax_side.texts[-1].set_ha("left")
    finish_ortho(ax_side, "", lim)   # title inside the panel, top-left
    ax_side.text(0.0, 0.965, "Side view (y-z)", transform=ax_side.transAxes,
                 fontsize=10.5, color=INK, style="italic", va="top", zorder=10)
    ortho_fill(ax_front, tips_full, cmap, norm, xy=(0, 2))
    ax_front.collections[-1].set_alpha(0.82)
    ortho_mesh(ax_front, tips_full, xy=(0, 2))
    finish_ortho(ax_front, "", lim)
    ax_front.text(0.0, 0.965, "Front view (x-z)", transform=ax_front.transAxes,
                  fontsize=10.5, color=INK, style="italic", va="top", zorder=10)

    # no colorbar -- the paper reads the height straight off the viridis ramp

    # ---- pass 2: right column top flush with the topmost 3D point ----------
    fig.canvas.draw()   # realize the projection with the final view
    _pts = np.vstack([
        tips_full.reshape(-1, 3),
        np.array([[sx * lim, sy * lim, sz * lim]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]),
        np.array([-Lz * uz, Lz * uz]),
    ])
    _x2, _y2, _ = proj3d.proj_transform(_pts[:, 0], _pts[:, 1], _pts[:, 2],
                                        ax3d.get_proj())
    _disp = ax3d.transData.transform(np.column_stack([_x2, _y2]))
    _top = _disp[:, 1].max() / fig.bbox.height   # topmost drawn point, fig frac
    _h = (_top - RIGHT_BOT - RIGHT_GAP) / 2.0
    ax_side.set_position([RIGHT_X, _top - _h, RIGHT_W, _h])
    ax_front.set_position([RIGHT_X, RIGHT_BOT, RIGHT_W, _h])

    fig.suptitle("Theoretical workspace of the robotic Bennett leg "
                 "(paper Fig. 5; generator = rotating a semicircular line)\n"
                 f"module from URDF: a = {A:.1f} mm, b = {B:.1f} mm, "
                 f"$\\beta$ = {np.degrees(BETA):.0f}$^\\circ$; real assembly, "
                 "drive axis 45$^\\circ$ from gravity; one-sided knee: axial "
                 "lower half (w $\\leq$ 0) only", fontsize=9.5, color=INK)

    if "--sigma" in sys.argv:  # scratch render: separate files
        for name in (f"_scratch_sig{abs(SIGMA):.0f}.png",):
            fig.savefig(OUT_DIR / name, dpi=200, facecolor=SURFACE)
            print(f"  wrote {name}")
        plt.close(fig)
        print(f"sigma {np.degrees(SIGMA):.0f}: z [{tips_full[:,:,2].min():.1f}, "
              f"{tips_full[:,:,2].max():.1f}]")
        return
    for name in ("fig5_workspace_natural.png", "fig5_workspace_natural.svg"):
        fig.savefig(OUT_DIR / name, dpi=300, facecolor=SURFACE)
        print(f"  wrote {name}")
    plt.close(fig)

    print(f"generator: r {r.min():.1f}..{r.max():.1f} mm, "
          f"w {w.min():.1f}..{w.max():.1f} mm")
    print(f"surface : |xyz| max {np.abs(tips_full).max():.1f} mm, "
          f"z range [{tips_full[:,:,2].min():.1f}, "
          f"{tips_full[:,:,2].max():.1f}] mm, lim {lim:.0f}")


if __name__ == "__main__":
    main()
