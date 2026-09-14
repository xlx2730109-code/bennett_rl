"""Animated companion to ``fig5_workspace_natural.png``.

No existing file is touched: this script only reads other modules and writes a
new GIF.

What is drawn
-------------
* the backdrop surface, built by the same revolve construction as
  ``plot_fig5_paper.revolve``: with ``--surface real`` (default) it is the
  REAL workspace -- the true _1/_2 joint locus over the XML calf range,
  revolved about the drive axis -- so the marker circle lies EXACTLY on it;
  with ``--surface paper`` it is the paper Fig. 5 analytic generator;
* on top of it the REAL closed-chain leg from ``bennett_leg_fk`` -- ONLY the
  four links thigh + calf + _1 + _2.  _3 is the redundant closure link of the
  parallelogram and is deliberately NOT drawn (user instruction);
* the skeleton is a CLOSED spatial four-bar (user instruction): the thigh
  and calf cranks both root at the ORIGIN and lie in the plane of the
  workspace's largest circle (their real axial offset is collapsed), while
  _1 and _2 keep their true 3D position below that plane.  _2 closes
  directly onto the calf end, so the redundant closure link _3 is absorbed
  and NOT drawn;
* the moving marker is the _1/_2 joint (p2 in ``bennett_leg_fk`` terms), i.e.
  the point where links _1 and _2 meet, at its true 3D position;
* the NATIVE joint axes at the two crank ends (user instruction): short
  segments through p1 along the thigh<->_1 hinge axis (U1B) and through p3
  along the calf-end hinge axis (U3), carried through the FK chain each
  frame.

Motion (user program: the marker traces a generator circle)
------------------------------------------
    q1 = 2 pi s                      (one full revolution about z1),  s in [0, 1)
    q2 = q2_c + q2_amp sin(2 pi m s) (q2_amp = 0 by default, i.e. q2 fixed)

With q2 fixed the passive geometry is frozen, so the q1 revolution carries
the whole leg about the drive axis as a RIGID body and p2 traces an EXACT
circle about z1 -- one generator circle of the workspace surface.  The loop
closes after one revolution.  q2_amp > 0 adds m gentle wobbles per
revolution (the leg articulates; the circle becomes approximate).

The passive angles are solved per frame with warm starting, so the closure
stays on one physical assembly branch (residual printed at run time).

Honest note on the surface
--------------------------
``--surface real`` (default) revolves the marker's OWN locus, so the marker
circle lies exactly on the backdrop -- nothing is faked.  ``--surface paper``
keeps the paper's THEORETICAL workspace instead: the real Urdf_Bennett_3
closed chain cannot reach it (p2 stays ~100 mm away in the lower half; the
real foot hugs it to ~9 mm only near q2 = +0.48).  The two backdrops differ
in shape -- that difference is the real-vs-theory gap this figure documents.

Usage:
  python scripts/analysis/plot_fig5_paper_anim.py
  python scripts/analysis/plot_fig5_paper_anim.py --surface paper
  python scripts/analysis/plot_fig5_paper_anim.py --q2_amp 0.1 --frames 120
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bennett_leg_fk as fk  # noqa: E402
import plot_bennett_workspace as pbw  # noqa: E402
import plot_fig5_paper as paper  # noqa: E402
from plot_bennett_workspace import INK, INK2, MUTED, OUT_DIR, SURFACE  # noqa: E402

A, B, BETA = paper.A, paper.B, paper.BETA           # mm, mm, rad (from the URDF)
Z1 = paper.U_DISP                                    # drive axis, body frame
E2 = paper.E2
MIRROR = np.array([1.0, -1.0, 1.0])                  # FR display, same as the figure
Z1_M = Z1 * MIRROR                                   # (0, -0.7071, +0.7071)
U1_M = fk.U1 * MIRROR
ZHAT = Z1_M / np.linalg.norm(Z1_M)                   # unit display drive axis
# add_axis3d / add_axis_side read the module global; pin it the way
# plot_fig5_paper patches it inside its own main().
pbw.U_PLOT = Z1_M

PURPLE = paper.PAPER_PURPLE
BLUE = "#2a78d6"        # driven links (thigh, calf)
GRAY = "#8a8880"        # passive links (_1, _2)
RED = "#d03b3b"         # the moving marker (the _1/_2 joint) and its trail
TRAIL_C = np.array([0.816, 0.231, 0.231])
AXIS_C = "#e0862c"      # native joint-axis segments at p1 / p3

# (from, to, colour, lw) -- the four links as a CLOSED spatial four-bar
# (user instruction): the thigh and calf cranks root at the origin and lie
# in the plane of the workspace's largest circle; _1 and _2 keep their true
# 3D position below that plane, and _2 closes directly onto the calf end
# (the redundant closure link _3 is absorbed there and NOT drawn).
LINKS = (
    ("hip", "p1", BLUE, 6.6),        # thigh (q1 crank, radial, in-plane)
    ("hip", "p3", BLUE, 6.6),        # calf  (q2 crank, radial, in-plane)
    ("p1", "p2", GRAY, 5.6),         # _1 (spatial, dips below the plane)
    ("p2", "p3", GRAY, 5.0),         # _2, closing onto the calf end
)
JOINTS = (("hip", 54, INK), ("p1", 54, GRAY),
          ("p3", 48, GRAY),
          ("p2", 165, RED))


# ------------------------------------------------------- real leg geometry ---
def leg_body(q2, x0=np.zeros(3)):
    """Real closed-chain leg at q1 = 0, hip frame, mm, NOT yet mirrored.

    The passive angles depend only on q2 (the two drive axes are coaxial, so
    q1 is a pure rigid rotation of everything downstream), which is why the
    caller can solve once and then just rotate.  ``x0`` warm-starts the
    passive solve so consecutive q2 values stay on one assembly branch.
    """
    qp, err, _ = fk.solve_passive(0.0, q2, x0)
    q3, q4, q5 = qp
    Tt = fk.rot_axis(fk.U1, 0.0)
    Tc = Tt @ fk.tf(fk.CALF_POS) @ fk.rot_axis(fk.U1, q2)
    T3 = Tc @ fk.tf(fk.P3_POS) @ fk.rot_axis(fk.U3, q3)
    T1 = Tt @ fk.tf(fk.L1_POS) @ fk.rot_axis(fk.U1B, q4)
    T2 = T1 @ fk.tf(fk.L2_POS) @ fk.rot_axis(fk.U2, q5)

    def g(T, p):
        return (T @ np.append(p, 1.0))[:3] * 1000.0

    pts = {"hip": np.zeros(3), "knee": g(Tt, fk.CALF_POS), "p1": g(Tt, fk.L1_POS),
           "p3": g(Tc, fk.P3_POS), "p2": g(T1, fk.L2_POS),
           "ankle": g(T3, fk.A3_LOCAL), "foot": g(T1, fk.FOOT_POS)}
    # native joint-axis directions at the crank ends (hip frame, q1 = 0):
    # p1 rides the thigh<->_1 hinge (axis U1B), p3 the calf-end hinge (U3)
    dirs = (Tt[:3, :3] @ fk.U1B, Tc[:3, :3] @ fk.U3)
    return pts, dirs, qp, err


def to_display(pts, q1):
    """Rotate the q1=0 configuration about the drive axis, then mirror."""
    R = fk.rot_axis(fk.U1, q1)[:3, :3]
    return {k: (R @ v) * MIRROR for k, v in pts.items()}


def dir_display(d, q1):
    """Same rigid rotation + mirror as ``to_display``, for a direction."""
    R = fk.rot_axis(fk.U1, q1)[:3, :3]
    return (R @ d) * MIRROR


def flatten(pts):
    """Drop each point's component along the drive axis (projection onto the
    plane through the origin that carries the workspace's largest circle).
    Applied to the crank ends p1/p3 only, so the two cranks are radial bars
    in that plane (user instruction); _1/_2 stay truly 3D."""
    return {k: v - (v @ ZHAT) * ZHAT for k, v in pts.items()}


# --------------------------------------------------- analytic backdrop grid ---
U1HAT = fk.U1 / np.linalg.norm(fk.U1)     # drive axis, body frame (un-mirrored)


def real_generator(n_gen):
    """The marker's TRUE locus over the XML calf range, as (r, w) about z1.

    Revolving this curve about the drive axis is the REAL workspace of the
    _1/_2 joint -- the surface the marker actually lives on.  The passive
    solve is warm-started sequentially from q2_min upward (one physical
    assembly branch) with the cold-start fallback for the spurious q=0
    stationary point.
    """
    q2s = np.linspace(-0.90, 0.55, n_gen)          # XML calf joint range
    r_g, w_g, x0, worst = [], [], np.zeros(3), 0.0
    for q2 in q2s:
        qp, err, _ = fk.solve_passive(0.0, float(q2), x0)
        if err > 1e-9:
            qp, err, _ = fk.solve_passive(0.0, float(q2), np.zeros(3))
        x0 = qp
        worst = max(worst, err)
        q3, q4, q5 = qp
        T1 = (fk.rot_axis(fk.U1, 0.0) @ fk.tf(fk.L1_POS)
              @ fk.rot_axis(fk.U1B, q4))
        p = (T1 @ np.append(fk.L2_POS, 1.0))[:3] * 1000.0   # marker, body frame
        w = float(p @ U1HAT)
        r_g.append(float(np.linalg.norm(p - w * U1HAT)))
        w_g.append(w)
    return np.array(r_g), np.array(w_g), worst


def surface_grid(n_gen, n_rot, r_g=None, w_g=None):
    """The backdrop as a surface of revolution about z1.

    Same construction as plot_fig5_paper.revolve; the generator is the
    paper's analytic curve unless (r_g, w_g) are given (the REAL marker
    locus from ``real_generator``).
    """
    if r_g is None:
        r_g, w_g, _ = paper.generator(n_gen)
    th = np.linspace(0.0, 2.0 * np.pi, n_rot + 1)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    R, W = r_g[None, :], w_g[None, :]
    return np.stack([R * c, -(R * s * E2[1] + W * Z1[1]),
                     R * s * E2[2] + W * Z1[2]], axis=2)


def gap_to_surface(points, r_an=None, w_an=None):
    """Nearest distance (mm) from points to a generator band in (r, w)."""
    if r_an is None:
        t = np.linspace(-np.pi / 2.0, 0.0, 3001)
        r_an = np.sqrt((B * np.cos(BETA) * np.sin(t)) ** 2
                       + (A + B * np.cos(t)) ** 2)
        w_an = B * np.sin(BETA) * np.sin(t)
    p = np.atleast_2d(points)
    w = p @ Z1_M
    r = np.linalg.norm(p - w[:, None] * Z1_M, axis=1)
    return np.array([np.hypot(r_an - rr, w_an - ww).min()
                     for rr, ww in zip(r, w)])


# ---------------------------------------------------------------- canvas ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q2_c", type=float, default=-0.20,
                    help="rad, fixed calf angle -- places the marker circle "
                         "(XML calf range [-0.9, +0.55])")
    ap.add_argument("--q2_amp", type=float, default=0.0,
                    help="rad, calf wobble amplitude (0 = exact circle)")
    ap.add_argument("--wobble", type=int, default=2,
                    help="q2 wobbles per revolution (used if q2_amp > 0)")
    ap.add_argument("--axis_half", type=float, default=35.0,
                    help="mm, half-length of the native joint-axis segments")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--path", type=int, default=360, help="samples on the closed path")
    ap.add_argument("--trail_frac", type=float, default=0.20)
    ap.add_argument("--surface", choices=("real", "paper"), default="real",
                    help="backdrop: REAL marker-locus surface (default) or "
                         "the paper Fig. 5 analytic surface")
    ap.add_argument("--n_rot", type=int, default=190, help="revolution steps for the surface")
    ap.add_argument("--n_gen", type=int, default=126, help="generator samples for the surface")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--gif", default="fig5_workspace_real_anim.gif")
    args = ap.parse_args()

    t0 = time.time()
    print(f"  URDF links: a = {A:.2f} mm (knee->p1, thigh bar), "
          f"b = {B:.2f} mm (p1->foot, _1 bar), beta = {np.degrees(BETA):.1f} deg")
    print(f"  |L2_POS| = {np.linalg.norm(fk.L2_POS) * 1000:.2f} mm  (p1->p2, "
          f"the _1/_2 link the marker rides)")

    # ---- backdrop generator (before the path: the gap check needs it) -----
    if args.surface == "real":
        r_g, w_g, gen_err = real_generator(args.n_gen)
        print(f"  REAL generator: q2 in [-0.90, +0.55] rad, "
              f"residual {gen_err:.1e} m, r in [{r_g.min():.0f}, {r_g.max():.0f}] mm, "
              f"w in [{w_g.min():.0f}, {w_g.max():.0f}] mm")

    # ---- the path p2 traces (user program: a generator circle) ------------
    # q1 makes ONE full revolution about the drive axis while q2 stays put,
    # so the leg sweeps around z1 as a rigid body and p2 traces an exact
    # circle about z1 (a generator circle of the workspace).  q2_amp > 0
    # superposes gentle wobbles; both programs close after one revolution.
    tau = np.linspace(0.0, 1.0, args.path, endpoint=False)
    q1s = 2.0 * np.pi * tau
    q2s = args.q2_c + args.q2_amp * np.sin(args.wobble * 2.0 * np.pi * tau)

    path, x0, worst_err = [], np.zeros(3), 0.0
    for q1, q2 in zip(q1s, q2s):
        body, dirs, qp, err = leg_body(float(q2), x0)
        if err > 1e-9:
            # warm start fell into the spurious stationary point at q=0
            # (zero gradient, ~cm-scale residual): re-solve from scratch.
            body, dirs, qp, err = leg_body(float(q2), np.zeros(3))
        x0 = qp
        worst_err = max(worst_err, err)
        disp = to_display(body, float(q1))
        flat = flatten({"p1": disp["p1"], "p3": disp["p3"]})
        path.append({"hip": disp["hip"], "p1": flat["p1"], "p3": flat["p3"],
                     "p2": disp["p2"],      # cranks in-plane, _1/_2 truly 3D
                     "ax1": dir_display(dirs[0], float(q1)),
                     "ax3": dir_display(dirs[1], float(q1))})
    tip_path = np.array([p["p2"] for p in path])
    circle_r = np.linalg.norm(tip_path - (tip_path @ ZHAT)[:, None] * ZHAT,
                              axis=1)
    print(f"  q2 range [{q2s.min():+.3f}, {q2s.max():+.3f}] rad "
          f"(XML calf range [-0.9, +0.55])")
    print(f"  closure residual over the path: {worst_err:.1e} m")
    print(f"  p2 z range [{tip_path[:, 2].min():.1f}, {tip_path[:, 2].max():.1f}] mm, "
          f"{(tip_path[:, 2] < 0).mean() * 100:.0f}% of the orbit below z = 0")
    if args.surface == "real":
        gap = gap_to_surface(tip_path, r_g, w_g)
        print(f"  p2 distance to the REAL surface: min {gap.min():.2f}, "
              f"mean {gap.mean():.2f}, max {gap.max():.2f} mm  "
              f"(0 = the circle lies on it)")
    else:
        gap = gap_to_surface(tip_path)
        print(f"  p2 distance to the analytic surface: min {gap.min():.1f}, "
              f"mean {gap.mean():.1f}, max {gap.max():.1f} mm  (the real closed chain "
              f"does not reach the paper's theoretical surface)")

    # ---- static backdrop --------------------------------------------------
    if args.surface == "real":
        tips = surface_grid(args.n_gen, args.n_rot, r_g, w_g)
    else:
        tips = surface_grid(args.n_gen, args.n_rot)
    lim = pbw.nice_lim(tips)
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(tips[:, :, 2].min(), tips[:, :, 2].max())

    # ---- figure: the reference figure's manual layout, verbatim -----------
    fig = plt.figure(figsize=(10.4, 6.3), facecolor=SURFACE)
    ax3d = fig.add_axes([0.012, 0.040, 0.630, 0.835], projection="3d")
    RIGHT_X, RIGHT_W = 0.665, 0.330
    RIGHT_BOT, RIGHT_GAP = 0.085, 0.035
    ax_side = fig.add_axes([RIGHT_X, 0.5, RIGHT_W, 0.35])
    ax_front = fig.add_axes([RIGHT_X, 0.05, RIGHT_W, 0.35])

    pbw.style3d(ax3d)
    ax3d.computed_zorder = False
    ax3d.tick_params(labelsize=8)
    ax3d.plot_surface(tips[:, :, 0], tips[:, :, 1], tips[:, :, 2],
                      cmap=cmap, norm=norm, rstride=args.stride,
                      cstride=args.stride, linewidth=0.15,
                      edgecolor=(0.02, 0.02, 0.08, 0.16), shade=False,
                      antialiased=True, rasterized=True, zorder=2, alpha=0.82)
    Lz = 1.38 * lim
    uz = Z1_M
    ax3d.plot([-Lz * uz[0], Lz * uz[0]], [-Lz * uz[1], Lz * uz[1]],
              [-Lz * uz[2], Lz * uz[2]], color=INK2, linewidth=0.9,
              linestyle=(0, (5, 4)), alpha=0.95, zorder=1)
    ax3d.text(1.03 * Lz * uz[0], 1.03 * Lz * uz[1], 1.03 * Lz * uz[2], "$z_1$",
              color=INK2, fontsize=8, ha="center", va="bottom", zorder=1)
    for d in np.eye(3):
        ax3d.quiver(0, 0, 0, *(d * 0.34 * lim), color=PURPLE, linewidth=1.4,
                    arrow_length_ratio=0.28, zorder=10)
    ax3d.set_box_aspect((1, 1, 1), zoom=1.02)
    ax3d.set_proj_type("ortho")
    ax3d.view_init(elev=30, azim=-45)
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    zp = ax3d.zaxis._PLANES
    ax3d.zaxis._PLANES = (zp[2], zp[3], zp[0], zp[1], zp[4], zp[5])
    for axis, nm in ((ax3d.xaxis, "x"), (ax3d.yaxis, "y"), (ax3d.zaxis, "z")):
        axis.set_label_text(f"{nm} (mm)", fontsize=11, color=PURPLE,
                            fontstyle="italic", fontweight="bold")
    ax3d.text2D(0.0, 0.80, "Isometric view", transform=ax3d.transAxes,
                fontsize=11, color=INK, style="italic", zorder=10)

    pbw.ortho_fill(ax_side, tips, cmap, norm, xy=(1, 2))
    ax_side.collections[-1].set_alpha(0.82)
    paper.ortho_mesh(ax_side, tips, xy=(1, 2), d_th=8, d_t=8)
    pbw.add_axis_side(ax_side, lim)
    z1_txt = ax_side.texts[-1]
    z1_txt.set_position((z1_txt.get_position()[0] + 8,
                         z1_txt.get_position()[1] - 5))
    z1_txt.set_va("top")
    pbw.finish_ortho(ax_side, "", lim)
    ax_side.text(0.0, 0.965, "Side view (y-z)", transform=ax_side.transAxes,
                 fontsize=10.5, color=INK, style="italic", va="top", zorder=10)

    pbw.ortho_fill(ax_front, tips, cmap, norm, xy=(0, 2))
    ax_front.collections[-1].set_alpha(0.82)
    paper.ortho_mesh(ax_front, tips, xy=(0, 2), d_th=8, d_t=8)
    pbw.finish_ortho(ax_front, "", lim)
    ax_front.text(0.0, 0.965, "Front view (x-z)", transform=ax_front.transAxes,
                  fontsize=10.5, color=INK, style="italic", va="top", zorder=10)

    # pass 2: right column top flush with the topmost 3D point (as the figure)
    fig.canvas.draw()
    pts = np.vstack([tips.reshape(-1, 3),
                     np.array([[sx * lim, sy * lim, sz * lim]
                               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]),
                     np.array([-Lz * uz, Lz * uz])])
    x2, y2, _ = proj3d.proj_transform(pts[:, 0], pts[:, 1], pts[:, 2], ax3d.get_proj())
    disp = ax3d.transData.transform(np.column_stack([x2, y2]))
    top = disp[:, 1].max() / fig.bbox.height
    h = (top - RIGHT_BOT - RIGHT_GAP) / 2.0
    ax_side.set_position([RIGHT_X, top - h, RIGHT_W, h])
    ax_front.set_position([RIGHT_X, RIGHT_BOT, RIGHT_W, h])

    head = ("Bennett leg workspace: REAL surface = the _1/_2 locus revolved"
            " about z1 -- the marker circle lies on it"
            if args.surface == "real" else
            "Bennett leg workspace, paper Fig. 5 surface + the REAL"
            " closed-chain leg")
    fig.suptitle(
        head + "\n"
        f"four links thigh / calf / _1 / _2 from Urdf_Bennett_3 "
        f"(a = {A:.1f} mm, b = {B:.1f} mm, $\\beta$ = {np.degrees(BETA):.0f}$^\\circ$)\n"
        "marker = _1/_2 joint on a q2-fixed, q1-revolution circle about z1;"
        "   cranks in the rim plane, loop closed on the calf end;"
        "   native axes at p1 / p3",
        fontsize=9.5, color=INK)

    # ---- animated artists -------------------------------------------------
    def ax3line(color, lw, ls="-", z=8, alpha=1.0):
        (ln,) = ax3d.plot([], [], [], color=color, lw=lw, ls=ls, alpha=alpha,
                          solid_capstyle="round", zorder=z)
        return ln

    def ax2line(ax, color, lw, ls="-", z=8, alpha=1.0):
        (ln,) = ax.plot([], [], color=color, lw=lw, ls=ls, alpha=alpha,
                        solid_capstyle="round", zorder=z)
        return ln

    seg3, seg_s, seg_f = [], [], []
    for k, (a_, b_, c_, lw) in enumerate(LINKS):
        seg3.append(ax3line(c_, lw, z=8 + 0.1 * k))
        seg_s.append(ax2line(ax_side, c_, 0.80 * lw, z=8 + 0.1 * k))
        seg_f.append(ax2line(ax_front, c_, 0.80 * lw, z=8 + 0.1 * k))

    dots3 = [ax3d.scatter([], [], [], s=s, c=c, edgecolor=SURFACE, linewidth=1.1,
                          depthshade=False, zorder=11 + 0.1 * i)
             for i, (_, s, c) in enumerate(JOINTS)]
    dot_s = [ax_side.scatter([], [], s=0.55 * s, c=c, edgecolor=SURFACE,
                             linewidth=0.9, zorder=10) for _, s, c in JOINTS]
    dot_f = [ax_front.scatter([], [], s=0.55 * s, c=c, edgecolor=SURFACE,
                              linewidth=0.9, zorder=10) for _, s, c in JOINTS]

    # native joint-axis segments through the crank ends (p1: thigh<->_1, U1B;
    # p3: calf-end hinge, U3) -- true directions, anchored on the drawn joints
    axis3 = [ax3line(AXIS_C, 3.2, z=9.5) for _ in range(2)]
    axis_s = [ax2line(ax_side, AXIS_C, 2.6, z=9.5) for _ in range(2)]
    axis_f = [ax2line(ax_front, AXIS_C, 2.6, z=9.5) for _ in range(2)]

    # static faded guide of the whole orbit
    for ax, ix in ((ax3d, None), (ax_side, 1), (ax_front, 0)):
        if ax is ax3d:
            ax.plot(tip_path[:, 0], tip_path[:, 1], tip_path[:, 2], color=RED,
                    lw=1.0, ls=(0, (4, 3)), alpha=0.35, zorder=6)
        else:
            ax.plot(tip_path[:, ix], tip_path[:, 2], color=RED, lw=1.0,
                    ls=(0, (4, 3)), alpha=0.35, zorder=6)

    comet3d = Line3DCollection([np.zeros((2, 3))], linewidths=2.8, zorder=13)
    ax3d.add_collection3d(comet3d)
    comet_s = LineCollection([np.zeros((2, 2))], linewidths=2.6, zorder=13)
    ax_side.add_collection(comet_s)
    comet_f = LineCollection([np.zeros((2, 2))], linewidths=2.6, zorder=13)
    ax_front.add_collection(comet_f)

    hud = ax3d.text2D(0.015, 0.995, "", transform=ax3d.transAxes, fontsize=8.5,
                      color=INK2, va="top", family="monospace", zorder=20)

    # freeze the view (the animated collections must not re-autoscale)
    ax3d.autoscale(False)
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    for ax in (ax_side, ax_front):
        ax.set_autoscale_on(False)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    # ---- frames -----------------------------------------------------------
    n = args.frames
    window = max(2, int(args.trail_frac * args.path))

    def update(i):
        k = int(round(i * args.path / n)) % args.path
        P = path[k]
        q1, q2 = float(q1s[k]), float(q2s[k])

        for (a_, b_, _c, _lw), ln3, lns, lnf in zip(LINKS, seg3, seg_s, seg_f):
            pa, pb = P[a_], P[b_]
            ln3.set_data_3d([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]])
            lns.set_data([pa[1], pb[1]], [pa[2], pb[2]])
            lnf.set_data([pa[0], pb[0]], [pa[2], pb[2]])
        for (nm, _s, _c), d3, ds, df in zip(JOINTS, dots3, dot_s, dot_f):
            p = P[nm]
            d3._offsets3d = ([p[0]], [p[1]], [p[2]])
            ds.set_offsets(np.array([[p[1], p[2]]]))
            df.set_offsets(np.array([[p[0], p[2]]]))

        for (nm, dn), l3, ls_, lf_ in zip((("p1", "ax1"), ("p3", "ax3")),
                                          axis3, axis_s, axis_f):
            p, u = P[nm], args.axis_half * P[dn]
            a, b = p + u, p - u
            l3.set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
            ls_.set_data([a[1], b[1]], [a[2], b[2]])
            lf_.set_data([a[0], b[0]], [a[2], b[2]])

        idx = (np.arange(k - window, k + 1)) % args.path
        segs = np.stack([tip_path[idx][:-1], tip_path[idx][1:]], axis=1)
        ramp = np.linspace(0.0, 1.0, len(segs))[:, None]
        rgba = np.concatenate([np.tile(TRAIL_C, (len(segs), 1)),
                               0.06 + 0.94 * ramp ** 2], axis=1)
        comet3d.set_segments(segs)
        comet3d.set_color(rgba)
        comet_s.set_segments(segs[:, :, [1, 2]])
        comet_s.set_color(rgba)
        comet_f.set_segments(segs[:, :, [0, 2]])
        comet_f.set_color(rgba)

        tip = P["p2"]
        hud.set_text(
            f"generator circle about z1, radius ~{circle_r.mean():.0f} mm\n"
            f"q1 (thigh, one revolution) = {np.degrees(q1) % 360:6.1f} deg\n"
            f"q2 (calf)                  = {np.degrees(q2):+7.1f} deg\n"
            f"_1/_2 joint                = ({tip[0]:+6.1f}, {tip[1]:+6.1f}, "
            f"{tip[2]:+6.1f}) mm")
        return ()

    anim = FuncAnimation(fig, update, frames=n, interval=1000.0 / args.fps,
                         blit=False, repeat=True)
    out = OUT_DIR / args.gif
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi,
              savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    print(f"  wrote {out.name} ({n} frames, {args.fps} fps, "
          f"{out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
