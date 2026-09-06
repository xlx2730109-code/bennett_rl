"""Paper Fig. 6 reproduction (Gu et al., MMT 176, 2022): manipulability of
the Bennett leg.

Left  -- the workspace torus (same generator/revolve machinery as Fig. 5,
         URDF dimensions) with the real closed-chain linkage drawn at a chosen
         pose (q1, q2), the tangent plane of the workspace at the foot point,
         and the manipulability ellipse: the image of the unit velocity circle
         in actuator space through the actuator Jacobian
             Ja(q) = dF/d(q1, q2)  (3x2, finite differences on the FK with the
             passive closure solved by Newton iterations);
         its semi-axes are sqrt(lambda_max), sqrt(lambda_min) of Ja Ja^T and
         it lies in the tangent plane BY CONSTRUCTION (the plane is spanned by
         the two columns of Ja).  Manipulability mo = sqrt(lambda1 lambda2)
         = |det Ja|.
Right -- the paper's schematic: unit circle in actuator space -> Ja(q) ->
         the ellipse on the tangent plane (inset A, gray box).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_bennett_workspace as pbw  # noqa: E402
import plot_fig5_paper as f5  # noqa: E402
from bennett_leg_fk import (  # noqa: E402
    A3_LOCAL,
    CALF_POS,
    FOOT_POS,
    L1_POS,
    L2_POS,
    P3_POS,
    U1,
    U1B,
    U2,
    U3,
    rot_axis,
    solve_passive,
    tf,
)
from plot_bennett_workspace import INK, OUT_DIR, SURFACE, nice_lim, style3d  # noqa: E402

# --- pose of the drawn linkage (the paper puts the leg front-low, folded) ---
Q1_DEG = 50.0      # hip revolution about the drive axis
Q2_DEG = -18.0     # thigh-shank bar angle (near-standing, slightly bent)
ELL_SCALE = 0.45   # display scale of the ellipse (semi-axis S1 * scale, mm)

C_THIGH = "#d13b2c"   # red thigh bar (paper)
C_GREEN = "#1f9d4d"   # green shank/calf bars (paper V)
C_BLUE = "#0b63b8"    # blue closure link
C_PLANE = "0.62"      # tangent-plane gray
C_BOXGRAY = "#8b8b8b"
C_ORANGE = "#f0a500"
C_ARROW = "#a8a8a8"


def chain(q1, q2):
    """All linkage anchor points (hip frame, m) at actuator pose (q1, q2)."""
    x, err, _ = solve_passive(q1, q2)
    if err > 1e-8:
        x, err, _ = solve_passive(q1, q2)
    assert err < 1e-8, f"closure did not converge at ({q1}, {q2}): {err}"
    q3, q4, q5 = x
    Tt = rot_axis(U1, q1)
    Tc = Tt @ tf(CALF_POS) @ rot_axis(U1, q2)
    T3 = Tc @ tf(P3_POS) @ rot_axis(U3, q3)
    T1 = Tt @ tf(L1_POS) @ rot_axis(U1B, q4)
    T2 = T1 @ tf(L2_POS) @ rot_axis(U2, q5)
    P = lambda T, p: (T @ np.append(p, 1.0))[:3]
    return {"O": np.zeros(3), "K": P(Tt, CALF_POS), "L1h": P(Tt, L1_POS),
            "P3h": P(Tc, P3_POS), "L2h": P(T1, L2_POS), "A": P(T3, A3_LOCAL),
            "F": P(T1, FOOT_POS)}


def disp(p):
    """hip frame (m) -> figure display coords used by Fig. 5 (FR mirror, mm)."""
    return np.array([p[0], -p[1], p[2]]) * 1000.0


def jac_act(q1, q2, h=1e-6):
    """Actuator Jacobian dF/d(q1, q2), 3x2, mm/rad (central differences)."""
    J = np.zeros((3, 2))
    for i in (0, 1):
        d = np.zeros(2)
        d[i] = h
        fp = disp(chain(*(np.array([q1, q2]) + d))["F"])
        fm = disp(chain(*(np.array([q1, q2]) - d))["F"])
        J[:, i] = (fp - fm) / (2.0 * h)
    return J


def draw_linkage(ax, c, lw=4.5, zorder=7):
    """The closed chain at pose c: red thigh, green shank+calf V, blue _2.
    Every point goes through disp() so it matches the Fig. 5 display frame."""
    d = {k: disp(p) for k, p in c.items()}
    seg = lambda a, b: ([d[a][0], d[b][0]], [d[a][1], d[b][1]],
                        [d[a][2], d[b][2]])
    ax.plot(*seg("O", "K"), color=C_THIGH, lw=lw + 1.0, zorder=zorder,
            solid_capstyle="round")
    ax.plot(*seg("L1h", "F"), color=C_GREEN, lw=lw, zorder=zorder,
            solid_capstyle="round")
    ax.plot(*seg("K", "P3h"), color=C_GREEN, lw=lw - 0.8, zorder=zorder,
            solid_capstyle="round")
    ax.plot(*seg("P3h", "A"), color=C_GREEN, lw=lw - 0.8, zorder=zorder,
            solid_capstyle="round")
    ax.plot(*seg("L2h", "A"), color=C_BLUE, lw=lw - 1.2, zorder=zorder,
            solid_capstyle="round")
    for k in ("O", "K", "L1h", "P3h", "L2h", "A"):
        ax.scatter(*d[k], s=16, color=INK, zorder=zorder + 1)
    ax.scatter(*d["F"], s=90, marker="*", color=INK, zorder=zorder + 1)


def main():
    global Q1_DEG, Q2_DEG
    if "--q1" in sys.argv:  # scratch pose probes
        Q1_DEG = float(sys.argv[sys.argv.index("--q1") + 1])
        Q2_DEG = float(sys.argv[sys.argv.index("--q2") + 1])
    q1, q2 = np.radians(Q1_DEG), np.radians(Q2_DEG)
    c = chain(q1, q2)
    F = disp(c["F"])                                     # foot point, mm

    # --- actuator Jacobian, ellipse axes ------------------------------------
    J = jac_act(q1, q2)                                  # mm/rad, 3x2
    M = J @ J.T
    lam, U = np.linalg.eigh(M)                           # ascending
    S = np.sqrt(np.clip(lam, 0.0, None))[::-1]           # S1 >= S2
    U = U[:, ::-1]
    mo = float(S[0] * S[1])
    print(f"pose q1={Q1_DEG} q2={Q2_DEG} deg  F={np.round(F, 1)} mm")
    print(f"Ja singular values sqrt(lam): {S[0]:.1f}, {S[1]:.1f} mm/rad  "
          f"mo={mo:.1f} mm^2/rad^2")

    # --- workspace torus (Fig. 5 machinery) ---------------------------------
    r, w, _ = f5.generator()
    tips = f5.revolve(r, w)
    lim = nice_lim(tips)
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(tips[:, :, 2].min(), tips[:, :, 2].max())

    fig = plt.figure(figsize=(13.2, 7.0), facecolor=SURFACE)
    ax3d = fig.add_axes([0.005, 0.0, 0.60, 0.97], projection="3d")
    style3d(ax3d)
    ax3d.computed_zorder = False
    ax3d.tick_params(labelsize=8)
    ax3d.plot_surface(tips[:, :, 0], tips[:, :, 1], tips[:, :, 2], cmap=cmap,
                      norm=norm, rstride=2, cstride=2, linewidth=0.15,
                      edgecolor=(0.02, 0.02, 0.08, 0.16), shade=False,
                      antialiased=True, rasterized=True, zorder=2, alpha=0.9)

    # tangent plane at F: spanned by the Ja columns (u1: max, u2: min)
    e1 = U[:, 0] * S[0] * ELL_SCALE
    e2 = U[:, 1] * S[1] * ELL_SCALE
    a1, a2 = 1.5, 1.9                                    # plane half-extents
    g1, g2 = np.meshgrid(np.linspace(-a1, a1, 2), np.linspace(-a2, a2, 2))
    plane = F[None, None, :] + g1[:, :, None] * e1[None, None, :] \
        + g2[:, :, None] * e2[None, None, :]
    ax3d.plot_surface(plane[:, :, 0], plane[:, :, 1], plane[:, :, 2],
                      color=C_PLANE, shade=False, zorder=4, alpha=0.95)

    # manipulability ellipse on that plane (thick black, paper)
    th = np.linspace(0.0, 2.0 * np.pi, 181)
    ell = F[None, :] + (np.cos(th)[:, None] * e1[None, :]
                        + np.sin(th)[:, None] * e2[None, :])
    ax3d.plot(ell[:, 0], ell[:, 1], ell[:, 2], color="black", lw=2.6,
              zorder=6)

    # linkage + drive axis + triad (drawn on top, front pose)
    draw_linkage(ax3d, c, zorder=7)
    pbw.U_PLOT = f5.U_DISP * np.array([1.0, -1.0, 1.0])
    L = 1.12 * lim
    uz = pbw.U_PLOT
    ax3d.plot([-L * uz[0], L * uz[0]], [-L * uz[1], L * uz[1]],
              [-L * uz[2], L * uz[2]], color=pbw.INK2, lw=0.9,
              linestyle=(0, (5, 4)), alpha=0.9, zorder=10)
    ax3d.text(1.06 * L * uz[0], 1.06 * L * uz[1], 1.06 * L * uz[2], "$z_1$",
              color=pbw.INK2, fontsize=8, ha="center", va="bottom", zorder=10)
    pbw.add_triad(ax3d, 0.30 * lim, zorder=10)

    ax3d.set_box_aspect((1, 1, 1), zoom=1.02)
    ax3d.set_proj_type("ortho")
    ax3d.view_init(elev=22, azim=-56)
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    _zp = ax3d.zaxis._PLANES
    ax3d.zaxis._PLANES = (_zp[2], _zp[3], _zp[0], _zp[1], _zp[4], _zp[5])
    ax3d.set_xlabel("x(mm)", fontsize=11, color=f5.PAPER_PURPLE,
                    fontstyle="italic", fontweight="bold")
    ax3d.set_ylabel("y(mm)", fontsize=11, color=f5.PAPER_PURPLE,
                    fontstyle="italic", fontweight="bold")
    ax3d.set_zlabel("z(mm)", fontsize=11, color=f5.PAPER_PURPLE,
                    fontstyle="italic", fontweight="bold")

    # ---- mini linkage inset (paper top-left, dashed gray box) --------------
    axl = fig.add_axes([0.055, 0.70, 0.14, 0.24], projection="3d")
    axl.computed_zorder = False
    axl.patch.set_visible(False)   # keep the inset interior paper-white-free
    draw_linkage(axl, c, lw=3.0, zorder=5)
    axl.view_init(elev=22, azim=-56)
    axl.set_proj_type("ortho")
    axl.set_box_aspect((1, 1, 1))
    pts = np.array([disp(p) for p in c.values()])   # tight box: fill the inset
    ctr = 0.5 * (pts.max(0) + pts.min(0))
    R = 0.60 * (pts.max(0) - pts.min(0)).max() + 8.0
    axl.set_xlim(ctr[0] - R, ctr[0] + R)
    axl.set_ylim(ctr[1] - R, ctr[1] + R)
    axl.set_zlim(ctr[2] - R, ctr[2] + R)
    axl.set_axis_off()
    pos = axl.get_position().expanded(1.10, 1.14)
    fig.add_artist(Rectangle((pos.x0, pos.y0), pos.width, pos.height,
                             fill=False, ec="0.55", lw=1.3,
                             linestyle=(0, (4, 3))))

    # ---- right column: unit circle -> Ja -> ellipse (paper schematics) -----
    axu = fig.add_axes([0.660, 0.545, 0.315, 0.365])
    axu.set_xlim(-1.55, 1.75)
    axu.set_ylim(-1.45, 1.85)
    axu.set_aspect("equal")
    for sp in axu.spines.values():
        sp.set_linestyle((0, (5, 3)))
        sp.set_linewidth(1.6)
        sp.set_color(INK)
    axu.set_xticks([])
    axu.set_yticks([])
    axu.set_facecolor(SURFACE)
    axu.add_patch(plt.Circle((0, 0), 0.85, fill=False, ec=INK, lw=2.2))
    axu.annotate("", xy=(0, 1.45), xytext=(0, 0.85),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.0))
    axu.annotate("", xy=(1.45, 0), xytext=(0.85, 0),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.0))
    axu.text(0.10, 1.47, r"$\dot{q}_2$", fontsize=15, color=INK)
    axu.text(1.30, -0.30, r"$\dot{q}_1$", fontsize=15, color=INK)

    fig.text(0.8175, 0.945, "Unit velocity circle\nin actuator space",
             ha="center", va="bottom", fontsize=13, color=INK)
    # Ja arrow between the two boxes
    axu.annotate("", xy=(0.790, 0.538), xytext=(0.790, 0.582),
                 xycoords="figure fraction", textcoords="figure fraction",
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6))
    fig.text(0.810, 0.550, r"$J_a(q)$", fontsize=14, color=INK,
             fontstyle="italic")

    axm = fig.add_axes([0.660, 0.115, 0.315, 0.385])
    axm.set_xlim(-1.45, 1.75)
    axm.set_ylim(-1.55, 1.65)
    axm.set_aspect("equal")
    axm.set_facecolor(C_BOXGRAY)
    for sp in axm.spines.values():
        sp.set_linestyle((0, (6, 3)))
        sp.set_linewidth(2.2)
        sp.set_color(C_ORANGE)
    axm.set_xticks([])
    axm.set_yticks([])
    e = S[1] / S[0]                                       # flat ellipse ratio
    axm.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec="none"))
    thm = np.linspace(0.0, 2.0 * np.pi, 181)
    axm.plot(e * np.sin(thm), np.cos(thm), color="black", lw=3.0)
    axm.annotate("", xy=(0, 1.42), xytext=(0, -1.42),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    axm.annotate("", xy=(1.42, 0), xytext=(-1.42, 0),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    axm.text(0.06, 1.44, r"$v_{max}$", fontsize=13, color=INK,
             fontstyle="italic")
    axm.text(1.16, -0.24, r"$v_{min}$", fontsize=13, color=INK,
             fontstyle="italic")
    # semi-axis callouts
    axm.annotate("", xy=(0.28, float(np.sqrt(max(1 - (0.28 / e) ** 2, 0)))),
                 xytext=(0.28, 0.0),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    axm.text(0.22, 0.55, r"$\sqrt{\lambda_{max}}$", fontsize=11, color=INK,
             ha="right")
    axm.annotate("", xy=(e, -0.42), xytext=(0.0, -0.42),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    axm.text(0.06, -0.60, r"$\sqrt{\lambda_{min}}$", fontsize=11, color=INK)
    fig.text(0.8175, 0.070, "Manipulability ellipse on the\ntangent plane",
             ha="center", va="top", fontsize=13, color=INK, fontweight="bold")

    # ---- gray connector arrows (drawn after the layout is final) -----------
    fig.canvas.draw()

    def ffrac(p3):
        x2, y2, _ = proj3d.proj_transform(p3[0], p3[1], p3[2],
                                          ax3d.get_proj())
        d = ax3d.transData.transform((x2, y2))
        return d / fig.bbox.size

    # orange dashed zoom rect around the tangent plane = region A (paper)
    corners = [F + s1 * a1 * e1 + s2 * a2 * e2
               for s1 in (-1.0, 1.0) for s2 in (-1.0, 1.0)]
    cf = np.array([ffrac(p) for p in corners])
    rx0, rx1 = cf[:, 0].min() - 0.012, cf[:, 0].max() + 0.012
    ry0, ry1 = cf[:, 1].min() - 0.016, cf[:, 1].max() + 0.016
    fig.add_artist(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False,
                             ec=C_ORANGE, lw=1.8, linestyle=(0, (5, 3)),
                             zorder=12))
    fig.text(rx1 + 0.006, ry1, "$\\mathcal{A}$", fontsize=15, color=C_ORANGE,
             fontstyle="italic", fontweight="bold", va="top")
    axu.annotate("", xy=(0.656, 0.36), xytext=(rx1, 0.5 * (ry0 + ry1)),
                 xycoords="figure fraction", textcoords="figure fraction",
                 arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=3.2,
                                 shrinkA=0, shrinkB=0))
    pl = ffrac(disp(c["K"]) + 0.0)
    axu.annotate("", xy=tuple(pl), xytext=(pos.x1, pos.y0 + 0.30 * pos.height),
                 xycoords="figure fraction", textcoords="figure fraction",
                 arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=3.2,
                                 shrinkA=0, shrinkB=4))

    for name in ("fig6_manipulability.png", "fig6_manipulability.svg"):
        fig.savefig(OUT_DIR / name, dpi=300, facecolor=SURFACE)
        print(f"  wrote {name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
