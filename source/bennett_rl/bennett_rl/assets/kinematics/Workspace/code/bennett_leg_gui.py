"""Interactive GUI for one Bennett closed-chain leg (hip frame, real model):

  * sliders for the two actuated angles: q1 (hip revolution, coaxial drive
    axis) and q2 (thigh-calf bar angle -- calf is a child of thigh, so q2 IS
    the bar angle);
  * the three passive closure angles (q3, q4, q5) are solved live so the
    parallelogram chain (_3 <-> _2) stays closed;
  * all links drawn: thigh bar, shank bar _1 (carries the foot), calf yoke,
    closure links _3 and _2, drive axis;
  * dragging a slider leaves the foot trail: fixed q2 + drag q1 -> exact
    horizontal circle (step 1); fixed q1 + drag q2 -> the generator arc
    (steps 2-3);
  * buttons: sweep q1 a full turn, sweep q2 over the physical branch,
    accumulate the workspace point cloud, clear traces.

Run:  D:/Conda/envs/env_isaaclab/python.exe scripts/analysis/bennett_leg_gui.py
      (add --smoke for a headless solver check)
"""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

for _be in ("tkagg", "qt5agg", "qtagg"):
    try:
        matplotlib.use(_be)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button, Slider  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
C_THIGH, C_SHANK, C_CALF = "#d13b2c", "#0b63b8", "#e8a13a"
C_L3, C_L2 = "#3f9d55", "#7b2d8b"
LIM = 0.42  # display half-width (m), hip frame

Q2_MAX_DEG = 40.0  # physical branch folds near +41 deg
Q2_MIN_DEG = -180.0  # q5 hits its passive limit here

WARM = np.zeros(3)  # warm start for the passive solve


def P(T, p):
    return (T @ np.append(p, 1.0))[:3]


def chain_points(q1, q2, x):
    """Every joint position (hip frame, m) needed to draw the linkage."""
    q3, q4, q5 = x
    Tt = rot_axis(U1, q1)
    Tc = Tt @ tf(CALF_POS) @ rot_axis(U1, q2)
    T3 = Tc @ tf(P3_POS) @ rot_axis(U3, q3)
    T1 = Tt @ tf(L1_POS) @ rot_axis(U1B, q4)
    T2 = T1 @ tf(L2_POS) @ rot_axis(U2, q5)
    return {
        "O": np.zeros(3),
        "K": P(Tt, CALF_POS),    # knee: calf hinge, on the drive axis
        "L1h": P(Tt, L1_POS),    # shank-bar hinge on the thigh
        "P3h": P(Tc, P3_POS),    # link _3 hinge on the calf yoke
        "L2h": P(T1, L2_POS),    # link _2 hinge on the shank bar
        "A": P(T3, A3_LOCAL),    # closure anchor (_3 tip == _2 tip)
        "F": P(T1, FOOT_POS),    # foot
    }


def solve_for(q1, q2):
    """Passive solve with warm start + cold fallback. None if unsolvable."""
    global WARM
    x, err, f = solve_passive(q1, q2, WARM)
    if err > 1e-8:
        x, err, f = solve_passive(q1, q2)
        if err > 1e-8:
            return None, err, None
    WARM = x
    return x, err, f


# ------------------------------------------------------------------ figure ---
fig = plt.figure(figsize=(13.2, 7.6), facecolor=SURFACE)
ax = fig.add_axes([0.0, 0.0, 0.70, 1.0], projection="3d")
ax.set_facecolor(SURFACE)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.set_pane_color((0.988, 0.988, 0.984, 1.0))
fig.text(0.355, 0.985, "Bennett 单腿闭链交互（hip 系）— q1 髋周转 / q2 杆夹角 / 足端轨迹",
         ha="center", fontsize=13, color=INK, weight="bold")

# static artists ---------------------------------------------------------
ax.plot([0, 0.33 * U1[0]], [0, 0.33 * U1[1]], [0, 0.33 * U1[2]],
        color=MUTED, lw=1.2, ls="--", label="驱动轴 (q1/q2 共轴)")
ax.plot([0, -0.33 * U1[0]], [0, -0.33 * U1[1]], [0, -0.33 * U1[2]],
        color=MUTED, lw=1.2, ls="--")
tri = 0.09
for vec, cc, lab in ((np.eye(3)[0], "#d13b2c", "x"),
                     (np.eye(3)[1], "#3f9d55", "y"),
                     (np.eye(3)[2], "#0b63b8", "z")):
    ax.plot([0, vec[0] * tri], [0, vec[1] * tri], [0, vec[2] * tri], color=cc, lw=1.4)

ln_thigh, = ax.plot([], [], [], color=C_THIGH, lw=5, solid_capstyle="round",
                    label="连杆1 大腿 thigh (O→膝)")
ln_shank, = ax.plot([], [], [], color=C_SHANK, lw=5, solid_capstyle="round",
                    label="连杆2 小腿 _1 (L1h→足端)")
ln_calf, = ax.plot([], [], [], color=C_CALF, lw=4, solid_capstyle="round",
                   label="连杆3 calf（含 _3, K→P3h→锚点）")
ln_l2, = ax.plot([], [], [], color=C_L2, lw=3, label="连杆4 _2 (L2h→锚点)")
(jt,) = ax.plot([], [], [], ls="none", marker="o", ms=4, mfc=INK, mec=INK,
                label="运动副")
(ft,) = ax.plot([], [], [], ls="none", marker="*", ms=15, mfc=INK, mec=SURFACE,
                mew=0.6, label="足端")

tr1_x, tr1_y, tr1_z = [], [], []
tr2_x, tr2_y, tr2_z = [], [], []
(tr1,) = ax.plot([], [], [], color=C_SHANK, lw=1.4, alpha=0.85,
                 label="轨迹: 固定 q2 转 q1（水平圆）")
(tr2,) = ax.plot([], [], [], color=C_THIGH, lw=1.4, alpha=0.85,
                 label="轨迹: 固定 q1 扫 q2（生成线）")
cloud_artists = []

ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_zlim(-LIM, LIM)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=14, azim=-58)
ax.set_xlabel("x (m)", fontsize=8, color=MUTED)
ax.set_ylabel("y (m)", fontsize=8, color=MUTED)
ax.set_zlabel("z (m)", fontsize=8, color=MUTED)
ax.tick_params(labelsize=7, colors=MUTED)
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)

# right panel ------------------------------------------------------------
txt = fig.text(0.735, 0.925, "", fontsize=8.8, color=INK,
               family=["Microsoft YaHei", "SimHei"], va="top")
fig.text(0.735, 0.335,
         "提示：固定 q2 拖 q1 → z轴 读数不变，\n"
         "足端扫出垂直于驱动轴的水平圆；\n"
         "固定 q1 拖 q2 → 足端沿生成线（半圆弧）\n"
         "移动，圆心随之升降（工作空间母线）。",
         fontsize=8.2, color=MUTED, va="top")

axq1 = fig.add_axes([0.755, 0.255, 0.215, 0.025])
axq2 = fig.add_axes([0.755, 0.205, 0.215, 0.025])
s_q1 = Slider(axq1, "q1 髋周转 (°)", -180.0, 180.0, valinit=math.degrees(0.08),
              valstep=0.05, color=C_SHANK)
s_q2 = Slider(axq2, "q2 杆夹角 (°)", Q2_MIN_DEG, Q2_MAX_DEG,
              valinit=math.degrees(-0.16), valstep=0.05, color=C_THIGH)
for sl in (s_q1, s_q2):
    sl.label.set_fontsize(8.5)


def _cap(lst):
    if len(lst) > 6000:
        del lst[::2]


def update(_=None):
    q1, q2 = math.radians(s_q1.val), math.radians(s_q2.val)
    x, err, f = solve_for(q1, q2)
    if x is None:
        txt.set_text(f"求解失败（接近折叠） err={err:.1e}\n请往回拖动滑杆")
        fig.canvas.draw_idle()
        return
    pts = chain_points(q1, q2, x)
    seg = lambda a, b: ([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                        [pts[a][2], pts[b][2]])
    for art, (a, b) in ((ln_thigh, ("O", "K")), (ln_shank, ("L1h", "F")),
                        (ln_l2, ("L2h", "A"))):
        xs, ys, zs = seg(a, b)
        art.set_data(xs, ys)
        art.set_3d_properties(zs)
    xs = [pts["K"][0], pts["P3h"][0], pts["A"][0]]      # calf + _3 闭链为一体
    ys = [pts["K"][1], pts["P3h"][1], pts["A"][1]]
    zs = [pts["K"][2], pts["P3h"][2], pts["A"][2]]
    ln_calf.set_data(xs, ys)
    ln_calf.set_3d_properties(zs)
    jx = [pts[k][0] for k in ("O", "K", "L1h", "P3h", "L2h", "A")]
    jy = [pts[k][1] for k in ("O", "K", "L1h", "P3h", "L2h", "A")]
    jz = [pts[k][2] for k in ("O", "K", "L1h", "P3h", "L2h", "A")]
    jt.set_data(jx, jy)
    jt.set_3d_properties(jz)
    ft.set_data([pts["F"][0]], [pts["F"][1]])
    ft.set_3d_properties([pts["F"][2]])

    fz = pts["F"] * 1000.0
    r_ax = float(np.linalg.norm(fz - (fz @ U1) * U1))
    z_ax = float(fz @ U1)
    q3d, q4d, q5d = (math.degrees(v) for v in x)
    txt.set_text(
        f"q1 = {s_q1.val:+7.2f}°   q2 = {s_q2.val:+7.2f}°\n"
        f"被动角 q3,q4,q5 = {q3d:+7.2f} {q4d:+7.2f} {q5d:+7.2f} °\n"
        f"足端 (hip系, mm) = {fz[0]:+8.1f} {fz[1]:+8.1f} {fz[2]:+8.1f}\n"
        f"轴半径 r = {r_ax:7.1f} mm   轴向 z = {z_ax:+7.1f} mm\n"
        f"闭环残差 = {err:.1e}"
    )
    fig.canvas.draw_idle()


def on_q1(val):
    q1, q2 = math.radians(s_q1.val), math.radians(s_q2.val)
    x, err, f = solve_for(q1, q2)
    if f is not None:
        fz = f * 1000.0
        tr1_x.append(fz[0]); tr1_y.append(fz[1]); tr1_z.append(fz[2])
        _cap(tr1_x); _cap(tr1_y); _cap(tr1_z)
        tr1.set_data(tr1_x, tr1_y)
        tr1.set_3d_properties(tr1_z)
    update(val)


def on_q2(val):
    q1, q2 = math.radians(s_q1.val), math.radians(s_q2.val)
    x, err, f = solve_for(q1, q2)
    if f is not None:
        fz = f * 1000.0
        tr2_x.append(fz[0]); tr2_y.append(fz[1]); tr2_z.append(fz[2])
        _cap(tr2_x); _cap(tr2_y); _cap(tr2_z)
        tr2.set_data(tr2_x, tr2_y)
        tr2.set_3d_properties(tr2_z)
    update(val)


s_q1.on_changed(on_q1)
s_q2.on_changed(on_q2)


def sweep_q1(_=None):
    orig = s_q1.val
    for v in np.linspace(-180.0, 180.0, 97):
        s_q1.set_val(float(v))
        plt.pause(0.004)
    s_q1.set_val(orig)


def sweep_q2(_=None):
    orig = s_q2.val
    for v in np.concatenate([np.linspace(Q2_MAX_DEG, Q2_MIN_DEG, 111),
                             np.linspace(Q2_MIN_DEG, orig, 56)]):
        s_q2.set_val(float(v))
        plt.pause(0.004)


def draw_cloud(_=None):
    clear(_=None)
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(Q2_MIN_DEG, Q2_MAX_DEG)
    for q2d in np.arange(Q2_MAX_DEG, Q2_MIN_DEG - 0.1, -10.0):
        s_q2.set_val(float(q2d))
        warm = WARM.copy()
        xs, ys, zs = [], [], []
        for q1d in np.linspace(-180.0, 180.0, 73):
            x, err, f = solve_passive(math.radians(q1d), math.radians(q2d), warm)
            if err > 1e-8:
                break
            warm = x
            fz = f * 1000.0
            xs.append(fz[0]); ys.append(fz[1]); zs.append(fz[2])
        if len(xs) > 2:
            art, = ax.plot(xs, ys, zs, color=cmap(norm(q2d)), lw=0.9, alpha=0.45)
            cloud_artists.append(art)
        plt.pause(0.001)
    fig.canvas.draw_idle()


def clear(_=None):
    for lst, art in (((tr1_x, tr1_y, tr1_z), tr1), ((tr2_x, tr2_y, tr2_z), tr2)):
        lst[0].clear(); lst[1].clear(); lst[2].clear()
        art.set_data([], [])
        art.set_3d_properties([])
    for art in cloud_artists:
        art.remove()
    cloud_artists.clear()
    fig.canvas.draw_idle()


def _button(rect, label, cb):
    bax = fig.add_axes(rect)
    btn = Button(bax, label, color="#efeee8", hovercolor="#e2e1d8")
    btn.label.set_fontsize(8.5)
    btn.on_clicked(cb)
    return btn


_button([0.755, 0.135, 0.100, 0.045], "周转 q1 一周", sweep_q1)
_button([0.870, 0.135, 0.100, 0.045], "扫 q2 生成线", sweep_q2)
_button([0.755, 0.075, 0.100, 0.045], "工作空间点云", draw_cloud)
_button([0.870, 0.075, 0.100, 0.045], "清空轨迹", clear)

update()

if "--smoke" in sys.argv:
    errs = []
    for q1d in (-120, -60, 0, 45, 120):
        for q2d in (-180, -135, -90, -45, 0, 30):
            x, e, f = solve_for(math.radians(q1d), math.radians(q2d))
            errs.append(e)
            assert x is not None and e < 1e-8, (q1d, q2d, e)
    print(f"smoke OK: 30 poses, max closure err {max(errs):.2e}")
    sys.exit(0)

plt.show()
