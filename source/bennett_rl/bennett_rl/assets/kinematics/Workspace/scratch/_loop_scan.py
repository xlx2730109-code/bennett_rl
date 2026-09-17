"""Scratch: LOCAL bottom loop -- q1 swings about the bottom direction while
q2 flexes in quadrature, so the marker loops around the workspace bottom
(the region the user circled) instead of revolving the full z1 circle.

q1 = q1_bottom + A1 sin(2 pi tau)
q2 = q2c + A2 sin(2 pi tau + psi)          tau in [0, 1)  -> closed loop
q1_bottom = -pi/2 - theta0(q2c)  (bottom of the marker circle about z1)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"e:/Project/Isaaclab/bennett_rl/scripts/analysis")))
import bennett_leg_fk as fk

MIR = np.array([1.0, -1.0, 1.0])
ZD = fk.U1 * MIR
ZD /= np.linalg.norm(ZD)
WD = np.array([0.0, abs(ZD[1]), abs(ZD[2])])   # (0, s, s) display in-plane unit
S = abs(ZD[2])


def p2_marker(q1, q2):
    """TRUE marker (T1 @ L2_POS), display coords, mm."""
    qp, err, _ = fk.solve_passive(0.0, float(q2), np.zeros(3))
    q3, q4, q5 = qp
    R1 = fk.rot_axis(fk.U1, float(q1))
    T1 = R1 @ fk.tf(fk.L1_POS) @ fk.rot_axis(fk.U1B, q4)
    return (T1 @ np.append(fk.L2_POS, 1.0))[:3] * 1000.0 * MIR, err


def theta0(q2c):
    p, _ = p2_marker(0.0, q2c)
    return np.arctan2(float(p @ WD), float(p[0]))


def loop(a1, a2, c, psi, n=361):
    q1b = -np.pi / 2.0 - theta0(c)
    tau = np.linspace(0.0, 1.0, n, endpoint=False)
    q1s = q1b + a1 * np.sin(2.0 * np.pi * tau)
    q2s = c + a2 * np.sin(2.0 * np.pi * tau + psi)
    pts, worst = [], 0.0
    for q1, q2 in zip(q1s, q2s):
        p, err = p2_marker(float(q1), float(q2))
        worst = max(worst, err)
        pts.append(p)
    return np.array(pts), worst, np.degrees(q1b)


print(f"local bottom loop;  target: centre z ~ -225, size ~ 200 x 150, round")
print(f"{'A1':>5} {'A2':>5} {'q2c':>5} {'psi':>4} | {'err':>9} | "
      f"{'x':>13} {'y':>13} {'z':>13} | {'round3d':>7} {'flat':>6} | ctr")
best = None
for a1 in (0.35, 0.45, 0.55):
    for a2 in (0.20, 0.30, 0.40):
        for c in (-0.10, 0.00, 0.10):
            for psi in (np.pi / 2, -np.pi / 2):
                P, err, q1b = loop(a1, a2, c, psi)
                if err > 1e-9 or not np.isfinite(P).all():
                    continue
                ctr, Q = P.mean(axis=0), P - P.mean(axis=0)
                _, sv, _ = np.linalg.svd(Q, full_matrices=False)
                r3, fl = sv[1] / sv[0], sv[2] / sv[0]
                ext = np.ptp(P, axis=0)
                tag = ""
                if r3 > 0.80 and -250 < ctr[2] < -190:
                    tag = "  <-- cand"
                    if best is None or r3 > best[0]:
                        best = (r3, a1, a2, c, np.degrees(psi), ext, ctr, q1b)
                print(f"{a1:5.2f} {a2:5.2f} {c:+5.2f} {np.degrees(psi):4.0f} | "
                      f"{err:9.1e} | [{ext[0]:5.0f}] [{ext[1]:5.0f}] "
                      f"[{ext[2]:5.0f}] | {r3:7.3f} {fl:6.3f} | "
                      f"({ctr[0]:+6.1f},{ctr[1]:+6.1f},{ctr[2]:+6.1f}){tag}")
if best:
    r3, a1, a2, c, psi, ext, ctr, q1b = best
    print(f"\nBEST round3d={r3:.3f}: A1={a1} A2={a2} q2c={c} psi={psi:.0f} deg "
          f"ext=({ext[0]:.0f},{ext[1]:.0f},{ext[2]:.0f}) ctr={ctr.round(1)} "
          f"q1_bottom={q1b:.1f} deg")
