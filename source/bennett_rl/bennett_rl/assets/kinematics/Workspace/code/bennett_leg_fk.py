"""Closed-chain forward kinematics for one Bennett leg of Urdf_Bennett_3.

The leg is a single spatial loop (Bennett-type 6R + spherical closure):
    base -> thigh(q1) -> calf(q2) -> _3(q3) ---(closure anchor)--- _2(q5) <- _1(q4) <- thigh
The two actuated axes are coaxial through the hip point. Given (q1, q2) the three
passive angles (q3, q4, q5) are solved so that the closure anchors coincide; the
tip (FL_foot, fixed to _1) then traces the leg workspace.

All geometry is taken verbatim from sim2sim/models/bennett_1/bennett_1.xml
(identical to the Urdf_Bennett_3 URDF).
"""

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------- geometry ---
HIP = np.array([0.211101, 0.0681715, 0.0115336])      # FL_thigh body pos (base frame)
U1 = np.array([0.0, 0.707107, 0.707107])              # thigh axis (also calf axis, coaxial)
CALF_POS = np.array([0.0, 0.0357089, 0.0357089])      # calf body pos (thigh frame)
P3_POS = np.array([0.0601104, 0.123095, -0.117665])   # _3 body pos (calf frame)
U3 = np.array([-0.836516, 0.512047, 0.19506])         # _3 axis
L1_POS = np.array([-0.0601104, 0.158804, -0.0819561])  # _1 body pos (thigh frame)
U1B = np.array([0.836516, 0.512047, 0.19506])         # _1 axis
L2_POS = np.array([0.0601104, -0.00113083, -0.17046])  # _2 body pos (_1 frame)
U2 = np.array([0.0, -0.99835, 0.0574284])             # _2 axis
FOOT_POS = np.array([0.0498505, -0.00336961, -0.209379])  # foot body pos (_1 frame)
A3_LOCAL = np.array([-0.0289418, 0.00268795, -0.0876585])  # connect anchor (in _3 frame)

for _u in (U1, U3, U1B, U2):  # unit axes
    _u /= np.linalg.norm(_u)


def rot_axis(u, q):
    """Rodrigues rotation about unit axis u by angle q (4x4)."""
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    x, y, z = u
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


def tf(pos):
    T = np.eye(4)
    T[:3, 3] = pos
    return T


def anchors(q1, q2, q3, q4, q5):
    """Closure anchor positions + tip, all in the HIP frame (origin at the
    thigh joint centre, robot body axes)."""
    Tt = rot_axis(U1, q1)
    Tc = Tt @ tf(CALF_POS) @ rot_axis(U1, q2)
    T3 = Tc @ tf(P3_POS) @ rot_axis(U3, q3)
    T1 = Tt @ tf(L1_POS) @ rot_axis(U1B, q4)
    T2 = T1 @ tf(L2_POS) @ rot_axis(U2, q5)
    a3 = (T3 @ np.append(A3_LOCAL, 1.0))[:3]
    a2 = (T2 @ np.append(A2_LOCAL, 1.0))[:3]
    foot = (T1 @ np.append(FOOT_POS, 1.0))[:3]
    return a3, a2, foot


# _2-side anchor in _2 local frame: from the q=0 assembly where both coincide
_T3_0 = tf(HIP) @ tf(CALF_POS) @ tf(P3_POS)
_T2_0 = tf(HIP) @ tf(L1_POS) @ tf(L2_POS)
A2_LOCAL = (np.linalg.inv(_T2_0) @ _T3_0 @ np.append(A3_LOCAL, 1.0))[:3]


def solve_passive(q1, q2, x0=np.zeros(3)):
    """Solve (q3, q4, q5) closing the chain for actuated angles (q1, q2)."""
    def err(x):
        a3, a2, _ = anchors(q1, q2, x[0], x[1], x[2])
        return a3 - a2

    sol = least_squares(err, x0, xtol=1e-13, ftol=1e-13, gtol=1e-13)
    a3, a2, foot = anchors(q1, q2, *sol.x)
    return sol.x, np.linalg.norm(a3 - a2), foot


if __name__ == "__main__":
    # sanity: default standing pose FL thigh +0.08 / calf -0.16
    x, e, foot = solve_passive(0.08, -0.16, np.array([0.0, 0.0, 0.0]))
    print(f"passive (q3,q4,q5) = {np.round(x, 4)}")
    print(f"closure error      = {e:.3e} m")
    print(f"tip in hip frame   = {np.round(foot, 4)} m")
    print(f"tip in base frame  = {np.round(foot + HIP, 4)} m")
    # mujoco cross-check (base frame): tip_base should equal foot + HIP
    print("expected mujoco foot-thigh =", np.round(foot, 4), "(hip frame)")
