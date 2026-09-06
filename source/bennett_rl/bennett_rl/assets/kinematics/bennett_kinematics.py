# Copyright (c) 2026, Bennett. All rights reserved.

"""Bennett leg kinematics: URDF parsing, FK, and closed-chain (Bennett) solve.

Geometry source of truth
------------------------
``assets/robots/Urdf_Bennett_3/urdf/Urdf_Bennett_3.urdf`` is parsed at import
time -- joint origins/axes are read from it, never transcribed.  The URDF is a
kinematic TREE (SolidWorks export): base -> thigh -> {calf -> _3, _1 -> {_2,
foot}}.  The real hardware closes the loop at the ankle: the tip of ``_3``
meets the tip of ``_2`` on a pin (the parallel four-bar / Bennett coupling
that keeps the foot branch oriented).  We reproduce that closure as a
point-on-point constraint between two body-fixed ankle anchors:

* anchor in ``_3`` local frame: taken verbatim from the validated sim2sim
  model ``sim2sim/models/bennett_1/bennett_1.xml`` ``<connect>`` lines, which
  ``_genfinal.py`` fitted as the closest point pair of the _3/_2 meshes at the
  zero pose;
* anchor in ``_2`` local frame: derived -- the same world point at the zero
  pose, expressed into ``_2`` (the meshes coincide there).

For driven angles (q_thigh, q_calf) the passive triple (q_1, q_2, q_3) is
solved by damped Gauss-Newton on r(q_p) = A3 - A2 = 0 (3 eqs / 3 unknowns),
warm-started across the sweep so the solution branch stays continuous.

Sign conventions are the URDF's: FL/RL thigh axes are +y+z (0, .707, .707),
FR/RR are mirrored; positive q rotates right-hand about the joint axis.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENNETT_URDF = (
    HERE.parent / "robots" / "Urdf_Bennett_3" / "urdf" / "Urdf_Bennett_3.urdf"
)

# ankle anchors in the _3 body frame (FL & RL legs; FR & RR differ slightly
# because the exported meshes are not perfectly mirrored), copied from
# sim2sim/models/bennett_1/bennett_1.xml <connect> entries.
ANKLE_ON_3 = {
    "FL": np.array([-0.0289418, 0.00268795, -0.0876585]),
    "FR": np.array([-0.0334567, 0.0156048, -0.0764966]),
    "RL": np.array([-0.0289418, 0.00268795, -0.0876585]),
    "RR": np.array([-0.0334567, 0.0156048, -0.0764966]),
}

# commanded joint targets used by the tasks (JOINT_TARGET_LIMITS)
LEG_JOINT_LIMITS = {
    "thigh": (-0.80, 0.80),
    "calf": (-0.90, 0.55),
}

# BENNETT_CFG_V5 initial (default) joint pose: FL/RL thigh +0.12, FR/RR -0.12,
# calf -0.24 all legs (assets/robots/bennett.py).
V5_DEFAULT_POSE = {
    "FL": (0.12, -0.24),
    "FR": (-0.12, -0.24),
    "RL": (0.12, -0.24),
    "RR": (-0.12, -0.24),
}


def rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation matrix about a (unit) axis by theta (Rodrigues)."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray  # translation in parent frame (rpy is zero in this URDF)
    axis: np.ndarray    # rotation axis in the joint (child-at-zero) frame
    joint_type: str     # revolute | fixed


@dataclass
class BennettLeg:
    """Kinematics of one leg (side ∈ {FL, FR, RL, RR}) parsed from the URDF."""

    side: str
    joints: dict = field(default_factory=dict)  # name -> Joint
    ankle_on_3: np.ndarray = None
    ankle_on_2: np.ndarray = None

    # ---------- frames ----------
    def _joint(self, suffix: str) -> Joint:
        return self.joints[f"{self.side}_{suffix}"]

    @staticmethod
    def _tf(joint: Joint, q: float):
        R = rot_axis(joint.axis, q)
        return R, joint.origin

    def frames(self, q_thigh: float, q_calf: float, q1: float, q2: float, q3: float):
        """World(=base)-frame poses R,t of thigh/calf/_1/_2/_3 and the foot point."""
        Rt, tt = self._tf(self._joint("thigh"), q_thigh)          # base -> thigh
        Rc, tc = self._tf(self._joint("calf"), q_calf)            # thigh -> calf
        R1, t1 = self._tf(self._joint("1"), q1)                   # thigh -> _1
        R2, t2 = self._tf(self._joint("2"), q2)                   # _1   -> _2
        R3, t3 = self._tf(self._joint("3"), q3)                   # calf -> _3
        Rfc, tfc = Rt @ Rc, tt + Rt @ tc
        Rt1, tt1 = Rt @ R1, tt + Rt @ t1
        Rt2, tt2 = Rt1 @ R2, tt1 + Rt1 @ t2
        Rt3, tt3 = Rfc @ R3, tfc + Rfc @ t3
        foot = tt1 + Rt1 @ self._joint("foot").origin             # fixed on _1
        return {
            "thigh": (Rt, tt), "calf": (Rfc, tfc), "_1": (Rt1, tt1),
            "_2": (Rt2, tt2), "_3": (Rt3, tt3), "foot": foot,
        }

    # ---------- closure ----------
    def _residual(self, q_thigh, q_calf, q_passive) -> np.ndarray:
        f = self.frames(q_thigh, q_calf, *q_passive)
        A3 = f["_3"][0] @ self.ankle_on_3 + f["_3"][1]
        A2 = f["_2"][0] @ self.ankle_on_2 + f["_2"][1]
        return A3 - A2

    def solve_passive(self, q_thigh, q_calf, warm=None, iters=40, tol=1e-9):
        """Gauss-Newton solve of the ankle closure; returns (q1,q2,q3, ||r||)."""
        x = np.zeros(3) if warm is None else np.array(warm, dtype=float)
        lam = 1e-6
        for _ in range(iters):
            r = self._residual(q_thigh, q_calf, x)
            n = np.linalg.norm(r)
            if n < tol:
                break
            J = np.zeros((3, 3))
            e = 1e-7
            for k in range(3):
                xp = x.copy()
                xp[k] += e
                J[:, k] = (self._residual(q_thigh, q_calf, xp) - r) / e
            dx, *_ = np.linalg.lstsq(J.T @ J + lam * np.eye(3), -J.T @ r, rcond=None)
            step = np.clip(dx, -0.3, 0.3)
            x = x + step
        r = self._residual(q_thigh, q_calf, x)
        return x, float(np.linalg.norm(r))

    # ---------- outputs ----------
    def foot_pos(self, q_thigh, q_calf, q_passive=None) -> np.ndarray:
        if q_passive is None:
            q_passive, _ = self.solve_passive(q_thigh, q_calf)
        return self.frames(q_thigh, q_calf, *q_passive)["foot"]

    def hip_pos(self) -> np.ndarray:
        return self._joint("thigh").origin


def load_leg_model(urdf_path: Path = BENNETT_URDF) -> dict:
    """Parse the URDF and build one BennettLeg per side."""
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, Joint] = {}
    for j in root.iter("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        a = j.find("axis")
        axis = np.array([float(v) for v in (a.get("xyz") if a is not None else "0 0 1").split()])
        joints[j.get("name")] = Joint(
            name=j.get("name"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            origin=xyz,
            axis=axis,
            joint_type=j.get("type"),
        )

    legs = {}
    for side in ("FL", "FR", "RL", "RR"):
        leg = BennettLeg(side=side, joints=joints, ankle_on_3=ANKLE_ON_3[side])
        # derive the _2-side anchor: same world point at the zero pose
        f0 = leg.frames(0.0, 0.0, 0.0, 0.0, 0.0)
        A0 = f0["_3"][0] @ leg.ankle_on_3 + f0["_3"][1]
        leg.ankle_on_2 = f0["_2"][0].T @ (A0 - f0["_2"][1])
        legs[side] = leg
    return legs


# ---------------------------------------------------------------------------
# grid sweep (shared by the chart scripts)
# ---------------------------------------------------------------------------
def sweep_workspace(
    leg: BennettLeg,
    q_thigh: np.ndarray,
    q_calf: np.ndarray,
) -> dict:
    """Solve the closed chain over a (q_thigh x q_calf) grid.

    Rows = q_thigh, cols = q_calf.  Gauss-Newton is warm-started column-wise
    (sweep calf up from the row centre, then down) so the passive branch stays
    continuous.  Returns arrays shaped (n_thigh, n_calf): foot xyz, passive
    q1/q2/q3 and the closure residual norm (metres).
    """
    nt, nc = len(q_thigh), len(q_calf)
    shape = (nt, nc)
    out = {k: np.full(shape, np.nan) for k in
           ("foot_x", "foot_y", "foot_z", "q1", "q2", "q3", "residual")}
    j_mid = nc // 2
    for i in range(nt):
        for direction in (+1, -1):
            warm = np.zeros(3)
            cols = range(j_mid, nc) if direction > 0 else range(j_mid - 1, -1, -1)
            for j in cols:
                qp, res = leg.solve_passive(q_thigh[i], q_calf[j], warm=warm)
                warm = qp
                foot = leg.frames(q_thigh[i], q_calf[j], *qp)["foot"]
                out["foot_x"][i, j] = foot[0]
                out["foot_y"][i, j] = foot[1]
                out["foot_z"][i, j] = foot[2]
                out["q1"][i, j], out["q2"][i, j], out["q3"][i, j] = qp
                out["residual"][i, j] = res
    return out


# ---------------------------------------------------------------------------
# self test / key numbers
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    legs = load_leg_model()
    print(f"[URDF] {BENNETT_URDF.name}: {len(legs['FL'].joints)} joints parsed")
    for side, leg in legs.items():
        # zero pose: closure residual is the mesh fit error from _genfinal
        _, res0 = leg.solve_passive(0.0, 0.0, iters=1)
        # V5 default pose
        q_t, q_c = V5_DEFAULT_POSE[side]
        qp, res = leg.solve_passive(q_t, q_c, warm=np.zeros(3))
        foot = leg.foot_pos(q_t, q_c, qp)
        hip = leg.hip_pos()
        L = hip[2] - foot[2]
        print(
            f"[{side}] zero-residual={res0:.4f} m | default q_p={np.round(qp, 3)} "
            f"|r|={res:.2e} m | foot=({foot[0]:.3f},{foot[1]:.3f},{foot[2]:.3f}) "
            f"| leg length={L:.3f} m"
        )
    # parallelogram behaviour: does q1 track q_calf (Bennett coupling)?
    leg = legs["FL"]
    warm = np.zeros(3)
    print("[coupling] q_calf -> q_1 (q_thigh=0):")
    for q_c in (0.0, -0.2, -0.4, -0.6, -0.8):
        qp, res = leg.solve_passive(0.0, q_c, warm=warm)
        warm = qp
        print(f"    q_calf={q_c:+.2f} -> q1={qp[0]:+.3f} q2={qp[1]:+.3f} q3={qp[2]:+.3f} |r|={res:.1e}")
