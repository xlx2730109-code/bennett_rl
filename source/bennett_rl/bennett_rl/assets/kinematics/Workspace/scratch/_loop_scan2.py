"""Scratch: extended local-loop scan -- small A1, large A2 corner."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(r"e:/Project/Isaaclab/bennett_rl/scripts/analysis")))
from _loop_scan import loop

print(f"{'A1':>5} {'A2':>5} {'q2c':>5} | {'ext x,y,z':>18} | "
      f"{'round3d':>7} {'flat':>6} | ctr")
for a1 in (0.20, 0.25, 0.30):
    for a2 in (0.35, 0.40, 0.45, 0.50):
        for c in (-0.05, 0.0, 0.05):
            P, err, _ = loop(a1, a2, c, np.pi / 2)
            if err > 1e-9 or not np.isfinite(P).all():
                continue
            ctr, Q = P.mean(axis=0), P - P.mean(axis=0)
            _, sv, _ = np.linalg.svd(Q, full_matrices=False)
            ext = np.ptp(P, axis=0)
            print(f"{a1:5.2f} {a2:5.2f} {c:+5.2f} | {ext[0]:5.0f},{ext[1]:5.0f},"
                  f"{ext[2]:5.0f}  | {sv[1] / sv[0]:7.3f} {sv[2] / sv[0]:6.3f} | "
                  f"({ctr[0]:+6.1f},{ctr[1]:+6.1f},{ctr[2]:+6.1f})")
