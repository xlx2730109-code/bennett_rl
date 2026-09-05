"""Temporary diagnostic: measure base displacement per single constant command.

Runs the free_gait3 policy in the MuJoCo sim2sim with ONE fixed command held for
N control steps, then reports the base xy displacement + yaw. This isolates
whether each command channel drives the base in the right direction/magnitude.
"""

import sys
import pathlib
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch

from sim2sim import MujocoRunner, load_config

cfg = load_config("quad_leg_free_gait3")
runner = MujocoRunner(cfg, headless=True)
policy = torch.jit.load(cfg.policy).eval()


def run(cmd, n=200):
    runner.reset()
    runner.set_command(*cmd)
    obs = runner._obs()
    x0 = runner.data.xpos[runner.base_id].copy()
    for _ in range(n):
        with torch.no_grad():
            raw = policy(torch.from_numpy(obs).float().reshape(1, -1)).numpy().reshape(-1)
        obs = runner.step_control(raw)
    b = runner.data.xpos[runner.base_id]
    w, x, y, z = runner.data.qpos[3:7]
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return b - x0, yaw


for name, cmd in [
    ("fwd", (0.30, 0.0, 0.0)),
    ("back", (-0.30, 0.0, 0.0)),
    ("left", (0.0, 0.10, 0.0)),
    ("right", (0.0, -0.10, 0.0)),
    ("turnL", (0.0, 0.0, 0.40)),
    ("turnR", (0.0, 0.0, -0.40)),
]:
    d, yaw = run(cmd)
    print(f"{name:6s} cmd={cmd}  ds_xy=({d[0]:+.3f},{d[1]:+.3f})  yaw={yaw:+6.1f}deg")
