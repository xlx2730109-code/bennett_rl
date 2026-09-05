"""Rich diagnostic: hold forward command, log joint angles + per-leg ankle height.

Reveals whether a stepping gait emerges and whether FL/RL mirror FR/RR. If the
left legs stay pinned (no lift cycle) while right legs step, the left leg is
broken (geometry/axis). If all four step but base stays slow/curved, it's obs/
command or gains.
"""

import sys, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sim2sim import MujocoRunner, load_config

cfg = load_config("quad_leg_free_gait3")
runner = MujocoRunner(cfg, headless=True)
policy = torch.jit.load(cfg.policy).eval()

m, d = runner.model, runner.data
# ankle-proxy body ids
ankle = {leg: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{leg}_2")
         for leg in ["FL", "FR", "RL", "RR"]} if False else None
import mujoco as mj
ankle = {leg: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{leg}_2")
         for leg in ["FL", "FR", "RL", "RR"]}
thigh_body = {leg: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{leg}_thigh")
              for leg in ["FL", "FR", "RL", "RR"]}

runner.reset()
runner.set_command(0.30, 0.0, 0.0)
obs = runner._obs()
x0 = runner.data.xpos[runner.base_id].copy()

print("step | FL.calf FR.calf RL.calf RR.calf | FL.z FR.z RL.z RR.z (ankle heights)")
for s in range(60):
    with torch.no_grad():
        raw = policy(torch.from_numpy(obs).float().reshape(1, -1)).numpy().reshape(-1)
    obs = runner.step_control(raw)
    if s % 4 == 0:
        q = {leg: d.qpos[m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{leg}_calf")]] for leg in ["FL","FR","RL","RR"]}
        h = {leg: d.xpos[ankle[leg]][2] for leg in ["FL","FR","RL","RR"]}
        print(f"{s:4d} | {q['FL']:+.3f} {q['FR']:+.3f} {q['RL']:+.3f} {q['RR']:+.3f} "
              f"| {h['FL']:+.4f} {h['FR']:+.4f} {h['RL']:+.4f} {h['RR']:+.4f}")
b = runner.data.xpos[runner.base_id]
print(f"\nbase disp = {b - x0}")
