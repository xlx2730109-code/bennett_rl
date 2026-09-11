# One-off gait diagnostic: fixed forward command, log the number of feet in
# contact each control step.  A proper crawl shows 3/3/3/3 rotation (one swing
# leg at a time); a broken replay shows 4/0 hops or 2-leg trotting.
import sys

sys.path.insert(0, r"E:\Project\Isaaclab\bennett_rl\sim2sim")

import numpy as np
import torch
import mujoco as mj

from sim2sim import MujocoRunner, load_config

cfg = load_config("quad_leg_go2_10")
runner = MujocoRunner(cfg, headless=True)

policy = torch.jit.load(cfg.policy)
policy.eval()

# floor geom id (world body) for contact filtering
floor_geoms = set(range(runner.model.ngeom))
# foot-side bodies: the *_1 distal segment carries the foot contact mesh
foot_bodies = ["FL_1", "FR_1", "RL_1", "RR_1"]
foot_geom_ids = {}
for b in foot_bodies:
    bid = mj.mj_name2id(runner.model, mj.mjtObj.mjOBJ_BODY, b)
    ids = [g for g in range(runner.model.ngeom) if runner.model.geom_bodyid[g] == bid]
    foot_geom_ids[b] = set(ids)

runner.set_command(0.10, 0.0, 0.0)
obs = runner._obs()
n = int(12.0 / cfg.step_dt)  # ~6.6 crawl cycles
contacts_per_step = []
base_z = []
for k in range(n):
    with torch.no_grad():
        raw = policy(torch.from_numpy(obs).float().reshape(1, -1)).cpu().numpy().reshape(-1)
    obs = runner.step_control(raw)
    touching = []
    for b, gids in foot_geom_ids.items():
        hit = any((c.geom1 in gids or c.geom2 in gids)
                  and (runner.model.geom_bodyid[c.geom1] == 0
                       or runner.model.geom_bodyid[c.geom2] == 0)
                  for c in runner.data.contact[:runner.data.ncon])
        touching.append(1 if hit else 0)
    contacts_per_step.append(touching)
    base_z.append(runner.data.xpos[runner.base_id][2])

contacts = np.array(contacts_per_step)
print(f"[GAIT] contact count per step: mean={contacts.sum(1).mean():.2f} "
      f"min={contacts.sum(1).min()} max={contacts.sum(1).max()}")
# count steps where exactly one leg is off the ground (crawl signature)
one_swing = np.sum(contacts.sum(1) == 3) / len(contacts)
all_stance = np.sum(contacts.sum(1) == 4) / len(contacts)
print(f"[GAIT] fraction of steps: 3-stance={one_swing:.2f} 4-stance={all_stance:.2f} "
      f"other={1.0 - one_swing - all_stance:.2f}")
# per-leg air fraction over the run (crawl duty 0.78 -> ~0.22 air each)
print(f"[GAIT] per-leg air fraction: "
      f"{dict(zip(foot_bodies, np.round(1.0 - contacts.mean(0), 3)))}")
b = np.array(base_z)
print(f"[GAIT] base z: mean={b.mean():.3f} min={b.min():.3f} max={b.max():.3f}")
xy = runner.data.xpos[runner.base_id]
print(f"[GAIT] final base xy=({xy[0]:.3f},{xy[1]:.3f}) after {n*cfg.step_dt:.0f}s at vx_cmd=0.10")
