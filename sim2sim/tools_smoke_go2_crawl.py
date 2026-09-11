# One-off parity check: sim2sim crawl schedule vs the training-side
# quad_leg_go2-10 gait_scheduler, step by step over a move/stop command trace.
import sys

sys.path.insert(0, r"E:\Project\Isaaclab\bennett_rl\sim2sim")
sys.path.insert(0, r"E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\tasks\manager_based\quad_leg_go2\quad_leg_go2-10\mdp")

import numpy as np
import torch

from sim2sim import MujocoRunner, load_config
from gait_scheduler import compute_crawl_schedule, compute_stand_schedule

FREQ, DUTY, DEADBAND = 0.55, 0.78, 0.025
SWING_FRAC = 1.0 - DUTY

cfg = load_config("quad_leg_go2_10")
runner = MujocoRunner(cfg, headless=True)

mismatch = 0
for k in range(600):
    t = k * cfg.step_dt
    # trace: move 1.3 s, stop 0.7 s, repeat -- exercises both gates
    moving = (t % 2.0) < 1.3
    cmd = (0.10, 0.03, 0.2) if moving else (0.0, 0.0, 0.0)
    runner.step = k
    runner.set_command(*cmd)
    gp, lp, dc, freq, duty, height = runner._advance_gait()

    # training side: commanded_* semantics
    phase_t = torch.remainder(torch.tensor(t * FREQ), 1.0)
    moving_t = torch.tensor(np.linalg.norm(cmd) >= DEADBAND)
    cmd_phase = torch.where(moving_t, phase_t, torch.zeros_like(phase_t))
    crawl = compute_crawl_schedule(cmd_phase.unsqueeze(0), duty_factor=DUTY)
    stand = compute_stand_schedule(cmd_phase.unsqueeze(0))
    lp_train = torch.where(moving_t, crawl.leg_phase, stand.leg_phase)[0].numpy()
    dc_train = torch.where(
        moving_t, crawl.desired_contact, stand.desired_contact)[0].numpy().astype(np.float32)
    gp_train = float(cmd_phase.item())
    f_train, d_train, h_train = (FREQ, DUTY, 0.065) if moving else (0.0, 1.0, 0.0)

    # dc mismatch only counts off the swing-fraction boundary: exactly at
    # leg_phase == swing_frac one ulp flips the flag on either side.
    on_boundary = np.any(np.abs(
        np.mod(lp_train - SWING_FRAC, 1.0)) < 1e-4) or np.any(
        np.abs(np.mod(lp_train - SWING_FRAC, 1.0) - 1.0) < 1e-4)
    dc_ok = (np.allclose(dc, dc_train) or on_boundary)
    ok = (np.isclose(gp, gp_train, atol=1e-6)
          and np.allclose(lp, lp_train, atol=1e-6)
          and dc_ok
          and np.isclose(freq, f_train) and np.isclose(duty, d_train) and np.isclose(height, h_train))
    if not ok:
        mismatch += 1
        if mismatch <= 3:
            print(f"MISMATCH k={k} sim gp={gp:.4f} lp={lp} dc={dc} "
                  f"| train gp={gp_train:.4f} lp={lp_train} dc={dc_train}")
print(f"[PARITY] schedule mismatches: {mismatch}/600")

# interleaving check: obs block layout vs torch.stack((sin,cos),-1).reshape
runner.step = 300
runner.set_command(0.10, 0.0, 0.0)
obs = runner._obs()
assert obs.shape[0] == 50, f"obs dim {obs.shape[0]} != 50"
lp_block = obs[35:43]  # 33 base + 2 global phase -> per-leg block at [35:43]
phase_t = torch.remainder(torch.tensor(300 * cfg.step_dt * FREQ), 1.0)
crawl = compute_crawl_schedule(phase_t.unsqueeze(0), duty_factor=DUTY)
ref = torch.stack((torch.sin(2 * np.pi * crawl.leg_phase),
                   torch.cos(2 * np.pi * crawl.leg_phase)), dim=-1).reshape(-1).numpy()
print(f"[PARITY] leg-phase block interleaved: {np.allclose(lp_block, ref, atol=1e-5)}")
print(f"[PARITY] global phase block {obs[33:35]} vs "
      f"[{np.sin(2*np.pi*phase_t.item()):.4f}, {np.cos(2*np.pi*phase_t.item()):.4f}]")
print(f"[PARITY] desired contacts {obs[43:47]} gait params {obs[47:50]}")
print(f"[OK] obs dim = {obs.shape[0]}")
