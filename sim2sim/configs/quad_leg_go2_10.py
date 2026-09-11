"""Task contract for the quad_leg_go2-10 low-speed crawl policy (50 -> 8).

go2-10 trains a *commanded crawl* policy: the canonical 33-D velocity block
plus a 17-D crawl gait block -- global phase sin/cos (2), per-leg phase sin/cos
(8), desired contacts (4) and gait params (3) -- built in
``quad_leg_go2-10/mdp/observations.py`` from a fixed 0.55 Hz / duty-0.78 /
swing-0.065 schedule.  This config mirrors that contract on the MuJoCo side
with ``obs_mode="crawl"``; ``sim2sim.py`` reproduces the schedule, including
the two details that differ from the trot tasks:

* the phase is a pure function of episode time (it keeps running through
  stand-stills; the commanded_* terms only gate the output to the stand
  schedule), and
* the per-leg sin/cos block is interleaved per leg, matching the training
  side's ``torch.stack((sin, cos), -1).reshape``.
"""

import pathlib

import numpy as np

from common import TaskConfig

_HERE = pathlib.Path(__file__).resolve().parent
_SIM = _HERE.parent
_LOG = pathlib.Path(r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_go2")


def _newest_policy(*roots: pathlib.Path) -> pathlib.Path:
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("flat/*/exported/policy.pt"))
    if not candidates:
        raise SystemExit("No exported go2-10 policy.pt found (train go2-10 first).")
    return max(candidates, key=lambda p: p.stat().st_mtime)


# Training ACTUATED_JOINTS + init_state.joint_pos (rough_env_cfg.py).  Same
# plant and default pose as the free-gait family (BENNETT_CFG_V5).
ACTUATED_JOINTS = [
    "FL_thigh", "FL_calf", "FR_thigh", "FR_calf",
    "RL_thigh", "RL_calf", "RR_thigh", "RR_calf",
]
DEFAULT_JOINT_POS = np.array([
    +0.08, -0.16,   # FL
    -0.08, -0.16,   # FR
    +0.08, -0.16,   # RL
    -0.08, -0.16,   # RR
], dtype=np.float32)

# actions.joint_pos: scale=0.20, preserve_order; go2-10 sets no action clip
# (None bounds are a no-op; targets are still limited by the MJCF actuator
# ctrlrange, whose numbers equal free_gait4's clip values).
ACTION_SCALE = 0.20

# Low-speed crawl clock, mirroring the rough_env_cfg.py constants.
GAIT = {
    "frequency_hz": 0.55,       # LOW_GAIT_FREQUENCY_HZ
    "duty_factor": 0.78,        # LOW_GAIT_DUTY_FACTOR
    "swing_height": 0.065,      # LOW_GAIT_SWING_HEIGHT
    "command_deadband": 0.025,  # COMMAND_DEADBAND (norm of the 3-D command)
    # swing start for [FL, FR, RL, RR] (mdp.gait_scheduler) -> FL->RR->FR->RL
    "phase_offsets": (0.0, 0.5, 0.75, 0.25),
}

# commands.base_velocity training sampler (UniformVelocityCommandCfg):
# low-speed omnidirectional range, 25 % standing envs, no heading tracking.
COMMAND = {
    "ranges": {
        "lin_vel_x": (-0.18, 0.18),   # LOW_SPEED_RANGE[1]
        "lin_vel_y": (-0.10, 0.10),
        "ang_vel_z": (-0.50, 0.50),
    },
    "resampling_time_range": (5.0, 8.0),
    "standing_frac": 0.25,
    "heading": False,
}

config = TaskConfig(
    task="quad_leg_go2_10",
    model=str(_SIM / "models" / "bennett_3" / "bennett_3.xml"),
    policy=str(_newest_policy(_LOG / "quad_leg_go2-10")),
    actuated_joints=ACTUATED_JOINTS,
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale=ACTION_SCALE,
    clip_low=None,
    clip_high=None,
    gait=GAIT,
    step_dt=0.02,
    phys_dt=0.002,
    obs_mode="crawl",        # 50-D = 33 base + 2 + 8 + 4 + 3 crawl block
    num_obs=50,
    num_actions=8,
    init_base_pos=(0.0, 0.0, 0.38),   # TARGET_BASE_HEIGHT
    command=COMMAND,
)
