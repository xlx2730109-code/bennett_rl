"""Task contract for the quad_leg_track fixed-base reference trace (36 -> 2).

All four legs track the RR-style base sine reference.  The policy outputs two
raw actions (base thigh, base calf) which are clamped/scaled/rate-limited and
then expanded to each of the 8 joints via LEG_SIGNS (FL=+1, FR=-1, RL=-1,
RR=+1); every joint is a controlled (trace) joint here.

    obs = phase(2) | ref_offsets(8) | tracking_error(8) | joint_pos_rel(8)
          | joint_vel_rel(8) | prev_raw_action(2)   => 36
"""

import math
import pathlib

import numpy as np

from common import TaskConfig

_HERE = pathlib.Path(__file__).resolve().parent
_SIM = _HERE.parent

ACTUATED_JOINTS = [
    "FL_thigh", "FL_calf", "FR_thigh", "FR_calf",
    "RL_thigh", "RL_calf", "RR_thigh", "RR_calf",
]
# source bennett.robot uses BENNETT_CFG_V3 (empty joint_pos) -> default 0
DEFAULT_JOINT_POS = np.zeros(len(ACTUATED_JOINTS), dtype=np.float32)

# QuadLegPositionAction: scale = 20 deg, max_joint_speed = 60 deg/s
SCALE = math.radians(20.0)
MAX_JOINT_SPEED = math.radians(60.0)

# Reference clock from mdp, identical to single-leg; per-joint sign = LEG_SIGNS
TRACE = {
    "amplitude_rad": math.radians(20.0),
    "max_speed_rad_s": math.radians(29.0),
    "calf_phase_rad": math.pi / 2.0,
    "scale": SCALE,
    "max_joint_speed": MAX_JOINT_SPEED,
    # (joint, kind 0=thigh/1=calf, leg_sign) in obs order; LEG_SIGNS
    "table": [
        ("FL_thigh", 0, +1.0), ("FL_calf", 1, +1.0),
        ("FR_thigh", 0, -1.0), ("FR_calf", 1, -1.0),
        ("RL_thigh", 0, -1.0), ("RL_calf", 1, -1.0),
        ("RR_thigh", 0, +1.0), ("RR_calf", 1, +1.0),
    ],
}

config = TaskConfig(
    task="quad_leg_track",
    model=str(_SIM / "models" / "bennett_3" / "bennett_3_fixed.xml"),
    policy=str(_SIM / "policies" / "quad_leg_track" / "policy.pt"),
    actuated_joints=ACTUATED_JOINTS,
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale=SCALE,
    clip_low=None,
    clip_high=None,
    gait={},
    trace=TRACE,
    step_dt=0.02,
    phys_dt=0.002,
    obs_mode="trace",
    num_obs=36,
    num_actions=2,
    ep_len_s=8.0,
    init_base_pos=(0.0, 0.0, 0.32),
)
