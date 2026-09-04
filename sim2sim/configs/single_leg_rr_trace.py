"""Task contract for the single_leg_rr_trace fixed-base reference trace (12 -> 2).

RR thigh/calf track a deterministic two-dimensional sinusoidal reference around
the default pose (all-zero for V3).  The policy outputs two raw actions which are
clamped/scaled/rate-limited, then applied to RR_thigh and RR_calf; the other six
leg joints are held at their default pose.

    obs = phase(2) | ref_offsets(2) | tracking_error(2) | joint_pos_rel(2)
          | joint_vel_rel(2) | prev_raw_action(2)   => 12
"""

import math
import pathlib

import numpy as np

from common import TaskConfig

_HERE = pathlib.Path(__file__).resolve().parent
_SIM = _HERE.parent

# All 8 driveable leg joints (training order); only RR is tracked, rest are held.
ACTUATED_JOINTS = [
    "FL_thigh", "FL_calf", "FR_thigh", "FR_calf",
    "RL_thigh", "RL_calf", "RR_thigh", "RR_calf",
]
# source bennett.robot uses BENNETT_CFG_V3 whose joint_pos is empty -> default 0
DEFAULT_JOINT_POS = np.zeros(len(ACTUATED_JOINTS), dtype=np.float32)

# SingleLegPositionAction: scale = 20 deg, max_joint_speed = 60 deg/s
SCALE = math.radians(20.0)
MAX_JOINT_SPEED = math.radians(60.0)

# Reference clock from mdp: freq_hz = max_speed/(2*pi*amplitude); phase = omega*t
TRACE = {
    "amplitude_rad": math.radians(20.0),
    "max_speed_rad_s": math.radians(29.0),
    "calf_phase_rad": math.pi / 2.0,
    "scale": SCALE,
    "max_joint_speed": MAX_JOINT_SPEED,
    # (joint, kind 0=thigh/1=calf, leg_sign) in obs order; RR sign = +1
    "table": [("RR_thigh", 0, +1.0), ("RR_calf", 1, +1.0)],
}

config = TaskConfig(
    task="single_leg_rr_trace",
    model=str(_SIM / "models" / "bennett_3" / "bennett_3_fixed.xml"),
    policy=str(_SIM / "policies" / "single_leg_rr_trace" / "policy.pt"),
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
    num_obs=12,
    num_actions=2,
    ep_len_s=8.0,
    init_base_pos=(0.0, 0.0, 0.32),
)
