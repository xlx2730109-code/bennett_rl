"""Task contract for the quad_leg_trot1 diagonal-trot policy (50 -> 8).

Pointed at the faithful Urdf_Bennett_3 model (bennett_1.xml), NOT the old
bennett_3.xml. The trot1 policy observes the full 50-D contract: base + command +
joint + action PLUS the commanded-trot gait block (phase/duty/...), so
``obs_mode="trot"`` must stay and the ``gait`` dict below must be populated.
The action side is the same 8-joint, scale=0.20, clip=JOINT_TARGET_LIMITS plant.
"""

import pathlib

import numpy as np

from common import TaskConfig

_HERE = pathlib.Path(__file__).resolve().parent
_SIM = _HERE.parent

# Training ACTUATED_JOINTS + init_state.joint_pos (default/stand pose).
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

# actions.joint_pos: scale=0.20, clip=JOINT_TARGET_LIMITS
ACTION_SCALE = 0.20
CLIP_LOW = np.array([-0.80, -0.90, -0.80, -0.90, -0.80, -0.90, -0.80, -0.90], dtype=np.float32)
CLIP_HIGH = np.array([0.80, 0.55, 0.80, 0.55, 0.80, 0.55, 0.80, 0.55], dtype=np.float32)

# GAIT_PARAMS from flat_env_cfg.py + trot offsets
GAIT = {
    "command_deadband": 0.025,
    "min_frequency_hz": 0.75,
    "max_frequency_hz": 1.35,
    "min_equivalent_speed": 0.08,
    "max_equivalent_speed": 0.35,
    "low_speed_duty_factor": 0.62,
    "high_speed_duty_factor": 0.54,
    "swing_height": 0.045,
    "yaw_equivalent_radius": 0.20,
    "phase_offsets": (0.0, 0.5, 0.5, 0.0),  # (FL, FR, RL, RR)
}

config = TaskConfig(
    task="quad_leg_trot1",
    model=str(_SIM / "models" / "bennett_1" / "bennett_1.xml"),
    policy=str(_SIM / "policies" / "quad_leg_trot1" / "policy.pt"),
    actuated_joints=ACTUATED_JOINTS,
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale=ACTION_SCALE,
    clip_low=CLIP_LOW,
    clip_high=CLIP_HIGH,
    gait=GAIT,
    step_dt=0.02,
    phys_dt=0.002,
    obs_mode="trot",
    num_obs=50,
    num_actions=8,
    init_base_pos=(0.0, 0.0, 0.36),
)
