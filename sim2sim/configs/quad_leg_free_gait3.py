"""Task contract for the trained quad_leg_free_gait3 emergent-gait policy (33 -> 8).

This is the gait-free sibling of `quad_leg_trot1.py`. A free-gait policy observes
NO clock / phase / desired-contact / gait-params terms: its observation is exactly
the base + command + joint + action block (33 dims), which we build with
``obs_mode="plain"``. The gait is *emergent* (driven in training by
``feet_air_time``), so there is no ``gait`` dict here at all.

Pointed at the newest free_gait3 export (the trained, walking-well policy).
"""

import pathlib

import numpy as np

from common import TaskConfig

_HERE = pathlib.Path(__file__).resolve().parent
_SIM = _HERE.parent
_LOG = pathlib.Path(r"E:\Project\Isaaclab\bennett_rl\logs\rsl_rl\quad_leg_free_gait")


def _newest_policy(root: pathlib.Path) -> pathlib.Path:
    # Path.glob does not expand wildcards in the *base* path, so the pattern must
    # include the "flat/*/exported/policy.pt" segment.
    candidates = root.glob("flat/*/exported/policy.pt")
    if not candidates:
        raise SystemExit("No trained quad_leg_free_gait3 policy found (run sim2sim after training).")
    return max(candidates, key=lambda p: p.stat().st_mtime)


# Training ACTUATED_JOINTS + init_state.joint_pos (default/stand pose). Same as
# trot1 -- free_gait3 inherits the same plant and default pose.
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

# actions.joint_pos: scale=0.20, clip=JOINT_TARGET_LIMITS (same as trot1)
ACTION_SCALE = 0.20
CLIP_LOW = np.array([-0.80, -0.90, -0.80, -0.90, -0.80, -0.90, -0.80, -0.90], dtype=np.float32)
CLIP_HIGH = np.array([0.80, 0.55, 0.80, 0.55, 0.80, 0.55, 0.80, 0.55], dtype=np.float32)

config = TaskConfig(
    task="quad_leg_free_gait3",
    model=str(_SIM / "models" / "bennett_3" / "bennett_3.xml"),
    policy=str(_newest_policy(_LOG / "quad_leg_free_gait3")),
    actuated_joints=ACTUATED_JOINTS,
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale=ACTION_SCALE,
    clip_low=CLIP_LOW,
    clip_high=CLIP_HIGH,
    gait={},                 # emergent gait -> no clock / no phase / no duty-blend
    step_dt=0.02,
    phys_dt=0.002,
    obs_mode="plain",        # 33-D base+command+joint+action, no gait block
    num_obs=33,
    num_actions=8,
    init_base_pos=(0.0, 0.0, 0.36),
)
