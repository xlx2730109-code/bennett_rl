"""MDP terms for Bennett Trot1."""

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .command_sampling import COMMAND_MODE_NAMES, sample_balanced_velocity_commands
from .commands import BalancedTrotVelocityCommand, BalancedTrotVelocityCommandCfg
from .gait import (
    LEG_NAMES,
    TROT_PHASE_OFFSETS,
    TrotSchedule,
    commanded_trot_desired_contacts,
    commanded_trot_gait_params,
    commanded_trot_global_phase_sin_cos,
    commanded_trot_leg_phase_sin_cos,
    compute_trot_schedule,
    get_commanded_trot_schedule,
    smooth_swing_profile,
    soft_swing_weights,
    speed_conditioned_gait_parameters,
)
from .rewards import (
    action_second_difference_l2,
    base_ang_vel_xy_l2,
    commanded_straight_lateral_yaw_vel_l2,
    commanded_yaw_error_l2,
    moving_touchdown_impact_l2,
    stand_still_base_vel_l2,
    stand_still_joint_pose_exp,
    stand_still_joint_vel_l2,
    trot_contact_match,
    trot_extra_swing_contacts,
    trot_missing_stance_contacts,
    trot_stance_feet_slide,
    trot_swing_foot_height_tracking,
)
