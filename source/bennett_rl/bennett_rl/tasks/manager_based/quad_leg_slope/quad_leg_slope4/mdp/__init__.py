"""Standalone Slope4 MDP terms."""

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401,F403

from .curriculums import validated_top_platform_level
from .gait import (
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
    base_clearance_above_feet_below_minimum,
    base_height_above_feet_l2,
    commanded_straight_lateral_yaw_vel_l2,
    commanded_yaw_error_l2,
    feet_lateral_boundary_excess_l2,
    minimum_support_contacts_l2,
    moving_touchdown_impact_l2,
    trot_contact_match,
    trot_extra_swing_contacts,
    trot_missing_stance_contacts,
    trot_stance_feet_slide,
    trot_swing_foot_height_tracking,
    trot_worst_swing_foot_height_shortfall_l2,
    uphill_velocity_progress,
)
from .terminations import top_platform_success
