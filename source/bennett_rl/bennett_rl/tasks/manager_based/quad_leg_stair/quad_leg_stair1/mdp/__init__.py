"""MDP terms local to Bennett Stair1."""

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .curriculums import validated_stair_level
from .observations import foot_contact_state, normalized_terrain_level
from .rewards import (
    base_height_above_feet_l2,
    lane_deviation_l2,
    minimum_support_contacts_l2,
    moving_touchdown_impact_l2,
    stair_swing_clearance,
    track_uphill_world_velocity_exp,
    uphill_velocity_progress,
)
from .terminations import (
    base_clearance_above_feet_below_minimum,
    insufficient_stair_progress,
    outside_stair_lane,
    stair_top_success,
)
from .terrain import AscendingStairsTerrainCfg
