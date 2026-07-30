"""FreeGait2 locomotion on the existing Bennett directional-slope terrain."""

from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from ...quad_leg_trot.quad_leg_trot1.flat_env_cfg import (
    FOOT_BODIES,
    TARGET_BASE_HEIGHT,
)

from . import mdp
from .flat_env_cfg import QuadLegSlope1FlatEnvCfg
from .slope_terrain import DirectionalSlopeTerrainCfg


SLOPE_ANGLES = tuple(
    math.radians(angle) for angle in (0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.8, 5.4, 6.0)
)
SLOPE_TERRAIN_SIZE = (5.0, 2.8)
SLOPE_APPROACH_LENGTH = 1.20
SLOPE_TOP_PLATFORM_LENGTH = 1.00
SLOPE_SPAWN_X = 0.65
SLOPE_TRANSITION_LENGTH = 0.25
MIN_BASE_CLEARANCE = 0.20


def _directional_slope_curriculum_generator() -> TerrainGeneratorCfg:
    """Build a row-wise 0-to-6-degree slope curriculum."""

    return TerrainGeneratorCfg(
        seed=42,
        size=SLOPE_TERRAIN_SIZE,
        border_width=3.0,
        num_rows=len(SLOPE_ANGLES),
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=True,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        sub_terrains={
            "directional_slope": DirectionalSlopeTerrainCfg(
                proportion=1.0,
                size=SLOPE_TERRAIN_SIZE,
                slope_angle_range=(SLOPE_ANGLES[0], SLOPE_ANGLES[-1]),
                approach_length=SLOPE_APPROACH_LENGTH,
                top_platform_length=SLOPE_TOP_PLATFORM_LENGTH,
                spawn_x=SLOPE_SPAWN_X,
                lane_width=2.4,
                transition_length=SLOPE_TRANSITION_LENGTH,
                transition_segments=8,
            )
        },
    )


def _directional_slope_play_generator() -> TerrainGeneratorCfg:
    """Build one deterministic evaluation lane for every fixed slope angle."""

    return TerrainGeneratorCfg(
        seed=42,
        size=SLOPE_TERRAIN_SIZE,
        border_width=3.0,
        num_rows=1,
        num_cols=len(SLOPE_ANGLES),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=True,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        sub_terrains={
            f"directional_slope_{index:02d}": DirectionalSlopeTerrainCfg(
                proportion=1.0 / len(SLOPE_ANGLES),
                size=SLOPE_TERRAIN_SIZE,
                slope_angle_range=(slope_angle, slope_angle),
                approach_length=SLOPE_APPROACH_LENGTH,
                top_platform_length=SLOPE_TOP_PLATFORM_LENGTH,
                spawn_x=SLOPE_SPAWN_X,
                lane_width=2.4,
                transition_length=SLOPE_TRANSITION_LENGTH,
                transition_segments=8,
            )
            for index, slope_angle in enumerate(SLOPE_ANGLES)
        },
    )


@configclass
class QuadLegSlope1EnvCfg(QuadLegSlope1FlatEnvCfg):
    """Use the common Slope1 objective and replace only the flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _directional_slope_curriculum_generator()
        self.scene.terrain.use_terrain_origins = True
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = CurrTerm(func=mdp.uphill_terrain_levels)

        foot_cfg = SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True)

        # FreeGait2's flat task measures root height in world coordinates.
        # On a ramp that would punish a correct robot merely for gaining
        # elevation, so preserve the same target as clearance above its feet.
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_above_feet_l2,
            weight=-8.0,
            params={
                "target_height": TARGET_BASE_HEIGHT,
                "asset_cfg": foot_cfg,
            },
        )
        self.terminations.root_height = DoneTerm(
            func=mdp.base_clearance_above_feet_below_minimum,
            params={
                # This term measures root height relative to the four foot
                # bodies. It must not reuse the old world-Z threshold.
                "minimum_clearance": MIN_BASE_CLEARANCE,
                "asset_cfg": foot_cfg,
            },
        )


@configclass
class QuadLegSlope1EnvCfg_PLAY(QuadLegSlope1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = len(SLOPE_ANGLES)
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_generator = _directional_slope_play_generator()
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
