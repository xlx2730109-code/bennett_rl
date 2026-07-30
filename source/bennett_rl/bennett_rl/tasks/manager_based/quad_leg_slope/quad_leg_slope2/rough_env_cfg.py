"""FreeGait2 locomotion on directional slope terrain — Slope2.

Design changes from Slope1 (training was stuck at level 0):
  1. Finer slope gradation at the low end (0.25° steps) so level 1 is
     nearly flat and the policy can survive the first promotion.
  2. Longer approach (1.8 m) and top platform (1.5 m) so the robot has
     flat ground to stabilise before / after the ramp.
  3. More forgiving minimum clearance (0.15 m) so a slightly sagging
     stance on a ramp does not terminate.
  4. Lighter base-height penalty (weight -6.0 vs -8.0) and lighter
     leg-lift-starvation penalty (weight -0.25 vs -0.40) to reduce
     conflicting gradient signals early in training.
"""

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
from .flat_env_cfg import QuadLegSlope2FlatEnvCfg
from .slope_terrain import DirectionalSlopeTerrainCfg


# ── gentler slope ladder ──────────────────────────────────────────────
# Finer steps at the low end so the first curriculum promotion (level 1)
# introduces only a barely perceptible tilt.
SLOPE_ANGLES = tuple(
    math.radians(angle)
    for angle in (0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.25, 5.0, 5.75, 6.5)
)

# Longer flat approach + top platform = more room to recover gait.
SLOPE_TERRAIN_SIZE = (6.0, 2.8)
SLOPE_APPROACH_LENGTH = 1.80
SLOPE_TOP_PLATFORM_LENGTH = 1.50
SLOPE_SPAWN_X = 1.20
SLOPE_TRANSITION_LENGTH = 0.25
MIN_BASE_CLEARANCE = 0.15


# ── generators ────────────────────────────────────────────────────────


def _directional_slope_curriculum_generator() -> TerrainGeneratorCfg:
    """Build a row-wise 0-to-6.5-degree slope curriculum."""

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


# ── env config ────────────────────────────────────────────────────────


@configclass
class QuadLegSlope2EnvCfg(QuadLegSlope2FlatEnvCfg):
    """Use the common Slope2 objective and replace only the flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _directional_slope_curriculum_generator()
        self.scene.terrain.use_terrain_origins = True
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner = None

        # The Slope2 curriculum uses a gentler upgrade threshold.
        self.curriculum.terrain_levels = CurrTerm(func=mdp.uphill_terrain_levels)

        foot_cfg = SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True)

        # ── rewards ────────────────────────────────────────────────
        # Base height: weight reduced from -8 to -6 so the policy is
        # not overwhelmingly punished for a slightly-low stance on the
        # ramp.
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_above_feet_l2,
            weight=-6.0,
            params={
                "target_height": TARGET_BASE_HEIGHT,
                "asset_cfg": foot_cfg,
            },
        )

        # ── terminations ───────────────────────────────────────────
        self.terminations.root_height = DoneTerm(
            func=mdp.base_clearance_above_feet_below_minimum,
            params={
                "minimum_clearance": MIN_BASE_CLEARANCE,
                "asset_cfg": foot_cfg,
            },
        )

        # ── tone down leg-lift starvation ──────────────────────────
        # Inherited from FreeGait2 with weight -0.40.  At that strength
        # the penalty dominated (-0.72 at step 266 in Slope1) while the
        # policy was still learning basic balance.  Reducing to -0.25
        # gives balance more room before the starvation term dominates.
        if self.rewards.leg_lift_starvation is not None:
            self.rewards.leg_lift_starvation.weight = -0.25


# ── play config ───────────────────────────────────────────────────────


@configclass
class QuadLegSlope2EnvCfg_PLAY(QuadLegSlope2EnvCfg):
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
