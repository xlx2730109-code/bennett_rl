"""Common locomotion objective and flat debug terrain for Slope2.

Changes from Slope1:
  - Slower speed range (0.10–0.20 vs 0.16–0.24 m/s) so the initial
    random policy can more easily achieve the commanded speed on flat
    ground, producing a clearer advantage signal for PPO.
  - Slightly wider tracking kernels (std 0.14 / 0.06 instead of 0.12 /
    0.05) so the policy is not heavily punished for imperfect tracking
    while it is still learning basic balance.
"""

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from ...quad_leg_free_gait.quad_leg_free_gait2.flat_env_cfg import (
    QuadLegFreeGait2FlatEnvCfg,
)


# Slower command floor = easier start for the random policy.
UPHILL_SPEED_RANGE = (0.10, 0.20)
UPHILL_HEADING_RAD = 0.0
MAX_HEADING_CORRECTION_RAD_S = 0.40


def _configure_straight_uphill_objective(env_cfg: QuadLegFreeGait2FlatEnvCfg) -> None:
    """Configure the common straight-ahead objective used on flat and slope."""

    env_cfg.commands.base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 7.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=UPHILL_SPEED_RANGE,
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(
                -MAX_HEADING_CORRECTION_RAD_S,
                MAX_HEADING_CORRECTION_RAD_S,
            ),
            heading=(UPHILL_HEADING_RAD, UPHILL_HEADING_RAD),
        ),
    )

    # Wider kernels = smoother gradient early in training.
    env_cfg.rewards.track_lin_vel_xy_exp.weight = 2.0
    env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.14
    env_cfg.rewards.track_lin_vel_xy_fine_exp.weight = 0.8
    env_cfg.rewards.track_lin_vel_xy_fine_exp.params["std"] = 0.06


@configclass
class QuadLegSlope2FlatEnvCfg(QuadLegFreeGait2FlatEnvCfg):
    """Flat A/B environment with the exact Slope2 locomotion objective."""

    def __post_init__(self):
        super().__post_init__()
        _configure_straight_uphill_objective(self)

        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42,
            curriculum=False,
            size=(256.0, 256.0),
            num_rows=1,
            num_cols=1,
            color_scheme="none",
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
        self.scene.terrain.use_terrain_origins = False
        self.scene.terrain.max_init_terrain_level = None


@configclass
class QuadLegSlope2FlatEnvCfg_PLAY(QuadLegSlope2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
