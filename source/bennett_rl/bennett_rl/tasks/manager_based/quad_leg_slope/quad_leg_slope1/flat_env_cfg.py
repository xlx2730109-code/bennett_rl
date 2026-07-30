"""Common locomotion objective and flat debug terrain for Slope1."""

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from ...quad_leg_free_gait.quad_leg_free_gait2.flat_env_cfg import (
    QuadLegFreeGait2FlatEnvCfg,
)


# Every Slope1 lane rises along world +X.  Keep a non-zero command floor and
# expose several speeds so a stationary survival policy cannot dominate PPO.
UPHILL_SPEED_RANGE = (0.16, 0.24)
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

    # The inherited std=0.15 kernel still pays a substantial reward at zero
    # velocity for the old 0.10 m/s command.  The non-zero command floor and
    # moderately narrower kernels make real tracking substantially better than
    # standing while preserving a smooth exploration signal.
    env_cfg.rewards.track_lin_vel_xy_exp.weight = 2.0
    env_cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.12
    env_cfg.rewards.track_lin_vel_xy_fine_exp.weight = 0.8
    env_cfg.rewards.track_lin_vel_xy_fine_exp.params["std"] = 0.05


@configclass
class QuadLegSlope1FlatEnvCfg(QuadLegFreeGait2FlatEnvCfg):
    """Flat A/B environment with the exact Slope1 locomotion objective."""

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
class QuadLegSlope1FlatEnvCfg_PLAY(QuadLegSlope1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
