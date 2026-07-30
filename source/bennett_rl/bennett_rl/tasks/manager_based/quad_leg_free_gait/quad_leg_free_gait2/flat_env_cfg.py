"""Second gait-free Bennett task: require all four legs to participate."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..quad_leg_free_gait1.flat_env_cfg import (
    COMMAND_DEADBAND,
    FOOT_BODIES,
    QuadLegFreeGait1FlatEnvCfg,
)

from . import mdp


@configclass
class QuadLegFreeGait2FlatEnvCfg(QuadLegFreeGait1FlatEnvCfg):
    """Add only a speed-conditioned per-leg participation constraint."""

    def __post_init__(self):
        super().__post_init__()

        contact_cfg = SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True)
        foot_cfg = SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True)
        self.rewards.leg_lift_starvation = RewTerm(
            func=mdp.gait_free_leg_lift_starvation_l2,
            weight=-0.40,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "contact_threshold": 1.0,
                "valid_lift_height": 0.020,
                "slow_allowed_time": 1.20,
                "fast_allowed_time": 0.65,
                "min_equivalent_speed": 0.06,
                "max_equivalent_speed": 0.35,
                "yaw_equivalent_radius": 0.20,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "max_normalized_excess": 2.0,
            },
        )


@configclass
class QuadLegFreeGait2FlatEnvCfg_PLAY(QuadLegFreeGait2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
