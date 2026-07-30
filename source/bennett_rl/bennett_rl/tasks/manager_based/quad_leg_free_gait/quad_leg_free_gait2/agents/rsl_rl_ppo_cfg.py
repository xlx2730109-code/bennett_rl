"""RSL-RL PPO configuration for QuadLeg FreeGait2."""

from isaaclab.utils import configclass

from ...quad_leg_free_gait1.agents.rsl_rl_ppo_cfg import QuadLegFreeGait1FlatPPORunnerCfg


@configclass
class QuadLegFreeGait2FlatPPORunnerCfg(QuadLegFreeGait1FlatPPORunnerCfg):
    """Keep FreeGait1 PPO unchanged so the foot-reward hypothesis is isolated."""

    experiment_name = "quad_leg_free_gait/quad_leg_free_gait2/flat"
