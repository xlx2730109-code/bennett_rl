"""RSL-RL PPO configuration for the FreeGait2 directional-slope task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ....quad_leg_free_gait.quad_leg_free_gait2.agents.rsl_rl_ppo_cfg import (
    QuadLegFreeGait2FlatPPORunnerCfg,
)


@configclass
class QuadLegSlope1PPORunnerCfg(QuadLegFreeGait2FlatPPORunnerCfg):
    experiment_name = "quad_leg_slope/quad_leg_slope1"

    # Use log standard deviation so PPO cannot optimize a sampling standard
    # deviation below zero. A fixed task-local learning rate also prevents the
    # adaptive schedule from rising to 1e-2 as it did in the failed run.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
