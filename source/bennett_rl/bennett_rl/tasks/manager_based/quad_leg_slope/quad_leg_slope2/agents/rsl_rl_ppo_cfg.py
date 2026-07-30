"""RSL-RL PPO configuration for the Slope2 directional-slope task.

Changes from Slope1:
  - entropy_coef 0.01 → 0.02 so the policy explores more aggressively
    early on, increasing the chance of discovering a viable slope gait.
  - num_learning_epochs 5 → 4 to reduce over-fitting to the
    flat-terrain data that dominates the early-training buffer.
  - learning_rate 3.0e-4 → 2.0e-4 for more stable gradient steps.
  - desired_kl 0.01 → 0.02 so the trust region does not prematurely
    freeze policy improvement.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ....quad_leg_free_gait.quad_leg_free_gait2.agents.rsl_rl_ppo_cfg import (
    QuadLegFreeGait2FlatPPORunnerCfg,
)


@configclass
class QuadLegSlope2PPORunnerCfg(QuadLegFreeGait2FlatPPORunnerCfg):
    experiment_name = "quad_leg_slope/quad_leg_slope2"

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
        entropy_coef=0.02,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )
