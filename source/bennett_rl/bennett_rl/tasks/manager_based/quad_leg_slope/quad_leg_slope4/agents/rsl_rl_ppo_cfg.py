"""Standalone RSL-RL PPO configuration for Bennett Slope4."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class QuadLegSlope4PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 100
    max_iterations = 3000
    save_interval = 50
    clip_actions = 3.5
    experiment_name = "quad_leg_slope/quad_leg_slope4"
    resume = False

    policy = RslRlPpoActorCriticCfg(
        # Keep log-std so the distribution scale cannot become negative.
        # Slope3 converged with roughly 3.4x Trot1's policy noise, which made
        # joint targets change too abruptly even though touchdown impact was low.
        init_noise_std=0.6,
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
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
