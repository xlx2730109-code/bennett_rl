"""Numerically safe PPO configuration for Bennett Stair1."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class QuadLegStair1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Train a deployable actor with a simulation-only privileged critic."""

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
    }
    num_steps_per_env = 100
    max_iterations = 3000
    save_interval = 50
    clip_actions = 3.5
    experiment_name = "quad_leg_stair/quad_leg_stair1"
    resume = False

    policy = RslRlPpoActorCriticCfg(
        # Reuse Slope4's stable initial amplitude for gait discovery.
        init_noise_std=0.6,
        # Directly optimized scalar std can cross zero. Log-space cannot.
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Slope4's proven value avoids the unbounded log-std growth observed
        # when Stair1 first encounters an unsolved height.
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
