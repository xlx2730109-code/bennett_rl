"""PPO configuration for the standalone Bennett Gravel1 task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class QuadLegGravel1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # The deployed actor sees only the hardware-compatible policy group. The
    # critic additionally sees simulation-only terrain and base-velocity data.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "privileged"],
    }
    # A 2 s rollout covers the 0.65--1.20 s leg-participation reward state and
    # matches the numerically stable Bennett Slope4 training setup.
    num_steps_per_env = 100
    max_iterations = 2000
    save_interval = 50
    clip_actions = 3.5
    experiment_name = "quad_leg_gravel/quad_leg_gravel1"
    resume = False

    policy = RslRlPpoActorCriticCfg(
        # RSL-RL's scalar mode optimizes std directly and can cross below zero.
        # Log parameterization guarantees a positive Normal scale.
        init_noise_std=0.6,
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
