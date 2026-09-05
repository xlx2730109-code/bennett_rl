# Copyright (c) 2026, Bennett. All rights reserved.

"""PPO agent config for Bennett on the go2 rough-terrain template.

Mirrors ``go2/agents/rsl_rl_ppo_cfg.py`` (UnitreeGo2RoughPPORunnerCfg) with
only the experiment name changed.  Iterations are raised from the template's
1500 to 3000: the terrain curriculum (6 levels x 6 sub-terrains) needs more
iterations than the go2 recipe's default budget; checkpoints save every 50
iterations so the run can be stopped early at any point.
"""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class BennettGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for Bennett on the go2 rough-terrain template."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "bennett_go2/rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
