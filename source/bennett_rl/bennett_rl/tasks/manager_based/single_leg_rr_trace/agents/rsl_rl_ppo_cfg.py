# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BennettSingleLegTracePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 800
    save_interval = 50
    experiment_name = "single_leg_rr_trace"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BennettSingleLegTrace50HzPPORunnerCfg(BennettSingleLegTracePPORunnerCfg):
    """PPO config for the 50 Hz single-leg trace task."""

    experiment_name = "single_leg_rr_trace_50hz"   


@configclass
class BennettSingleLegTrace250HzPPORunnerCfg(BennettSingleLegTracePPORunnerCfg):
    """PPO config for the 250 Hz single-leg trace task.

    The rollout length is increased so each PPO update still sees about
    0.48 seconds of simulated time: 120 steps / 250 Hz ~= 24 steps / 50 Hz.
    """

    num_steps_per_env = 120
    experiment_name = "single_leg_rr_trace_250hz"


PPORunnerCfg = BennettSingleLegTrace50HzPPORunnerCfg
