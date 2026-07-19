# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  #每个环境的步骤数，如果训练波动较大，可以适当增加这个值以获得更稳定的训练，但会增加每次更新的时间
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],  #定义actor网络的隐藏层维度为[512, 256, 128]
        critic_hidden_dims=[512, 256, 128], #定义critic网络的隐藏层维度为[512, 256, 128],任务简单可以适当减小网络规模,如[128, 128, 128]
        activation="elu",    #定义激活函数为elu
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


@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
