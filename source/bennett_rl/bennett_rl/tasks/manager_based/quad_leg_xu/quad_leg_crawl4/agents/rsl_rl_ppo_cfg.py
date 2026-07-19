"""RSL-RL PPO configuration for Bennett crawl4."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BennettCrawl4FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100          # 0.02s * 100 = 2.0s = 一个完整步态周期（0.5Hz）
    max_iterations = 2000
    save_interval = 50
    experiment_name = "quad_leg_xu/quad_leg_crawl4"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,          # 更高的初始探索噪声
        actor_obs_normalization=False, # 观测已通过 clip/scale 归一化，不需要网络内再归一化
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,            # 略高的熵系数，鼓励探索
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,         # 稍高学习率，配合 adaptive schedule
        schedule="adaptive",          # adaptive scheduler 自动调节 LR
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


PPORunnerCfg = BennettCrawl4FlatPPORunnerCfg
