import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-BennettRL-Go2-Slope2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegGo2Slope2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegGo2Slope2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BennettRL-Go2-Slope2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegGo2Slope2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegGo2Slope2PPORunnerCfg",
    },
)
