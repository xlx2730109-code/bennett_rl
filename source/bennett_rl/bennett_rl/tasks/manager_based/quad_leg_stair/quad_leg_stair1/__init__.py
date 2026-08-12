"""Gym registration for the independent Bennett Stair1 task."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-BennettRL-QuadLeg-Stair1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegStair1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegStair1PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BennettRL-QuadLeg-Stair1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegStair1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegStair1PPORunnerCfg",
    },
)

