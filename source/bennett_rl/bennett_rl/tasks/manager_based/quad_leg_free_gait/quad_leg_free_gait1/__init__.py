"""Registration for the first gait-free Bennett flat-ground task."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-BennettRL-Flat-QuadLeg-FreeGait1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegFreeGait1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegFreeGait1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BennettRL-Flat-QuadLeg-FreeGait1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegFreeGait1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegFreeGait1FlatPPORunnerCfg",
    },
)
