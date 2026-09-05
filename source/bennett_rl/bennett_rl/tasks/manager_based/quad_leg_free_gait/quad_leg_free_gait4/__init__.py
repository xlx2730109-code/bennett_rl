"""Registration for the higher-clearance free-gait Bennett flat-ground task."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-BennettRL-Flat-QuadLeg-FreeGait4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegFreeGait4FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegFreeGait4FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BennettRL-Flat-QuadLeg-FreeGait4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegFreeGait4FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegFreeGait4FlatPPORunnerCfg",
    },
)
