"""Register the standalone Bennett Gravel1 task."""

import gymnasium as gym

from . import agents


_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"


gym.register(
    id="Isaac-BennettRL-QuadLeg-Gravel1-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegGravel1EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegGravel1PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-QuadLeg-Gravel1-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegGravel1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegGravel1PPORunnerCfg"
        ),
    },
)
