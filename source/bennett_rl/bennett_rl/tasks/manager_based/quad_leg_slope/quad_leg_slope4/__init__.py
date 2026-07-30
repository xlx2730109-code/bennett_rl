"""Register the standalone Bennett Slope4 task."""

import gymnasium as gym

from . import agents


_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"


gym.register(
    id="Isaac-BennettRL-QuadLeg-Slope4-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegSlope4EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope4PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-QuadLeg-Slope4-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegSlope4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope4PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-QuadLeg-Slope4-Flat-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegSlope4FlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope4PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-QuadLeg-Slope4-Flat-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegSlope4FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope4PPORunnerCfg"
        ),
    },
)
