"""Register the FreeGait2-based Bennett directional-slope task (Slope2)."""

import gymnasium as gym

from . import agents


_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"


gym.register(
    id="Isaac-BennettRL-QuadLeg-Slope2-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegSlope2EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-QuadLeg-Slope2-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegSlope2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-QuadLeg-Slope2-Flat-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegSlope2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-QuadLeg-Slope2-Flat-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegSlope2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope2PPORunnerCfg"
        ),
    },
)
