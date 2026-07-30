"""Register the FreeGait2-based Bennett directional-slope task."""

import gymnasium as gym

from . import agents


_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"


gym.register(
    # old: this task ID previously used a robot-specific family name.
    id="Isaac-BennettRL-QuadLeg-Slope1-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegSlope1EnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope1PPORunnerCfg"
        ),
    },
)

gym.register(
    # old: this Play ID previously used a robot-specific family name.
    id="Isaac-BennettRL-QuadLeg-Slope1-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:QuadLegSlope1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope1PPORunnerCfg"
        ),
    },
)

gym.register(
    # old: this Flat ID previously used a robot-specific family name.
    id="Isaac-BennettRL-QuadLeg-Slope1-Flat-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegSlope1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope1PPORunnerCfg"
        ),
    },
)

gym.register(
    # old: this Flat-Play ID previously used a robot-specific family name.
    id="Isaac-BennettRL-QuadLeg-Slope1-Flat-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegSlope1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegSlope1PPORunnerCfg"
        ),
    },
)
