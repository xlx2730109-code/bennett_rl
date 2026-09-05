# Copyright (c) 2026, Bennett. All rights reserved.

"""Registration for the Bennett rough-terrain task on the go2 template.

Mirrors the official go2 rough-terrain recipe (stairs/boxes/rough/slopes with
terrain-level curriculum); only the robot is swapped to Bennett.
"""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-BennettRL-Rough-BennettGo2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:BennettGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettGo2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BennettRL-Rough-BennettGo2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:BennettGo2RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettGo2RoughPPORunnerCfg",
    },
)
