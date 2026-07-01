"""Bennett crawl locomotion tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-BennettRL-QuadCrawl3-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:BennettQuadCrawlFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettQuadCrawlFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BennettRL-QuadCrawl3-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:BennettQuadCrawlFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettQuadCrawlFlatPPORunnerCfg",
    },
)
