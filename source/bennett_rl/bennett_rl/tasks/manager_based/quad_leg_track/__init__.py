import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Bennett-QuadLeg-Track-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bennett_rl_env_cfg:BennettQuadLegTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettQuadLegTrackPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Bennett-QuadLeg-Track-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bennett_rl_env_cfg:BennettQuadLegTrackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettQuadLegTrackPPORunnerCfg",
    },
)
