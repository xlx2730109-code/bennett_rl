"""Isolated DM-J8006 motor-model A/B tasks based on Bennett Trot1."""

import gymnasium as gym

from . import agents


_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"


gym.register(
    id="Isaac-BennettRL-Flat-QuadLeg-Trot-MotorAB-A-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegTrotMotorABACfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegTrotMotorABARunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-Flat-QuadLeg-Trot-MotorAB-A-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegTrotMotorABACfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegTrotMotorABARunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-Flat-QuadLeg-Trot-MotorAB-B-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegTrotMotorABBCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegTrotMotorABBRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-BennettRL-Flat-QuadLeg-Trot-MotorAB-B-Play-v0",
    entry_point=_ENTRY_POINT,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:QuadLegTrotMotorABBCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:QuadLegTrotMotorABBRunnerCfg"
        ),
    },
)
