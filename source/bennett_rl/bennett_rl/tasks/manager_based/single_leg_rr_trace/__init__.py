# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Bennett-SingleLeg-RR-Trace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bennett_rl_env_cfg:BennettSingleLegTraceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettSingleLegTrace50HzPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Bennett-SingleLeg-RR-Trace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bennett_rl_env_cfg:BennettSingleLegTraceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettSingleLegTrace50HzPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Bennett-SingleLeg-RR-Trace-50Hz-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bennett_rl_env_cfg:BennettSingleLegTrace50HzEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettSingleLegTrace50HzPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Bennett-SingleLeg-RR-Trace-50Hz-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bennett_rl_env_cfg:BennettSingleLegTrace50HzEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettSingleLegTrace50HzPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Bennett-SingleLeg-RR-Trace-250Hz-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bennett_rl_env_cfg:BennettSingleLegTrace250HzEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettSingleLegTrace250HzPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Bennett-SingleLeg-RR-Trace-250Hz-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bennett_rl_env_cfg:BennettSingleLegTrace250HzEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BennettSingleLegTrace250HzPPORunnerCfg",
    },
)
