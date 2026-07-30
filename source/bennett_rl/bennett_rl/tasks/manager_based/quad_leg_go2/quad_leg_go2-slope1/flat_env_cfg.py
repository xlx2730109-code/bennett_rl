# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from .rough_env_cfg import QuadLegGo2Slope1EnvCfg


@configclass
class QuadLegGo2Slope1FlatEnvCfg(QuadLegGo2Slope1EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Use a local flat mesh instead of slope terrain.
        # This lets you debug the locomotion policy on flat ground first,
        # since the first slope-curriculum level *is* flat anyway.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42,
            curriculum=False,
            size=(256.0, 256.0),
            num_rows=1,
            num_cols=1,
            color_scheme="none",
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
        self.scene.terrain.use_terrain_origins = False
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class QuadLegGo2Slope1FlatEnvCfg_PLAY(QuadLegGo2Slope1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
