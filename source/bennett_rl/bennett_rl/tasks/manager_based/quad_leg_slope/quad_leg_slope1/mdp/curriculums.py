"""Directional terrain curriculum for Slope1."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def uphill_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Advance slope difficulty only from real world-+X uphill progress.

    This follows Isaac Lab's ``terrain_levels_vel`` curriculum, but replaces
    planar distance with signed +X distance because every Slope1 ramp has one
    known uphill direction.  Sideways motion and backwards motion therefore
    cannot unlock steeper ramps.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    uphill_distance = (
        asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    )
    move_up = uphill_distance > terrain.cfg.terrain_generator.size[0] / 2.0
    required_distance = (
        torch.abs(command[env_ids, 0]) * env.max_episode_length_s * 0.5
    )
    move_down = (uphill_distance < required_distance) & ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
