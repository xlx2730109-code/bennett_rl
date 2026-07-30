"""Directional terrain curriculum for Slope2.

Changes from Slope1:
  - Upgrade threshold lowered from size[0]/2 to size[0]/3 so the
    initial policy only needs to walk ≈2 m past the spawn point
    (instead of ≈2.5 m) to unlock the next slope level.
  - Downgrade requires the robot to actually fall behind (less than
    40 % of expected distance) to prevent spurious demotion when an
    episode happens to end early.
"""

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
    planar distance with signed +X distance because every Slope2 ramp has one
    known uphill direction.  Sideways motion and backwards motion therefore
    cannot unlock steeper ramps.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    uphill_distance = (
        asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    )

    # Upgrade: walk 1/3 of the terrain length (≈2 m for a 6 m lane).
    # This is easier than Slope1's size[0]/2 threshold.
    move_up = uphill_distance > terrain.cfg.terrain_generator.size[0] / 3.0

    # Downgrade: only if the robot lags significantly behind the
    # commanded distance (40 % of expected, instead of 50 %).
    required_distance = (
        torch.abs(command[env_ids, 0]) * env.max_episode_length_s * 0.4
    )
    move_down = (uphill_distance < required_distance) & ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
