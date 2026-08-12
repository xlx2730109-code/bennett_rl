"""Terrain-relative termination terms for Gravel1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_above_terrain_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_clearance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when root clearance above scanned terrain becomes unsafe."""

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    clearance = asset.data.root_pos_w[:, 2] - terrain_height
    return clearance < float(minimum_clearance)
