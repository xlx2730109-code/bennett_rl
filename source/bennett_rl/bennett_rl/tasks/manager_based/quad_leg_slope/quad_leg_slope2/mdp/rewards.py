"""Coordinate-invariant base-height terms for sloped terrain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _base_clearance_above_feet(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return asset.data.root_pos_w[:, 2] - torch.mean(foot_z, dim=1)


def base_height_above_feet_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize base clearance error without depending on world elevation."""

    clearance = _base_clearance_above_feet(env, asset_cfg)
    return torch.square(clearance - float(target_height))


def base_clearance_above_feet_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_clearance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate a collapsed robot equally on the approach, ramp, or top."""

    return _base_clearance_above_feet(env, asset_cfg) < float(minimum_clearance)
