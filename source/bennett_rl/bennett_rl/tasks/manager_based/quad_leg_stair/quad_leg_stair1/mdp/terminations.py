"""Failure and success termination terms for Bennett Stair1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_clearance_above_feet_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_clearance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate a collapsed posture using foot-relative, not world-Z, height."""

    asset: Articulation = env.scene[asset_cfg.name]
    mean_foot_height = torch.mean(asset.data.body_pos_w[:, asset_cfg.body_ids, 2], dim=1)
    clearance = asset.data.root_pos_w[:, 2] - mean_foot_height
    return clearance < float(minimum_clearance)


def outside_stair_lane(
    env: ManagerBasedRLEnv,
    maximum_lateral_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate robots that leave the usable straight staircase lane."""

    asset: Articulation = env.scene[asset_cfg.name]
    relative_y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.abs(relative_y) > float(maximum_lateral_distance)


class stair_top_success(ManagerTermBase):
    """Finish after all four feet reach and hold the top platform.

    Foot-contact order is deliberately unconstrained.  Requiring all four
    feet to remain supported while a nonzero forward command is active can
    reject a valid ascent and let the robot walk off the far edge instead of
    resetting at the bottom.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._hold_time = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._hold_time.zero_()
        else:
            self._hold_time[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        top_platform_start_distances: tuple[float, ...],
        foot_margin: float,
        hold_time_s: float,
        minimum_clearance: float,
        minimum_upright_cosine: float,
        maximum_lateral_distance: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        terrain: TerrainImporter = env.scene.terrain
        distances = torch.as_tensor(
            top_platform_start_distances, device=env.device, dtype=torch.float32
        )
        if torch.any(terrain.terrain_types >= distances.numel()):
            raise RuntimeError("Terrain type has no matching top-platform distance.")

        asset: Articulation = env.scene[asset_cfg.name]
        foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
        relative_foot_x = foot_pos[:, :, 0] - env.scene.env_origins[:, 0, None]
        required_x = distances[terrain.terrain_types] + float(foot_margin)
        all_feet_on_top = torch.amin(relative_foot_x, dim=1) >= required_x

        base_clearance = asset.data.root_pos_w[:, 2] - torch.mean(foot_pos[:, :, 2], dim=1)
        upright = -asset.data.projected_gravity_b[:, 2] >= float(minimum_upright_cosine)
        relative_y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
        in_lane = torch.abs(relative_y) <= float(maximum_lateral_distance)
        success_now = (
            all_feet_on_top
            & (base_clearance >= float(minimum_clearance))
            & upright
            & in_lane
        )

        self._hold_time = torch.where(
            success_now,
            self._hold_time + float(env.step_dt),
            torch.zeros_like(self._hold_time),
        )
        return self._hold_time >= float(hold_time_s)
