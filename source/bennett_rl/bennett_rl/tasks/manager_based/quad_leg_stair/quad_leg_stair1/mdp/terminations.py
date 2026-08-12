"""Failure and success termination terms for Bennett Stair1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.terrains import TerrainImporter

from .support import support_foot_height_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_clearance_above_feet_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_clearance: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float,
    minimum_support_contacts: int,
) -> torch.Tensor:
    """Terminate low clearance only from a reliable support-foot reference."""

    asset: Articulation = env.scene[asset_cfg.name]
    support_height, valid = support_foot_height_w(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        contact_threshold=contact_threshold,
        minimum_support_contacts=minimum_support_contacts,
    )
    clearance = asset.data.root_pos_w[:, 2] - support_height
    return valid & (clearance < float(minimum_clearance))


class insufficient_stair_progress(ManagerTermBase):
    """Fail moving episodes that repeatedly make too little world-+X progress."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._anchor_x = torch.zeros(env.num_envs, device=env.device)
        self._elapsed = torch.zeros(env.num_envs, device=env.device)
        self._commanded_distance = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._anchor_x[env_ids] = 0.0
        self._elapsed[env_ids] = 0.0
        self._commanded_distance[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        command_deadband: float,
        window_s: float,
        minimum_command_fraction: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        if window_s <= 0.0:
            raise ValueError("window_s must be positive.")
        if not 0.0 < minimum_command_fraction <= 1.0:
            raise ValueError("minimum_command_fraction must be in (0, 1].")

        asset: Articulation = env.scene[asset_cfg.name]
        position_x = asset.data.root_pos_w[:, 0]
        command_x = env.command_manager.get_command(command_name)[:, 0]
        moving = command_x >= float(command_deadband)

        newly_active = moving & ~self._initialized
        self._anchor_x[newly_active] = position_x[newly_active]
        self._elapsed[newly_active] = 0.0
        self._commanded_distance[newly_active] = 0.0
        self._initialized[newly_active] = True

        inactive = ~moving
        self._elapsed[inactive] = 0.0
        self._commanded_distance[inactive] = 0.0
        self._initialized[inactive] = False

        active = moving & self._initialized & ~newly_active
        self._elapsed[active] += float(env.step_dt)
        self._commanded_distance[active] += command_x[active] * float(env.step_dt)

        window_complete = active & (self._elapsed >= float(window_s))
        actual_progress = position_x - self._anchor_x
        required_progress = self._commanded_distance * float(minimum_command_fraction)
        failure = window_complete & (actual_progress < required_progress)

        passed_window = window_complete & ~failure
        self._anchor_x[passed_window] = position_x[passed_window]
        self._elapsed[passed_window] = 0.0
        self._commanded_distance[passed_window] = 0.0
        return failure


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
