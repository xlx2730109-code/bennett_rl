"""Gait-agnostic stair rewards for Bennett Stair1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter

from .support import support_foot_height_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _moving_mask(env: ManagerBasedRLEnv, command_name: str, command_deadband: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return command[:, 0] >= float(command_deadband)


def _foot_contacts(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(forces, dim=-1).amax(dim=1) > float(threshold)


def base_height_above_feet_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float,
    minimum_support_contacts: int,
) -> torch.Tensor:
    """Penalize body-height error relative only to reliable support feet."""

    asset: Articulation = env.scene[asset_cfg.name]
    support_height, valid = support_foot_height_w(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        contact_threshold=contact_threshold,
        minimum_support_contacts=minimum_support_contacts,
    )
    clearance = asset.data.root_pos_w[:, 2] - support_height
    return torch.square(clearance - float(target_height)) * valid.to(clearance.dtype)


def uphill_velocity_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed world-+X progress without granting credit for standing."""

    command = env.command_manager.get_command(command_name)
    target_speed = torch.clamp(torch.abs(command[:, 0]), min=0.05)
    asset: Articulation = env.scene[asset_cfg.name]
    normalized = asset.data.root_lin_vel_w[:, 0] / target_speed
    return torch.clamp(normalized, min=-1.0, max=1.0)


def track_uphill_world_velocity_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track the fixed world-+X stair direction instead of body-frame motion.

    Every Stair1 lane ascends along world +X.  Using the standard body-frame
    velocity kernel lets turning or local rocking earn tracking reward without
    making uphill progress.  This kernel also rejects world-Y drift while
    retaining the same command magnitude and exponential shape.
    """

    if float(std) <= 0.0:
        raise ValueError("std must be positive.")
    command = env.command_manager.get_command(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_error = torch.square(command[:, 0] - asset.data.root_lin_vel_w[:, 0])
    velocity_error += torch.square(asset.data.root_lin_vel_w[:, 1])
    score = torch.exp(-velocity_error / float(std) ** 2)

    # A wide exponential kernel is useful for discovering motion from scratch,
    # but its raw value is also high while standing.  Center it on the exact
    # zero-velocity score so stationary behavior receives zero rather than a
    # large survival-compatible reward.  Preserve the normal stationary target
    # when an explicit zero command is used during deployment.
    stationary_score = torch.exp(-torch.square(command[:, 0]) / float(std) ** 2)
    normalizer = torch.clamp(1.0 - stationary_score, min=1.0e-4)
    centered_score = torch.clamp(
        (score - stationary_score) / normalizer,
        min=-1.0,
        max=1.0,
    )
    moving_command = torch.abs(command[:, 0]) >= 1.0e-3
    return torch.where(moving_command, centered_score, score)


def moving_touchdown_impact_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    soft_force_limit: float,
    max_normalized_excess: float,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    """Penalize only the excessive part of a moving foot's touchdown force."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].to(
        torch.float32
    )
    force = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1
    ).amax(dim=1)
    excess = torch.clamp(
        torch.relu(force - float(soft_force_limit))
        / max(float(soft_force_limit), 1.0e-6),
        max=float(max_normalized_excess),
    )
    moving = _moving_mask(env, command_name, command_deadband).unsqueeze(1).to(
        torch.float32
    )
    return torch.sum(torch.square(excess) * first_contact * moving, dim=1)


def stair_swing_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float,
    clearance_margin: float,
    height_levels: tuple[float, ...],
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    """Reward adequate swing clearance without prescribing phase or leg order."""

    contact = _foot_contacts(env, sensor_cfg, contact_threshold)
    swing = ~contact
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    negative_infinity = torch.full_like(foot_height, -torch.inf)
    highest_support = torch.where(contact, foot_height, negative_infinity).amax(dim=1, keepdim=True)
    has_support = contact.any(dim=1, keepdim=True)
    fallback = foot_height.amin(dim=1, keepdim=True)
    highest_support = torch.where(has_support, highest_support, fallback)
    relative_height = torch.clamp(foot_height - highest_support, min=0.0)

    terrain: TerrainImporter = env.scene.terrain
    level_heights = torch.as_tensor(height_levels, device=env.device, dtype=foot_height.dtype)
    level_ids = torch.clamp(terrain.terrain_levels, min=0, max=len(height_levels) - 1)
    target = level_heights[level_ids].unsqueeze(1) + float(clearance_margin)
    clearance_score = torch.clamp(relative_height / target.clamp_min(1.0e-6), min=0.0, max=1.0)
    swing_count = swing.sum(dim=1).clamp_min(1).to(torch.float32)
    moving = _moving_mask(env, command_name, command_deadband).to(torch.float32)
    return moving * has_support.squeeze(1).to(torch.float32) * torch.sum(
        clearance_score * swing.to(torch.float32), dim=1
    ) / swing_count


def stair_swing_overclearance_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float,
    maximum_clearance: float,
    normalization: float,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    """Penalize excessive swing height without prescribing gait or leg order.

    A blind policy needs one clearance envelope that remains safe for the
    largest trained step.  The limit is therefore absolute rather than tied to
    the current curriculum level: otherwise the actor would be punished for a
    conservative 10-cm-capable swing on the easier rows that it cannot observe.
    Every foot uses the same limit and only its excess above that limit is
    penalized.
    """

    if maximum_clearance <= 0.0:
        raise ValueError("maximum_clearance must be positive.")
    if normalization <= 0.0:
        raise ValueError("normalization must be positive.")

    contact = _foot_contacts(env, sensor_cfg, contact_threshold)
    swing = ~contact
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    negative_infinity = torch.full_like(foot_height, -torch.inf)
    highest_support = torch.where(contact, foot_height, negative_infinity).amax(
        dim=1, keepdim=True
    )
    has_support = contact.any(dim=1, keepdim=True)
    # Keep the arithmetic finite even in an all-feet-swing transition frame.
    highest_support = torch.where(
        has_support, highest_support, foot_height.amin(dim=1, keepdim=True)
    )
    relative_height = torch.clamp(foot_height - highest_support, min=0.0)
    normalized_excess = torch.clamp(
        torch.relu(relative_height - float(maximum_clearance))
        / float(normalization),
        max=2.0,
    )
    moving = _moving_mask(env, command_name, command_deadband).to(torch.float32)
    return moving * has_support.squeeze(1).to(torch.float32) * torch.sum(
        torch.square(normalized_excess) * swing.to(torch.float32), dim=1
    )


def action_soft_limit_l2(
    env: ManagerBasedRLEnv,
    soft_limit: float,
    hard_limit: float,
) -> torch.Tensor:
    """Penalize only the part of an applied action near the runner clip.

    PPO may emit values outside the runner clip even though the environment
    always receives a clipped action.  Keeping a penalty-free interior retains
    the large stair-climbing workspace, while the boundary cost discourages a
    policy from using clipping as a persistent joint target.
    """

    if soft_limit < 0.0:
        raise ValueError("soft_limit must be non-negative.")
    if hard_limit <= soft_limit:
        raise ValueError("hard_limit must be greater than soft_limit.")

    action = env.action_manager.action
    normalized_excess = torch.clamp(
        torch.relu(torch.abs(action) - float(soft_limit))
        / float(hard_limit - soft_limit),
        max=1.0,
    )
    return torch.sum(torch.square(normalized_excess), dim=1)


def minimum_support_contacts_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    minimum_contacts: int,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    """Softly discourage flight while leaving contact order unconstrained."""

    contact_count = _foot_contacts(env, sensor_cfg, threshold).sum(dim=1).to(torch.float32)
    shortfall = torch.relu(float(minimum_contacts) - contact_count)
    moving = _moving_mask(env, command_name, command_deadband).to(torch.float32)
    return moving * torch.square(shortfall)


def lane_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Keep the blind policy in its staircase lane without enforcing a gait."""

    asset: Articulation = env.scene[asset_cfg.name]
    relative_y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.square(relative_y)
