"""Gait-agnostic foot-contact rewards.

These terms never assign a phase, leg order, diagonal pair, or gait frequency.
They only discourage slipping/flight and reward useful clearance while moving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _moving_mask(env: ManagerBasedRLEnv, command_name: str, command_deadband: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :3], dim=1) >= float(command_deadband)


def _foot_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=1) > float(threshold)


def gait_free_stance_feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    max_value: float,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    """Penalize horizontal velocity of feet that are actually in contact."""

    contact = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide = torch.linalg.vector_norm(velocity_xy, dim=-1) * contact
    moving = _moving_mask(env, command_name, command_deadband).to(torch.float32)
    return moving * torch.clamp(torch.sum(slide, dim=1), max=float(max_value))


def minimum_support_contacts_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    minimum_contacts: int,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    """Penalize fewer than N supporting feet without prescribing which feet."""

    contact_count = _foot_contacts(env, sensor_cfg, threshold).sum(dim=1).to(torch.float32)
    shortfall = torch.relu(float(minimum_contacts) - contact_count)
    moving = _moving_mask(env, command_name, command_deadband).to(torch.float32)
    return moving * torch.square(shortfall)


def gait_free_swing_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    target_height: float,
    sigma: float,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    """Reward moderate clearance for whichever feet the policy elects to swing."""

    contact = _foot_contacts(env, sensor_cfg, threshold)
    contact_float = contact.to(torch.float32)
    swing = (~contact).to(torch.float32)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    stance_count = contact_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    stance_height = (foot_height * contact_float).sum(dim=1, keepdim=True) / stance_count
    relative_height = torch.clamp(foot_height - stance_height, min=0.0)

    target = max(float(target_height), 1.0e-6)
    score = torch.exp(-torch.square(relative_height - target) / max(float(sigma) ** 2, 1.0e-8))
    lift_gate = torch.clamp(relative_height / (0.25 * target), min=0.0, max=1.0)
    swing_count = swing.sum(dim=1).clamp_min(1.0)
    has_support = (contact_float.sum(dim=1) > 0.0).to(torch.float32)
    moving = _moving_mask(env, command_name, command_deadband).to(torch.float32)
    return moving * has_support * torch.sum(score * lift_gate * swing, dim=1) / swing_count
