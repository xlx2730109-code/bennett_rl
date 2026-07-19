"""Reward terms for Bennett crawl locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .observations import crawl_global_phase
from .gait_scheduler import compute_crawl_schedule, compute_stand_schedule

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _crawl_schedule(env: ManagerBasedRLEnv, frequency_hz: float, duty_factor: float):
    return compute_crawl_schedule(
        crawl_global_phase(env, frequency_hz=frequency_hz),
        duty_factor=duty_factor,
    )


def _moving_mask(env: ManagerBasedRLEnv, command_name: str, command_deadband: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.abs(command[:, 0]) >= command_deadband


def _commanded_schedule(
    env: ManagerBasedRLEnv,
    frequency_hz: float,
    duty_factor: float,
    command_name: str,
    command_deadband: float,
):
    phase = crawl_global_phase(env, frequency_hz=frequency_hz)
    crawl_schedule = compute_crawl_schedule(phase, duty_factor=duty_factor)
    stand_schedule = compute_stand_schedule(phase)
    moving = _moving_mask(env, command_name, command_deadband).unsqueeze(1)
    return type(crawl_schedule)(
        global_phase=torch.where(moving.squeeze(1), crawl_schedule.global_phase, torch.zeros_like(crawl_schedule.global_phase)),
        leg_phase=torch.where(moving, crawl_schedule.leg_phase, stand_schedule.leg_phase),
        desired_contact=torch.where(moving, crawl_schedule.desired_contact, stand_schedule.desired_contact),
        desired_swing=torch.where(moving, crawl_schedule.desired_swing, stand_schedule.desired_swing),
    )


def _foot_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_force = contact_force.norm(dim=-1).max(dim=1)[0]
    return contact_force > threshold


def track_fixed_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    target: float,
    sigma: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking a fixed forward crawl velocity in the robot base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_lin_vel_b[:, 0] - target)
    return torch.exp(-error / max(sigma**2, 1.0e-6))


def lateral_yaw_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize sideways drift and yaw spin during straight crawl."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1]) + torch.square(asset.data.root_ang_vel_b[:, 2])


def base_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity that makes the base rock during crawl."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def pitch_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base pitch (forward/backward tilt) using L1 norm.

    L1 is more effective than L2 for moderate-to-large pitch angles because
    it doesn't square the angle, giving a stronger signal when the robot pitches.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.projected_gravity_b[:, 0])


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    sigma: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping the root height close to the crawl standing height."""
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = asset.data.root_pos_w[:, 2] - target_height
    return torch.exp(-torch.square(height_error) / max(sigma**2, 1.0e-6))


def default_joint_pose_exp(
    env: ManagerBasedRLEnv,
    sigma: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward actuated joints staying near the configured default crawl pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.mean(torch.square(joint_error), dim=1) / max(sigma**2, 1.0e-6))


def crawl_contact_match(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Reward feet whose measured contact state matches the scheduled crawl contact state."""
    contacts = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    desired_contacts = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband).desired_contact.to(
        torch.float32
    )
    return 1.0 - torch.mean(torch.abs(contacts - desired_contacts), dim=1)


def crawl_missing_stance_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Count scheduled stance feet that are not in contact."""
    contacts = _foot_contacts(env, sensor_cfg, threshold)
    desired_contacts = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband).desired_contact
    return torch.sum((desired_contacts & ~contacts).to(torch.float32), dim=1)


def crawl_extra_swing_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Count scheduled swing feet that are still touching the ground."""
    contacts = _foot_contacts(env, sensor_cfg, threshold)
    desired_swing = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband).desired_swing
    return torch.sum((desired_swing & contacts).to(torch.float32), dim=1)


def crawl_stance_feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    threshold: float = 1.0,
    max_value: float = 2.0,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Penalize horizontal sliding of feet that should be supporting the robot."""
    contacts = _foot_contacts(env, sensor_cfg, threshold)
    desired_contacts = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband).desired_contact
    stance_mask = contacts & desired_contacts

    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide = torch.linalg.norm(foot_vel_xy, dim=-1) * stance_mask.to(torch.float32)
    return torch.clamp(torch.sum(slide, dim=1), max=max_value)


def crawl_swing_foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    target_height: float,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Reward scheduled swing feet being at least target_height above the environment origin."""
    desired_swing = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband).desired_swing.to(
        torch.float32
    )
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    clearance = torch.clamp(foot_height / max(target_height, 1.0e-6), min=0.0, max=1.0)
    desired_count = torch.clamp(torch.sum(desired_swing, dim=1), min=1.0)
    return torch.sum(clearance * desired_swing, dim=1) / desired_count


def stand_still_base_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base horizontal velocity only while the command is stopped."""

    asset: Articulation = env.scene[asset_cfg.name]
    stopped = torch.logical_not(_moving_mask(env, command_name, command_deadband)).to(torch.float32)
    return stopped * torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def stand_still_joint_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize joint motion only while the command is stopped."""

    asset: Articulation = env.scene[asset_cfg.name]
    stopped = torch.logical_not(_moving_mask(env, command_name, command_deadband)).to(torch.float32)
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return stopped * torch.mean(torch.square(joint_vel), dim=1)


def stand_still_joint_pose_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
    sigma: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward default joint pose only while the command is stopped."""

    stopped = torch.logical_not(_moving_mask(env, command_name, command_deadband)).to(torch.float32)
    return stopped * default_joint_pose_exp(env, sigma=sigma, asset_cfg=asset_cfg)

