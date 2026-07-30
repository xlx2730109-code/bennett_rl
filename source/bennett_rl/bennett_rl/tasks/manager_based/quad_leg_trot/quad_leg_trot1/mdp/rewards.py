"""Soft diagonal-trot shaping terms for Bennett."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .gait import get_commanded_trot_schedule, smooth_swing_profile, soft_swing_weights

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _moving_mask(env: ManagerBasedRLEnv, command_name: str, command_deadband: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :3], dim=1) >= float(command_deadband)


def _foot_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=1) > float(threshold)


def trot_contact_match(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    transition_fraction: float,
    command_name: str,
    command_deadband: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    min_equivalent_speed: float,
    max_equivalent_speed: float,
    low_speed_duty_factor: float,
    high_speed_duty_factor: float,
    swing_height: float,
    yaw_equivalent_radius: float,
) -> torch.Tensor:
    schedule = get_commanded_trot_schedule(
        env,
        command_name,
        command_deadband,
        min_frequency_hz,
        max_frequency_hz,
        min_equivalent_speed,
        max_equivalent_speed,
        low_speed_duty_factor,
        high_speed_duty_factor,
        swing_height,
        yaw_equivalent_radius,
    )
    contact = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    swing = soft_swing_weights(schedule, transition_fraction)
    desired_contact = 1.0 - swing
    return 1.0 - torch.mean(torch.abs(contact - desired_contact), dim=1)


def trot_missing_stance_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    transition_fraction: float,
    command_name: str,
    command_deadband: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    min_equivalent_speed: float,
    max_equivalent_speed: float,
    low_speed_duty_factor: float,
    high_speed_duty_factor: float,
    swing_height: float,
    yaw_equivalent_radius: float,
) -> torch.Tensor:
    schedule = get_commanded_trot_schedule(
        env,
        command_name,
        command_deadband,
        min_frequency_hz,
        max_frequency_hz,
        min_equivalent_speed,
        max_equivalent_speed,
        low_speed_duty_factor,
        high_speed_duty_factor,
        swing_height,
        yaw_equivalent_radius,
    )
    contact = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    desired_contact = 1.0 - soft_swing_weights(schedule, transition_fraction)
    moving = schedule.frequency_hz.gt(0.0).unsqueeze(1).to(torch.float32)
    return torch.sum(desired_contact * (1.0 - contact) * moving, dim=1)


def trot_extra_swing_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    transition_fraction: float,
    command_name: str,
    command_deadband: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    min_equivalent_speed: float,
    max_equivalent_speed: float,
    low_speed_duty_factor: float,
    high_speed_duty_factor: float,
    swing_height: float,
    yaw_equivalent_radius: float,
) -> torch.Tensor:
    schedule = get_commanded_trot_schedule(
        env,
        command_name,
        command_deadband,
        min_frequency_hz,
        max_frequency_hz,
        min_equivalent_speed,
        max_equivalent_speed,
        low_speed_duty_factor,
        high_speed_duty_factor,
        swing_height,
        yaw_equivalent_radius,
    )
    contact = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    swing = soft_swing_weights(schedule, transition_fraction)
    return torch.sum(swing * contact, dim=1)


def _relative_foot_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    desired_contact: torch.Tensor,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    stance_weight = desired_contact.to(torch.float32)
    stance_count = stance_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    stance_height = (foot_height * stance_weight).sum(dim=1, keepdim=True) / stance_count
    return foot_height - stance_height


def trot_swing_foot_height_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sigma: float,
    transition_fraction: float,
    command_name: str,
    command_deadband: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    min_equivalent_speed: float,
    max_equivalent_speed: float,
    low_speed_duty_factor: float,
    high_speed_duty_factor: float,
    swing_height: float,
    yaw_equivalent_radius: float,
) -> torch.Tensor:
    """Track a smooth relative lift trajectory; zero lift always scores zero."""

    schedule = get_commanded_trot_schedule(
        env,
        command_name,
        command_deadband,
        min_frequency_hz,
        max_frequency_hz,
        min_equivalent_speed,
        max_equivalent_speed,
        low_speed_duty_factor,
        high_speed_duty_factor,
        swing_height,
        yaw_equivalent_radius,
    )
    swing_weight = soft_swing_weights(schedule, transition_fraction)
    desired_height = schedule.swing_height[:, None] * smooth_swing_profile(schedule)
    relative_height = _relative_foot_height(env, asset_cfg, schedule.desired_contact)
    score = torch.exp(-torch.square(relative_height - desired_height) / max(float(sigma) ** 2, 1.0e-8))
    gate_denominator = torch.maximum(
        desired_height,
        torch.clamp(0.15 * schedule.swing_height[:, None], min=1.0e-6),
    )
    lift_gate = torch.clamp(relative_height / gate_denominator, min=0.0, max=1.0)
    denominator = swing_weight.sum(dim=1).clamp_min(1.0)
    return torch.sum(score * lift_gate * swing_weight, dim=1) / denominator


def trot_stance_feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    transition_fraction: float,
    max_value: float,
    command_name: str,
    command_deadband: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    min_equivalent_speed: float,
    max_equivalent_speed: float,
    low_speed_duty_factor: float,
    high_speed_duty_factor: float,
    swing_height: float,
    yaw_equivalent_radius: float,
) -> torch.Tensor:
    schedule = get_commanded_trot_schedule(
        env,
        command_name,
        command_deadband,
        min_frequency_hz,
        max_frequency_hz,
        min_equivalent_speed,
        max_equivalent_speed,
        low_speed_duty_factor,
        high_speed_duty_factor,
        swing_height,
        yaw_equivalent_radius,
    )
    contact = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    desired_contact = 1.0 - soft_swing_weights(schedule, transition_fraction)
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide = torch.linalg.vector_norm(velocity_xy, dim=-1) * contact * desired_contact
    return torch.clamp(torch.sum(slide, dim=1), max=float(max_value))


def moving_touchdown_impact_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    soft_force_limit: float,
    max_normalized_excess: float,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].to(torch.float32)
    force = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :],
        dim=-1,
    ).amax(dim=1)
    excess = torch.clamp(
        torch.relu(force - float(soft_force_limit)) / max(float(soft_force_limit), 1.0e-6),
        max=float(max_normalized_excess),
    )
    moving = _moving_mask(env, command_name, command_deadband).unsqueeze(1).to(torch.float32)
    return torch.sum(torch.square(excess) * first_contact * moving, dim=1)


def commanded_straight_lateral_yaw_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    straight = (
        (torch.abs(command[:, 0]) >= float(command_deadband))
        & (torch.abs(command[:, 2]) < float(command_deadband))
    ).to(torch.float32)
    asset: Articulation = env.scene[asset_cfg.name]
    return straight * (
        torch.square(asset.data.root_lin_vel_b[:, 1]) + torch.square(asset.data.root_ang_vel_b[:, 2])
    )


def commanded_yaw_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    moving = _moving_mask(env, command_name, command_deadband).to(torch.float32)
    asset: Articulation = env.scene[asset_cfg.name]
    return moving * torch.square(asset.data.root_ang_vel_b[:, 2] - command[:, 2])


def base_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


class action_second_difference_l2(ManagerTermBase):
    """Penalize action curvature without filtering the policy action."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_previous_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids: torch.Tensor):
        self._previous_previous_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        current = env.action_manager.action
        previous = env.action_manager.prev_action
        second_difference = current - 2.0 * previous + self._previous_previous_action
        self._previous_previous_action.copy_(previous)
        return torch.sum(torch.square(second_difference), dim=1)


def stand_still_base_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    stopped = ~_moving_mask(env, command_name, command_deadband)
    asset: Articulation = env.scene[asset_cfg.name]
    return stopped.to(torch.float32) * torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def stand_still_joint_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    stopped = ~_moving_mask(env, command_name, command_deadband)
    asset: Articulation = env.scene[asset_cfg.name]
    return stopped.to(torch.float32) * torch.mean(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=1,
    )


def stand_still_joint_pose_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
    sigma: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    stopped = ~_moving_mask(env, command_name, command_deadband)
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    score = torch.exp(-torch.mean(torch.square(error), dim=1) / max(float(sigma) ** 2, 1.0e-8))
    return stopped.to(torch.float32) * score
