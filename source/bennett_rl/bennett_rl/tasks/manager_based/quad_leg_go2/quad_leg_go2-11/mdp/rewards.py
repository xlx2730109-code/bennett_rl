"""Reward terms for Bennett crawl locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .gait_scheduler import (
    compute_crawl_schedule,
    compute_stand_schedule,
    smooth_swing_profile,
    soft_swing_weights,
)
from .observations import commanded_crawl_global_phase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _crawl_schedule(env: ManagerBasedRLEnv, frequency_hz: float, duty_factor: float):
    return compute_crawl_schedule(
        commanded_crawl_global_phase(env, frequency_hz=frequency_hz),
        duty_factor=duty_factor,
    )


def _moving_mask(env: ManagerBasedRLEnv, command_name: str, command_deadband: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :3], dim=1) >= command_deadband


def _commanded_schedule(
    env: ManagerBasedRLEnv,
    frequency_hz: float,
    duty_factor: float,
    command_name: str,
    command_deadband: float,
):
    phase = commanded_crawl_global_phase(
        env,
        frequency_hz=frequency_hz,
        command_name=command_name,
        command_deadband=command_deadband,
    )
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


def commanded_straight_lateral_yaw_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float = 0.025,
    straight_command_deadband: float = 0.02,
    yaw_command_deadband: float = 0.03,
    yaw_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize sideways drift and yaw only for near-straight movement commands."""

    command = env.command_manager.get_command(command_name)
    moving = torch.abs(command[:, 0]) >= command_deadband
    straight_command = (torch.abs(command[:, 1]) <= straight_command_deadband) & (
        torch.abs(command[:, 2]) <= yaw_command_deadband
    )
    mask = (moving & straight_command).to(torch.float32)

    asset: Articulation = env.scene[asset_cfg.name]
    lateral = torch.square(asset.data.root_lin_vel_b[:, 1])
    yaw = torch.square(asset.data.root_ang_vel_b[:, 2])
    return mask * (lateral + yaw_weight * yaw)


def commanded_lateral_yaw_vel_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float = 0.025,
    yaw_weight: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral and yaw velocity error for all moving velocity commands."""

    command = env.command_manager.get_command(command_name)
    moving = torch.linalg.vector_norm(command[:, :3], dim=1) >= command_deadband

    asset: Articulation = env.scene[asset_cfg.name]
    lateral_error = asset.data.root_lin_vel_b[:, 1] - command[:, 1]
    yaw_error = asset.data.root_ang_vel_b[:, 2] - command[:, 2]
    return moving.to(torch.float32) * (torch.square(lateral_error) + yaw_weight * torch.square(yaw_error))


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
    transition_fraction: float = 0.04,
) -> torch.Tensor:
    """Reward contact agreement using smooth swing/stance transitions."""
    contacts = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    schedule = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband)
    desired_swing = soft_swing_weights(schedule, duty_factor, transition_fraction)
    desired_contacts = 1.0 - desired_swing
    return 1.0 - torch.mean(torch.abs(contacts - desired_contacts), dim=1)


def _soft_contact_weights(
    env: ManagerBasedRLEnv,
    frequency_hz: float,
    duty_factor: float,
    command_name: str,
    command_deadband: float,
    transition_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    schedule = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband)
    desired_swing = soft_swing_weights(schedule, duty_factor, transition_fraction)
    moving = _moving_mask(env, command_name, command_deadband).unsqueeze(1).to(torch.float32)
    desired_swing = desired_swing * moving
    desired_contact = 1.0 - desired_swing
    return desired_contact, desired_swing


def crawl_missing_stance_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
    transition_fraction: float = 0.04,
) -> torch.Tensor:
    """Penalize missing stance contacts with soft phase-boundary weights."""
    contacts = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    desired_contacts, _ = _soft_contact_weights(
        env,
        frequency_hz,
        duty_factor,
        command_name,
        command_deadband,
        transition_fraction,
    )
    return torch.sum(desired_contacts * (1.0 - contacts), dim=1)


def crawl_extra_swing_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    threshold: float = 1.0,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
    transition_fraction: float = 0.04,
) -> torch.Tensor:
    """Penalize swing contacts without a discontinuity at lift-off/touchdown."""
    contacts = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    _, desired_swing = _soft_contact_weights(
        env,
        frequency_hz,
        duty_factor,
        command_name,
        command_deadband,
        transition_fraction,
    )
    return torch.sum(desired_swing * contacts, dim=1)


def crawl_swing_touchdown_events(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    foot_index: int,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Penalize a new touchdown while one scheduled swing leg should stay airborne."""

    if not 0 <= foot_index < 4:
        raise ValueError(f"foot_index must be in [0, 3], got {foot_index}.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    desired_swing = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband).desired_swing
    return (first_contact[:, foot_index] & desired_swing[:, foot_index]).to(torch.float32)


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
    transition_fraction: float = 0.04,
) -> torch.Tensor:
    """Penalize stance-foot sliding with soft contact-transition weights."""
    contacts = _foot_contacts(env, sensor_cfg, threshold).to(torch.float32)
    desired_contacts, _ = _soft_contact_weights(
        env,
        frequency_hz,
        duty_factor,
        command_name,
        command_deadband,
        transition_fraction,
    )
    stance_weight = contacts * desired_contacts

    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide = torch.linalg.norm(foot_vel_xy, dim=-1) * stance_weight
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


def _relative_foot_height_and_vertical_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    desired_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure each foot relative to the scheduled stance-foot support plane."""

    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_vertical_velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    stance_weight = desired_contact.to(torch.float32)
    stance_count = stance_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    stance_height = (foot_height * stance_weight).sum(dim=1, keepdim=True) / stance_count
    stance_vertical_velocity = (foot_vertical_velocity * stance_weight).sum(dim=1, keepdim=True) / stance_count
    return foot_height - stance_height, foot_vertical_velocity - stance_vertical_velocity


def crawl_swing_foot_height_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    target_height: float,
    sigma: float = 0.012,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Track a smooth zero-slope lift-and-lower trajectory during swing.

    The lift gate makes zero physical lift score exactly zero, preventing the
    no-lift exploit that a plain Gaussian trajectory reward can admit.
    """

    schedule = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband)
    profile, _ = smooth_swing_profile(schedule, duty_factor)
    relative_height, _ = _relative_foot_height_and_vertical_velocity(env, asset_cfg, schedule.desired_contact)
    desired_height = float(target_height) * profile
    error_score = torch.exp(-torch.square(relative_height - desired_height) / max(float(sigma) ** 2, 1.0e-8))
    lift_gate_denominator = desired_height.clamp_min(max(0.15 * float(target_height), 1.0e-6))
    lift_gate = torch.clamp(relative_height / lift_gate_denominator, min=0.0, max=1.0)
    swing_mask = schedule.desired_swing.to(torch.float32)
    swing_count = swing_mask.sum(dim=1).clamp_min(1.0)
    return torch.sum(error_score * lift_gate * swing_mask, dim=1) / swing_count


def crawl_swing_foot_vertical_velocity_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    target_height: float,
    sigma: float = 0.18,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Track the derivative of the smooth foot-height trajectory."""

    schedule = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband)
    _, profile_derivative = smooth_swing_profile(schedule, duty_factor)
    _, relative_vertical_velocity = _relative_foot_height_and_vertical_velocity(
        env,
        asset_cfg,
        schedule.desired_contact,
    )
    desired_vertical_velocity = float(target_height) * float(frequency_hz) * profile_derivative
    velocity_score = torch.exp(
        -torch.square(relative_vertical_velocity - desired_vertical_velocity) / max(float(sigma) ** 2, 1.0e-8)
    )
    peak_velocity = max(
        float(target_height) * float(frequency_hz) * torch.pi / max(1.0 - float(duty_factor), 1.0e-6),
        1.0e-6,
    )
    motion_weight = 0.25 + 0.75 * torch.clamp(
        torch.abs(desired_vertical_velocity) / peak_velocity,
        min=0.0,
        max=1.0,
    )
    swing_mask = schedule.desired_swing.to(torch.float32)
    swing_count = swing_mask.sum(dim=1).clamp_min(1.0)
    return torch.sum(velocity_score * motion_weight * swing_mask, dim=1) / swing_count


def crawl_touchdown_impact_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    soft_force_limit: float = 40.0,
    max_normalized_excess: float = 3.0,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Penalize excessive contact force only on new moving-mode touchdowns."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].to(torch.float32)
    # The peak often occurs inside one of the 5 ms physics substeps and can be
    # gone by the 20 ms policy sample.  Use the contact-sensor history window.
    force = torch.linalg.norm(
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :],
        dim=-1,
    ).amax(dim=1)
    normalized_excess = torch.clamp(
        torch.relu(force - float(soft_force_limit)) / max(float(soft_force_limit), 1.0e-6),
        max=float(max_normalized_excess),
    )
    moving = _moving_mask(env, command_name, command_deadband).unsqueeze(1).to(torch.float32)
    return torch.sum(torch.square(normalized_excess) * first_contact * moving, dim=1)


class action_second_difference_l2(ManagerTermBase):
    """Penalize action curvature ``a[t] - 2 a[t-1] + a[t-2]``."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_previous_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids: torch.Tensor):
        self._previous_previous_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        current_action = env.action_manager.action
        previous_action = env.action_manager.prev_action
        second_difference = current_action - 2.0 * previous_action + self._previous_previous_action
        self._previous_previous_action.copy_(previous_action)
        return torch.sum(torch.square(second_difference), dim=1)


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

