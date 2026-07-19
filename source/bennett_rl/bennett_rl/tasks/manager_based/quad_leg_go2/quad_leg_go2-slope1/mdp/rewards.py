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
    return torch.linalg.vector_norm(command[:, :3], dim=1) >= command_deadband


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


# ----------------------------------------------------------------------
# Slope-specific reward and termination functions
# ----------------------------------------------------------------------

def _slope_angle_by_env(env: ManagerBasedRLEnv, slope_angles: tuple[float, ...]) -> torch.Tensor:
    """Return the slope angle for each environment based on its terrain column."""
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        return torch.zeros(env.num_envs, device=env.device)
    col_idx = terrain.terrain_types[: env.num_envs]
    angles = torch.tensor(slope_angles, device=env.device)
    return angles[col_idx]


def _directional_slope_profile(
    env: ManagerBasedRLEnv,
    x_position_w: torch.Tensor,
    slope_angles: tuple[float, ...],
    approach_length: float,
    top_platform_length: float,
    spawn_x: float,
    transition_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return terrain height and local pitch for the smooth ramp profile."""
    terrain = env.scene.terrain
    total_length = float(terrain.cfg.terrain_generator.size[0])
    ramp_start_x = approach_length
    ramp_end_x = total_length - top_platform_length
    ramp_length = ramp_end_x - ramp_start_x
    transition = min(transition_length, ramp_length * 0.25)
    slope_angle = _slope_angle_by_env(env, slope_angles)
    env_origin_x = env.scene.env_origins[:, 0]
    env_origin_z = env.scene.env_origins[:, 2]
    while slope_angle.ndim < x_position_w.ndim:
        slope_angle = slope_angle.unsqueeze(-1)
        env_origin_x = env_origin_x.unsqueeze(-1)
        env_origin_z = env_origin_z.unsqueeze(-1)
    local_x = x_position_w - env_origin_x + spawn_x
    top_height = torch.tan(slope_angle) * ramp_length
    effective_slope = top_height / max(ramp_length - transition, 1.0e-6)
    bottom_end_x = ramp_start_x + transition
    top_start_x = ramp_end_x - transition
    bottom_height = 0.5 * effective_slope * transition
    middle_height = bottom_height + effective_slope * max(top_start_x - bottom_end_x, 0.0)
    height = torch.zeros_like(local_x)
    gradient = torch.zeros_like(local_x)
    bottom_mask = (local_x >= ramp_start_x) & (local_x < bottom_end_x)
    u_bottom = torch.clamp((local_x - ramp_start_x) / max(transition, 1.0e-6), 0.0, 1.0)
    smooth_bottom = 3.0 * u_bottom**2 - 2.0 * u_bottom**3
    integral_bottom = u_bottom**3 - 0.5 * u_bottom**4
    height = torch.where(bottom_mask, effective_slope * transition * integral_bottom, height)
    gradient = torch.where(bottom_mask, effective_slope * smooth_bottom, gradient)
    middle_mask = (local_x >= bottom_end_x) & (local_x < top_start_x)
    height = torch.where(middle_mask, bottom_height + effective_slope * (local_x - bottom_end_x), height)
    gradient = torch.where(middle_mask, effective_slope, gradient)
    top_mask = (local_x >= top_start_x) & (local_x < ramp_end_x)
    u_top = torch.clamp((local_x - top_start_x) / max(transition, 1.0e-6), 0.0, 1.0)
    smooth_top = 3.0 * u_top**2 - 2.0 * u_top**3
    integral_top = u_top**3 - 0.5 * u_top**4
    height = torch.where(top_mask, middle_height + effective_slope * transition * integral_top, height)
    gradient = torch.where(top_mask, effective_slope - effective_slope * smooth_top, gradient)
    flat_mask = local_x >= ramp_end_x
    height = torch.where(flat_mask, top_height, height)
    return height, gradient


def slope_aligned_orientation_l2(
    env: ManagerBasedRLEnv,
    slope_angles: tuple[float, ...],
    approach_length: float,
    top_platform_length: float,
    spawn_x: float,
    transition_length: float,
    slope_follow_ratio: float = 0.85,
    roll_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise base roll/pitch that deviate from the local slope pitch."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_pos = asset.data.root_pos_w[:, :2]
    _, gradient = _directional_slope_profile(
        env, base_pos[:, 0], slope_angles, approach_length, top_platform_length, spawn_x, transition_length
    )
    target_pitch = torch.atan(gradient)
    projected_gravity = asset.data.projected_gravity_b
    actual_pitch = torch.atan2(projected_gravity[:, 0], projected_gravity[:, 2])
    actual_roll = torch.atan2(projected_gravity[:, 1], projected_gravity[:, 2])
    pitch_error = actual_pitch - slope_follow_ratio * target_pitch
    roll_error = actual_roll
    return torch.square(pitch_error) + roll_weight * torch.square(roll_error)


def base_height_above_feet_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise deviation of base height above average foot z-position."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    avg_foot_z = foot_pos.mean(dim=1)
    base_z = asset.data.root_pos_w[:, 2]
    height_above_feet = base_z - avg_foot_z
    return torch.square(height_above_feet - target_height)


def base_clearance_above_feet_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_clearance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when base-to-feet clearance drops below threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    avg_foot_z = foot_pos.mean(dim=1)
    base_z = asset.data.root_pos_w[:, 2]
    clearance = base_z - avg_foot_z
    return clearance < minimum_clearance


def crawl_slope_swing_foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    frequency_hz: float,
    duty_factor: float,
    target_height: float,
    slope_angles: tuple[float, ...],
    approach_length: float,
    top_platform_length: float,
    spawn_x: float,
    transition_length: float,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Reward swing feet reaching target height above the local slope surface."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    terrain_height, _ = _directional_slope_profile(
        env, foot_pos[:, :, 0], slope_angles, approach_length, top_platform_length, spawn_x, transition_length
    )
    clearance = torch.clamp(foot_pos[:, :, 2] - terrain_height, min=0.0)
    desired_swing = _commanded_schedule(env, frequency_hz, duty_factor, command_name, command_deadband).desired_swing
    foot_error = torch.where(desired_swing, clearance - target_height, torch.zeros_like(clearance))
    return torch.exp(-torch.mean(torch.square(foot_error), dim=1) / 0.01)


def track_straight_line_y_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
    max_lateral_error: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise base deviation from the world +x centre-line."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_origin_y = env.scene.env_origins[:, 1]
    base_y = asset.data.root_pos_w[:, 1]
    lateral_error = base_y - env_origin_y
    return torch.clamp(torch.square(lateral_error) / max(max_lateral_error**2, 1.0e-6), max=1.0)

