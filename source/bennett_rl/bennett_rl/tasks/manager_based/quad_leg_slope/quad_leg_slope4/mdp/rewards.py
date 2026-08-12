"""Task-local rewards and slope-coordinate terms for standalone Slope4."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

from .gait import get_commanded_trot_schedule, smooth_swing_profile, soft_swing_weights

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _moving_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :3], dim=1) >= float(command_deadband)


def _foot_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=1) > float(threshold)


class action_second_difference_l2(ManagerTermBase):
    """Penalize action curvature without filtering policy actions."""

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


def uphill_velocity_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed world-+X progress; standing scores zero, backward is negative."""

    command = env.command_manager.get_command(command_name)
    target_speed = torch.clamp(torch.abs(command[:, 0]), min=0.05)
    asset: Articulation = env.scene[asset_cfg.name]
    normalized = asset.data.root_lin_vel_w[:, 0] / target_speed
    return torch.clamp(normalized, min=-1.0, max=1.0)


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
        torch.square(asset.data.root_lin_vel_b[:, 1])
        + torch.square(asset.data.root_ang_vel_b[:, 2])
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


def moving_touchdown_impact_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    soft_force_limit: float,
    max_normalized_excess: float,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].to(
        torch.float32
    )
    force = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :],
        dim=-1,
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
    """Reward the scheduled diagonal contact pattern with soft transitions."""

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
    """Penalize a foot that is missing during its scheduled stance."""

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
    """Penalize a foot that remains planted during scheduled swing."""

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
    slope_angles: tuple[float, ...] | None = None,
    terrain_size_x: float = 0.0,
    approach_length: float = 0.0,
    top_platform_length: float = 0.0,
    spawn_x: float = 0.0,
    transition_length: float = 0.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_height = foot_pos[:, :, 2]

    if slope_angles is not None:
        # World-Z rise on a ramp is not foot clearance: simply moving a foot
        # uphill would otherwise look like lifting it.  Remove the exact local
        # terrain profile first, then subtract stance-foot clearance so the
        # foot collision radius cancels out.
        terrain = env.scene.terrain
        angle_table = torch.as_tensor(
            slope_angles, device=foot_height.device, dtype=foot_height.dtype
        )
        slope_angle = angle_table[terrain.terrain_levels]
        local_x = foot_pos[:, :, 0] - env.scene.env_origins[:, 0, None] + float(spawn_x)
        ground_height = _directional_slope_height(
            local_x,
            slope_angle[:, None],
            terrain_size_x=float(terrain_size_x),
            approach_length=float(approach_length),
            top_platform_length=float(top_platform_length),
            transition_length=float(transition_length),
        )
        foot_height = foot_height - (
            env.scene.env_origins[:, 2, None] + ground_height
        )

    stance_weight = desired_contact.to(torch.float32)
    stance_count = stance_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    stance_height = (
        (foot_height * stance_weight).sum(dim=1, keepdim=True) / stance_count
    )
    return foot_height - stance_height


def _directional_slope_height(
    local_x: torch.Tensor,
    slope_angle: torch.Tensor,
    terrain_size_x: float,
    approach_length: float,
    top_platform_length: float,
    transition_length: float,
) -> torch.Tensor:
    """Evaluate the same smooth ramp profile used by ``slope_terrain.py``."""

    ramp_start = float(approach_length)
    ramp_end = float(terrain_size_x) - float(top_platform_length)
    ramp_length = ramp_end - ramp_start
    transition = min(float(transition_length), 0.25 * ramp_length)
    if ramp_length <= 0.5 or transition <= 0.0:
        raise ValueError("Invalid directional-slope geometry for foot clearance.")

    top_height = torch.tan(slope_angle) * ramp_length
    effective_slope = top_height / max(ramp_length - transition, 1.0e-6)
    bottom_end = ramp_start + transition
    top_start = ramp_end - transition
    bottom_height = 0.5 * effective_slope * transition

    height = torch.zeros_like(local_x)
    bottom_u = torch.clamp((local_x - ramp_start) / transition, 0.0, 1.0)
    bottom_profile = effective_slope * transition * (
        bottom_u**3 - 0.5 * bottom_u**4
    )
    height = torch.where(
        (local_x >= ramp_start) & (local_x < bottom_end),
        bottom_profile,
        height,
    )

    middle_profile = bottom_height + effective_slope * (local_x - bottom_end)
    height = torch.where(
        (local_x >= bottom_end) & (local_x < top_start),
        middle_profile,
        height,
    )

    top_u = torch.clamp((local_x - top_start) / transition, 0.0, 1.0)
    middle_height = bottom_height + effective_slope * max(
        top_start - bottom_end, 0.0
    )
    top_profile = middle_height + effective_slope * transition * (
        top_u - top_u**3 + 0.5 * top_u**4
    )
    height = torch.where(
        (local_x >= top_start) & (local_x < ramp_end),
        top_profile,
        height,
    )
    height = torch.where(local_x >= ramp_end, top_height, height)
    return height


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
    slope_angles: tuple[float, ...] | None = None,
    terrain_size_x: float = 0.0,
    approach_length: float = 0.0,
    top_platform_length: float = 0.0,
    spawn_x: float = 0.0,
    transition_length: float = 0.0,
) -> torch.Tensor:
    """Track true clearance above flat ground or the local slope profile."""

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
    relative_height = _relative_foot_height(
        env,
        asset_cfg,
        schedule.desired_contact,
        slope_angles=slope_angles,
        terrain_size_x=terrain_size_x,
        approach_length=approach_length,
        top_platform_length=top_platform_length,
        spawn_x=spawn_x,
        transition_length=transition_length,
    )
    score = torch.exp(
        -torch.square(relative_height - desired_height)
        / max(float(sigma) ** 2, 1.0e-8)
    )
    gate_denominator = torch.maximum(
        desired_height,
        torch.clamp(0.15 * schedule.swing_height[:, None], min=1.0e-6),
    )
    lift_gate = torch.clamp(
        relative_height / gate_denominator,
        min=0.0,
        max=1.0,
    )
    denominator = swing_weight.sum(dim=1).clamp_min(1.0)
    return torch.sum(score * lift_gate * swing_weight, dim=1) / denominator


def trot_worst_swing_foot_height_shortfall_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height_fraction: float,
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
    slope_angles: tuple[float, ...] | None = None,
    terrain_size_x: float = 0.0,
    approach_length: float = 0.0,
    top_platform_length: float = 0.0,
    spawn_x: float = 0.0,
    transition_length: float = 0.0,
) -> torch.Tensor:
    """Penalize the worst under-lifted foot in each scheduled diagonal swing.

    Unlike the mean tracking reward, this term cannot be optimized by lifting
    one foot correctly while sacrificing its diagonal partner.
    """

    if not 0.0 < float(minimum_height_fraction) <= 1.0:
        raise ValueError("minimum_height_fraction must be in (0, 1].")

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
    minimum_height = (
        float(minimum_height_fraction)
        * schedule.swing_height[:, None]
        * smooth_swing_profile(schedule)
    )
    relative_height = _relative_foot_height(
        env,
        asset_cfg,
        schedule.desired_contact,
        slope_angles=slope_angles,
        terrain_size_x=terrain_size_x,
        approach_length=approach_length,
        top_platform_length=top_platform_length,
        spawn_x=spawn_x,
        transition_length=transition_length,
    )
    normalized_shortfall = torch.relu(minimum_height - relative_height) / torch.clamp(
        schedule.swing_height[:, None],
        min=1.0e-6,
    )
    per_foot_penalty = torch.square(normalized_shortfall) * swing_weight
    return torch.amax(per_foot_penalty, dim=1)


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
    """Penalize sliding only while the schedule asks a foot to support."""

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
    desired_contact = 1.0 - soft_swing_weights(
        schedule,
        transition_fraction,
    )
    asset: Articulation = env.scene[asset_cfg.name]
    velocity_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide = (
        torch.linalg.vector_norm(velocity_xy, dim=-1)
        * contact
        * desired_contact
    )
    return torch.clamp(torch.sum(slide, dim=1), max=float(max_value))


def feet_lateral_boundary_excess_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_outward_y: float,
    penalty_band: float,
    max_normalized_excess: float,
) -> torch.Tensor:
    """Apply one strong outward soft boundary to all four body-frame feet.

    The admissible region has exactly zero cost. Beyond the common boundary,
    the normalized quadratic cost grows rapidly but remains bounded for PPO
    numerical stability.
    """

    if len(asset_cfg.body_ids) != 4:
        raise ValueError("Expected ordered FL, FR, RL, RR foot bodies.")
    if float(max_outward_y) <= 0.0:
        raise ValueError("max_outward_y must be positive.")
    if float(penalty_band) <= 0.0:
        raise ValueError("penalty_band must be positive.")
    if float(max_normalized_excess) <= 0.0:
        raise ValueError("max_normalized_excess must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    relative_pos_w = foot_pos_w - asset.data.root_pos_w[:, None, :]
    num_envs, num_feet = relative_pos_w.shape[:2]
    root_quat_w = (
        asset.data.root_quat_w[:, None, :]
        .expand(-1, num_feet, -1)
        .reshape(-1, 4)
    )
    foot_pos_b = quat_apply_inverse(
        root_quat_w,
        relative_pos_w.reshape(-1, 3),
    ).reshape(num_envs, num_feet, 3)

    # Ordered feet are FL, FR, RL, RR: left is +Y and right is -Y.
    outward_sign = foot_pos_b.new_tensor((1.0, -1.0, 1.0, -1.0))
    outward_y = foot_pos_b[:, :, 1] * outward_sign
    normalized_excess = torch.clamp(
        torch.relu(outward_y - float(max_outward_y)) / float(penalty_band),
        max=float(max_normalized_excess),
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
    """Keep at least one diagonal pair available during uphill locomotion."""

    contact_count = _foot_contacts(env, sensor_cfg, threshold).sum(dim=1).to(
        torch.float32
    )
    shortfall = torch.relu(float(minimum_contacts) - contact_count)
    moving = _moving_mask(env, command_name, command_deadband).to(torch.float32)
    return moving * torch.square(shortfall)


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
    clearance = _base_clearance_above_feet(env, asset_cfg)
    return torch.square(clearance - float(target_height))


def base_clearance_above_feet_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_clearance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return _base_clearance_above_feet(env, asset_cfg) < float(minimum_clearance)
