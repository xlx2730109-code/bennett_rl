from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _phase(env: ManagerBasedRLEnv, cycle_time: float | torch.Tensor) -> torch.Tensor:
    return torch.remainder(env.episode_length_buf.float() * env.step_dt / cycle_time, 1.0)


def _desired_trot_stance(env: ManagerBasedRLEnv, cycle_time: float, duty_factor: float) -> torch.Tensor:
    phase = _phase(env, cycle_time)
    stance_a = phase < duty_factor
    stance_b = torch.remainder(phase + 0.5, 1.0) < duty_factor
    return torch.stack((stance_a, stance_b, stance_b, stance_a), dim=1)


def gait_phase_sin_cos(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    phase = _phase(env, cycle_time)
    return torch.stack((torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)), dim=1)


def trot_contact_phase(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    cycle_time: float,
    duty_factor: float,
    contact_force_threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    in_contact = contact_force > contact_force_threshold
    desired_stance = _desired_trot_stance(env, cycle_time, duty_factor)

    contact_match = torch.where(desired_stance, in_contact, ~in_contact)
    reward = contact_match.float().mean(dim=1)

    command = env.command_manager.get_command(command_name)
    reward *= (torch.norm(command[:, :2], dim=1) > 0.1).float()
    return reward


def swing_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    cycle_time: float,
    duty_factor: float,
    target_height: float,
    contact_force_threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    in_contact = contact_force > contact_force_threshold
    desired_swing = ~_desired_trot_stance(env, cycle_time, duty_factor)

    asset = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    clearance = torch.clamp(foot_height / target_height, min=0.0, max=1.0)
    valid_swing = desired_swing & ~in_contact
    reward = torch.sum(clearance * valid_swing.float(), dim=1) / torch.clamp(
        torch.sum(desired_swing.float(), dim=1), min=1.0
    )

    command = env.command_manager.get_command(command_name)
    reward *= (torch.norm(command[:, :2], dim=1) > 0.1).float()
    return reward


def _foot_x_trajectory_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    desired_swing: torch.Tensor,
    swing_progress: torch.Tensor,
    stride_length: float | torch.Tensor,
    target_x_by_foot: tuple[float, float, float, float],
    min_command_speed: float = 0.1,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    rel_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, None, :]
    yaw = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, rel_pos_w.shape[1], -1)
    rel_pos_b = quat_apply_inverse(yaw.reshape(-1, 4), rel_pos_w.reshape(-1, 3)).reshape_as(rel_pos_w)

    nominal_x = torch.tensor(target_x_by_foot, device=rel_pos_b.device, dtype=rel_pos_b.dtype)
    stride = torch.as_tensor(stride_length, device=rel_pos_b.device, dtype=rel_pos_b.dtype)
    if stride.ndim == 0:
        target_x = nominal_x + stride * (swing_progress - 0.5)
    else:
        target_x = nominal_x[None, :] + stride[:, None] * (swing_progress - 0.5)

    x_error = torch.square(rel_pos_b[:, :, 0] - target_x)
    penalty = torch.sum(x_error * desired_swing.float(), dim=1) / torch.clamp(
        torch.sum(desired_swing.float(), dim=1), min=1.0
    )

    command = env.command_manager.get_command(command_name)
    penalty *= (torch.norm(command[:, :2], dim=1) > min_command_speed).float()
    return penalty


def trot_swing_foot_x_trajectory_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cycle_time: float,
    duty_factor: float,
    stride_length: float,
    target_x_by_foot: tuple[float, float, float, float],
) -> torch.Tensor:
    phase = _phase(env, cycle_time)
    swing_fraction = 1.0 - duty_factor
    local_phase = torch.stack(
        (
            phase,
            torch.remainder(phase + 0.5, 1.0),
            torch.remainder(phase + 0.5, 1.0),
            phase,
        ),
        dim=1,
    )
    desired_swing = local_phase >= duty_factor
    swing_progress = torch.clamp((local_phase - duty_factor) / max(swing_fraction, 1.0e-6), min=0.0, max=1.0)
    return _foot_x_trajectory_penalty(
        env, command_name, asset_cfg, desired_swing, swing_progress, stride_length, target_x_by_foot
    )


def foot_lateral_position_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_abs_y: float,
    target_y_by_foot: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    rel_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, None, :]
    yaw = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, rel_pos_w.shape[1], -1)
    rel_pos_b = quat_apply_inverse(yaw.reshape(-1, 4), rel_pos_w.reshape(-1, 3)).reshape_as(rel_pos_w)

    if target_y_by_foot is None:
        target_y_by_foot = (target_abs_y, -target_abs_y, target_abs_y, -target_abs_y)
    target_y = torch.tensor(target_y_by_foot, device=rel_pos_b.device, dtype=rel_pos_b.dtype)
    return torch.sum(torch.square(rel_pos_b[:, :, 1] - target_y), dim=1)


def track_straight_line_y_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_lateral_error: float = 0.45,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lateral_error = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    normalized_error = lateral_error / max(max_lateral_error, 1.0e-6)
    penalty = torch.square(normalized_error)

    command = env.command_manager.get_command(command_name)
    penalty *= (torch.norm(command[:, :2], dim=1) > 0.1).float()
    return penalty

