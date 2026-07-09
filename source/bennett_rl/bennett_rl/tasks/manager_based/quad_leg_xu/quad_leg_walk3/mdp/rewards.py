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


def _desired_walk_stance(
    env: ManagerBasedRLEnv,
    cycle_time: float,
    duty_factor: float,
    swing_offsets: tuple[float, float, float, float],
) -> torch.Tensor:
    phase = _phase(env, cycle_time)
    swing_fraction = 1.0 - duty_factor
    offsets = torch.tensor(swing_offsets, device=phase.device, dtype=phase.dtype)
    local_phase = torch.remainder(phase[:, None] - offsets[None, :], 1.0)
    desired_swing = local_phase < swing_fraction
    return ~desired_swing


def gait_phase_sin_cos(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    phase = _phase(env, cycle_time)
    return torch.stack((torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)), dim=1)


def walk_contact_phase(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    cycle_time: float,
    duty_factor: float,
    swing_offsets: tuple[float, float, float, float],
    contact_force_threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    in_contact = contact_force > contact_force_threshold
    desired_stance = _desired_walk_stance(env, cycle_time, duty_factor, swing_offsets)

    contact_match = torch.where(desired_stance, in_contact, ~in_contact)
    reward = contact_match.float().mean(dim=1)

    command = env.command_manager.get_command(command_name)
    reward *= (torch.norm(command[:, :2], dim=1) > 0.05).float()
    return reward


def walk_swing_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    cycle_time: float,
    duty_factor: float,
    swing_offsets: tuple[float, float, float, float],
    target_height: float,
    contact_force_threshold: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    in_contact = contact_force > contact_force_threshold
    desired_swing = ~_desired_walk_stance(env, cycle_time, duty_factor, swing_offsets)

    asset = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    clearance = torch.clamp(foot_height / target_height, min=0.0, max=1.0)
    valid_swing = desired_swing & ~in_contact
    reward = torch.sum(clearance * valid_swing.float(), dim=1) / torch.clamp(
        torch.sum(desired_swing.float(), dim=1), min=1.0
    )

    command = env.command_manager.get_command(command_name)
    reward *= (torch.norm(command[:, :2], dim=1) > 0.05).float()
    return reward


def feet_air_time_target_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_air_time: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    air_time_error = torch.square(last_air_time - target_air_time)
    penalty = torch.sum(air_time_error * first_contact, dim=1) / torch.clamp(
        torch.sum(first_contact, dim=1), min=1.0
    )

    command = env.command_manager.get_command(command_name)
    penalty *= (torch.norm(command[:, :2], dim=1) > 0.05).float()
    return penalty


def pitch_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base pitch (forward/backward tilt) using L1 norm.

    L1 gives a stronger signal than L2 for moderate pitch,
    which helps keep the body level during slow walking.
    """
    asset = env.scene[asset_cfg.name]
    return torch.abs(asset.data.projected_gravity_b[:, 0])


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
