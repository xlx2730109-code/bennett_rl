"""Observation helpers for Bennett crawl locomotion."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .gait_scheduler import compute_crawl_schedule, compute_stand_schedule

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def crawl_global_phase(env: ManagerBasedRLEnv, frequency_hz: float = 0.5) -> torch.Tensor:
    """Global crawl phase in cycles, shaped ``[num_envs]``."""

    elapsed_s = env.episode_length_buf.to(torch.float32) * env.step_dt
    return torch.remainder(elapsed_s * frequency_hz, 1.0)


def _moving_mask(env: ManagerBasedRLEnv, command_name: str, command_deadband: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.abs(command[:, 0]) >= command_deadband


def commanded_crawl_global_phase(
    env: ManagerBasedRLEnv,
    frequency_hz: float = 0.5,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Global phase that is frozen at zero while the velocity command is stopped."""

    phase = crawl_global_phase(env, frequency_hz)
    return torch.where(_moving_mask(env, command_name, command_deadband), phase, torch.zeros_like(phase))


def crawl_global_phase_sin_cos(env: ManagerBasedRLEnv, frequency_hz: float = 0.5) -> torch.Tensor:
    """Global crawl phase as ``[sin, cos]``."""

    phase_rad = 2.0 * math.pi * crawl_global_phase(env, frequency_hz)
    return torch.stack((torch.sin(phase_rad), torch.cos(phase_rad)), dim=-1)


def commanded_crawl_global_phase_sin_cos(
    env: ManagerBasedRLEnv,
    frequency_hz: float = 0.5,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Command-gated global phase. Stopped commands always observe ``[0, 1]``."""

    phase_rad = 2.0 * math.pi * commanded_crawl_global_phase(env, frequency_hz, command_name, command_deadband)
    return torch.stack((torch.sin(phase_rad), torch.cos(phase_rad)), dim=-1)


def crawl_leg_phase_sin_cos(
    env: ManagerBasedRLEnv,
    frequency_hz: float = 0.5,
    duty_factor: float = 0.85,
) -> torch.Tensor:
    """Per-leg crawl phase as FL/FR/RL/RR ``sin, cos`` pairs."""

    schedule = compute_crawl_schedule(crawl_global_phase(env, frequency_hz), duty_factor=duty_factor)
    phase_rad = 2.0 * math.pi * schedule.leg_phase
    return torch.stack((torch.sin(phase_rad), torch.cos(phase_rad)), dim=-1).reshape(env.num_envs, -1)


def commanded_crawl_leg_phase_sin_cos(
    env: ManagerBasedRLEnv,
    frequency_hz: float = 0.5,
    duty_factor: float = 0.85,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Per-leg phase with all phases held at zero for stand-still commands."""

    phase = commanded_crawl_global_phase(env, frequency_hz, command_name, command_deadband)
    crawl_schedule = compute_crawl_schedule(phase, duty_factor=duty_factor)
    stand_schedule = compute_stand_schedule(phase)
    moving = _moving_mask(env, command_name, command_deadband).unsqueeze(1)
    leg_phase = torch.where(moving, crawl_schedule.leg_phase, stand_schedule.leg_phase)
    phase_rad = 2.0 * math.pi * leg_phase
    return torch.stack((torch.sin(phase_rad), torch.cos(phase_rad)), dim=-1).reshape(env.num_envs, -1)


def crawl_desired_contacts(
    env: ManagerBasedRLEnv,
    frequency_hz: float = 0.5,
    duty_factor: float = 0.85,
) -> torch.Tensor:
    """Desired FL/FR/RL/RR contact flags for crawl, returned as floats."""

    schedule = compute_crawl_schedule(crawl_global_phase(env, frequency_hz), duty_factor=duty_factor)
    return schedule.desired_contact.to(torch.float32)


def commanded_crawl_desired_contacts(
    env: ManagerBasedRLEnv,
    frequency_hz: float = 0.5,
    duty_factor: float = 0.85,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Desired contacts: all four feet in stance while stopped, crawl schedule while moving."""

    phase = commanded_crawl_global_phase(env, frequency_hz, command_name, command_deadband)
    crawl_schedule = compute_crawl_schedule(phase, duty_factor=duty_factor)
    stand_schedule = compute_stand_schedule(phase)
    moving = _moving_mask(env, command_name, command_deadband).unsqueeze(1)
    desired_contact = torch.where(moving, crawl_schedule.desired_contact, stand_schedule.desired_contact)
    return desired_contact.to(torch.float32)


def crawl_gait_params(
    env: ManagerBasedRLEnv,
    frequency_hz: float = 0.5,
    duty_factor: float = 0.85,
    swing_height: float = 0.025,
) -> torch.Tensor:
    """Current gait parameters ``[frequency_hz, duty_factor, swing_height]``."""

    values = torch.tensor((frequency_hz, duty_factor, swing_height), dtype=torch.float32, device=env.device)
    return values.unsqueeze(0).repeat(env.num_envs, 1)


def commanded_crawl_gait_params(
    env: ManagerBasedRLEnv,
    frequency_hz: float = 0.5,
    duty_factor: float = 0.85,
    swing_height: float = 0.025,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Gait parameters with a distinct stopped mode ``[0, 1, 0]``."""

    moving_values = torch.tensor((frequency_hz, duty_factor, swing_height), dtype=torch.float32, device=env.device)
    stopped_values = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=env.device)
    moving = _moving_mask(env, command_name, command_deadband).unsqueeze(1)
    return torch.where(
        moving,
        moving_values.unsqueeze(0).expand(env.num_envs, -1),
        stopped_values.unsqueeze(0).expand(env.num_envs, -1),
    )


def fixed_velocity_command(
    env: ManagerBasedRLEnv,
    lin_vel_x: float = 0.10,
    lin_vel_y: float = 0.0,
    ang_vel_z: float = 0.0,
) -> torch.Tensor:
    """Fixed Stage-1 crawl command observation ``[vx, vy, yaw_rate]``."""

    command = torch.tensor((lin_vel_x, lin_vel_y, ang_vel_z), dtype=torch.float32, device=env.device)
    return command.unsqueeze(0).repeat(env.num_envs, 1)
