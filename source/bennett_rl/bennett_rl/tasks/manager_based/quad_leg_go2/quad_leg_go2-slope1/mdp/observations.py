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
    return torch.linalg.vector_norm(command[:, :3], dim=1) >= command_deadband


def _stateful_commanded_phase(
    env: ManagerBasedRLEnv,
    frequency_hz: float,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    """Run one gait clock per environment and reset it on a stopped-to-moving edge.

    Resetting to phase zero makes the scheduled swing foot start at zero
    clearance rather than jumping into the middle of its lift trajectory.
    The episode-length stamp prevents multiple observation/reward terms from
    advancing the shared clock more than once in one policy step.
    """

    cache_name = "_bennett_slope1_gait_clocks"
    clocks = getattr(env, cache_name, None)
    if clocks is None:
        clocks = {}
        setattr(env, cache_name, clocks)

    key = (float(frequency_hz), str(command_name), float(command_deadband))
    state = clocks.get(key)
    if state is None:
        state = {
            "phase": torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
            "was_moving": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "last_step": torch.full_like(env.episode_length_buf, -1),
        }
        clocks[key] = state

    phase = state["phase"]
    was_moving = state["was_moving"]
    last_step = state["last_step"]
    current_step = env.episode_length_buf
    needs_update = current_step != last_step
    if torch.any(needs_update):
        moving = _moving_mask(env, command_name, command_deadband)
        episode_reset = (current_step < last_step) | (current_step == 0)
        starting = needs_update & moving & (~was_moving | episode_reset)
        continuing = needs_update & moving & was_moving & ~episode_reset
        stopped = needs_update & ~moving

        phase[starting | stopped | (needs_update & episode_reset)] = 0.0
        phase[continuing] = torch.remainder(
            phase[continuing] + env.step_dt * float(frequency_hz),
            1.0,
        )
        was_moving[needs_update] = moving[needs_update]
        last_step[needs_update] = current_step[needs_update]
    return phase


def commanded_crawl_global_phase(
    env: ManagerBasedRLEnv,
    frequency_hz: float = 0.5,
    command_name: str = "base_velocity",
    command_deadband: float = 0.025,
) -> torch.Tensor:
    """Command-gated phase that starts from zero and stays continuous while moving."""

    return _stateful_commanded_phase(env, frequency_hz, command_name, command_deadband)


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
