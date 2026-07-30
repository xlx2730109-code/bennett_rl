"""Speed-conditioned diagonal-trot clock, observations, and pure scheduling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


LEG_NAMES = ("FL", "FR", "RL", "RR")
# Diagonal pair A: FL + RR.  Diagonal pair B: FR + RL.
TROT_PHASE_OFFSETS = (0.0, 0.5, 0.5, 0.0)
_CLOCK_ATTRIBUTE = "_quad_leg_trot1_clock"


@dataclass(frozen=True)
class TrotSchedule:
    global_phase: torch.Tensor
    leg_phase: torch.Tensor
    desired_contact: torch.Tensor
    desired_swing: torch.Tensor
    frequency_hz: torch.Tensor
    duty_factor: torch.Tensor
    swing_height: torch.Tensor


@dataclass
class _ClockState:
    phase: torch.Tensor
    last_step: torch.Tensor
    was_moving: torch.Tensor


def speed_conditioned_gait_parameters(
    command: torch.Tensor,
    *,
    command_deadband: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    min_equivalent_speed: float,
    max_equivalent_speed: float,
    low_speed_duty_factor: float,
    high_speed_duty_factor: float,
    swing_height: float,
    yaw_equivalent_radius: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map command magnitude continuously to gait frequency and stance duty factor."""

    if command.ndim != 2 or command.shape[1] < 3:
        raise ValueError(f"command must have shape [N, >=3], got {tuple(command.shape)}.")
    if not 0.5 <= high_speed_duty_factor <= low_speed_duty_factor < 1.0:
        raise ValueError("duty factors must satisfy 0.5 <= high <= low < 1.0.")
    if not 0.0 < min_frequency_hz <= max_frequency_hz:
        raise ValueError("frequency range must be positive and ordered.")
    if not 0.0 <= min_equivalent_speed < max_equivalent_speed:
        raise ValueError("equivalent-speed range must be non-negative and ordered.")

    moving = torch.linalg.vector_norm(command[:, :3], dim=1) >= float(command_deadband)
    equivalent_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    equivalent_speed = equivalent_speed + float(yaw_equivalent_radius) * torch.abs(command[:, 2])
    blend = torch.clamp(
        (equivalent_speed - float(min_equivalent_speed))
        / max(float(max_equivalent_speed - min_equivalent_speed), 1.0e-6),
        min=0.0,
        max=1.0,
    )
    frequency = float(min_frequency_hz) + blend * float(max_frequency_hz - min_frequency_hz)
    duty = float(low_speed_duty_factor) + blend * float(high_speed_duty_factor - low_speed_duty_factor)
    height = torch.full_like(frequency, float(swing_height))
    frequency = torch.where(moving, frequency, torch.zeros_like(frequency))
    duty = torch.where(moving, duty, torch.ones_like(duty))
    height = torch.where(moving, height, torch.zeros_like(height))
    return frequency, duty, height, moving


def compute_trot_schedule(
    global_phase: torch.Tensor,
    frequency_hz: torch.Tensor,
    duty_factor: torch.Tensor,
    swing_height: torch.Tensor,
    moving: torch.Tensor,
) -> TrotSchedule:
    """Build an alternating diagonal-pair contact schedule."""

    offsets = torch.as_tensor(TROT_PHASE_OFFSETS, device=global_phase.device, dtype=global_phase.dtype)
    leg_phase = torch.remainder(global_phase[:, None] - offsets[None, :], 1.0)
    swing_fraction = (1.0 - duty_factor).clamp(min=0.0, max=0.5)
    desired_swing = (leg_phase < swing_fraction[:, None]) & moving[:, None]
    desired_contact = ~desired_swing
    leg_phase = torch.where(moving[:, None], leg_phase, torch.zeros_like(leg_phase))
    return TrotSchedule(
        global_phase=torch.where(moving, global_phase, torch.zeros_like(global_phase)),
        leg_phase=leg_phase,
        desired_contact=desired_contact,
        desired_swing=desired_swing,
        frequency_hz=frequency_hz,
        duty_factor=duty_factor,
        swing_height=swing_height,
    )


def smooth_swing_profile(schedule: TrotSchedule) -> torch.Tensor:
    """Zero-slope foot-height profile over each scheduled swing."""

    swing_fraction = (1.0 - schedule.duty_factor).clamp_min(1.0e-6)
    progress = torch.clamp(schedule.leg_phase / swing_fraction[:, None], min=0.0, max=1.0)
    profile = torch.sin(torch.pi * progress).square()
    return profile * schedule.desired_swing.to(profile.dtype)


def soft_swing_weights(schedule: TrotSchedule, transition_fraction: float) -> torch.Tensor:
    """Smooth the lift-off and touchdown edges of the contact target."""

    swing_fraction = (1.0 - schedule.duty_factor).clamp_min(1.0e-6)
    blend = torch.minimum(
        torch.full_like(swing_fraction, float(transition_fraction)),
        0.45 * swing_fraction,
    ).clamp_min(1.0e-6)
    ramp_up = torch.clamp(schedule.leg_phase / blend[:, None], min=0.0, max=1.0)
    ramp_down = torch.clamp(
        (swing_fraction[:, None] - schedule.leg_phase) / blend[:, None],
        min=0.0,
        max=1.0,
    )
    return torch.minimum(ramp_up, ramp_down) * schedule.desired_swing.to(ramp_up.dtype)


def _schedule_from_env(
    env: ManagerBasedRLEnv,
    *,
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
) -> TrotSchedule:
    command = env.command_manager.get_command(command_name)
    frequency, duty, height, moving = speed_conditioned_gait_parameters(
        command,
        command_deadband=command_deadband,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        min_equivalent_speed=min_equivalent_speed,
        max_equivalent_speed=max_equivalent_speed,
        low_speed_duty_factor=low_speed_duty_factor,
        high_speed_duty_factor=high_speed_duty_factor,
        swing_height=swing_height,
        yaw_equivalent_radius=yaw_equivalent_radius,
    )

    current_step = env.episode_length_buf
    state = getattr(env, _CLOCK_ATTRIBUTE, None)
    if state is None or state.phase.shape != current_step.shape:
        state = _ClockState(
            phase=torch.zeros_like(current_step, dtype=torch.float32),
            last_step=current_step.clone(),
            was_moving=torch.zeros_like(moving),
        )
        setattr(env, _CLOCK_ATTRIBUTE, state)

    reset = current_step < state.last_step
    delta_steps = torch.clamp(current_step - state.last_step, min=0).to(torch.float32)
    state.phase = torch.remainder(state.phase + delta_steps * float(env.step_dt) * frequency, 1.0)
    state.phase = torch.where(reset, torch.zeros_like(state.phase), state.phase)
    state.was_moving = torch.where(reset, torch.zeros_like(state.was_moving), state.was_moving)

    # Start inside the double-stance interval instead of instantly unloading a
    # diagonal pair when a non-zero keyboard command arrives.
    just_started = moving & ~state.was_moving
    start_phase = (1.0 - duty).clamp(min=0.0, max=0.5)
    state.phase = torch.where(just_started, start_phase, state.phase)
    state.last_step.copy_(current_step)
    state.was_moving.copy_(moving)
    return compute_trot_schedule(state.phase, frequency, duty, height, moving)


def commanded_trot_global_phase_sin_cos(
    env: ManagerBasedRLEnv,
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
    schedule = _schedule_from_env(
        env,
        command_name=command_name,
        command_deadband=command_deadband,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        min_equivalent_speed=min_equivalent_speed,
        max_equivalent_speed=max_equivalent_speed,
        low_speed_duty_factor=low_speed_duty_factor,
        high_speed_duty_factor=high_speed_duty_factor,
        swing_height=swing_height,
        yaw_equivalent_radius=yaw_equivalent_radius,
    )
    angle = 2.0 * torch.pi * schedule.global_phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)


def commanded_trot_leg_phase_sin_cos(
    env: ManagerBasedRLEnv,
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
    schedule = _schedule_from_env(
        env,
        command_name=command_name,
        command_deadband=command_deadband,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        min_equivalent_speed=min_equivalent_speed,
        max_equivalent_speed=max_equivalent_speed,
        low_speed_duty_factor=low_speed_duty_factor,
        high_speed_duty_factor=high_speed_duty_factor,
        swing_height=swing_height,
        yaw_equivalent_radius=yaw_equivalent_radius,
    )
    angle = 2.0 * torch.pi * schedule.leg_phase
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)


def commanded_trot_desired_contacts(
    env: ManagerBasedRLEnv,
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
    return _schedule_from_env(
        env,
        command_name=command_name,
        command_deadband=command_deadband,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        min_equivalent_speed=min_equivalent_speed,
        max_equivalent_speed=max_equivalent_speed,
        low_speed_duty_factor=low_speed_duty_factor,
        high_speed_duty_factor=high_speed_duty_factor,
        swing_height=swing_height,
        yaw_equivalent_radius=yaw_equivalent_radius,
    ).desired_contact.to(torch.float32)


def commanded_trot_gait_params(
    env: ManagerBasedRLEnv,
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
    schedule = _schedule_from_env(
        env,
        command_name=command_name,
        command_deadband=command_deadband,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        min_equivalent_speed=min_equivalent_speed,
        max_equivalent_speed=max_equivalent_speed,
        low_speed_duty_factor=low_speed_duty_factor,
        high_speed_duty_factor=high_speed_duty_factor,
        swing_height=swing_height,
        yaw_equivalent_radius=yaw_equivalent_radius,
    )
    return torch.stack((schedule.frequency_hz, schedule.duty_factor, schedule.swing_height), dim=1)


def get_commanded_trot_schedule(
    env: ManagerBasedRLEnv,
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
) -> TrotSchedule:
    """Public reward-side access to the same cached clock used by observations."""

    return _schedule_from_env(
        env,
        command_name=command_name,
        command_deadband=command_deadband,
        min_frequency_hz=min_frequency_hz,
        max_frequency_hz=max_frequency_hz,
        min_equivalent_speed=min_equivalent_speed,
        max_equivalent_speed=max_equivalent_speed,
        low_speed_duty_factor=low_speed_duty_factor,
        high_speed_duty_factor=high_speed_duty_factor,
        swing_height=swing_height,
        yaw_equivalent_radius=yaw_equivalent_radius,
    )
