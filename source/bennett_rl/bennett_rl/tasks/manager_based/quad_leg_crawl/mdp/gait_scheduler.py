"""Deterministic gait schedules for Bennett crawl locomotion.

The scheduler is kept independent from Isaac Lab so it can be unit-tested
before being connected to observations, rewards, or actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LEG_NAMES = ("FL", "FR", "RL", "RR")

# Swing starts for leg order [FL, FR, RL, RR].  This gives the sequence
# FL -> RR -> FR -> RL when phase advances from 0 to 1.
CRAWL_SWING_START_OFFSETS = (0.0, 0.5, 0.75, 0.25)


@dataclass(frozen=True)
class GaitSchedule:
    """Batch gait schedule tensors.

    Shapes:
        global_phase: ``[num_envs]`` in ``[0, 1)``.
        leg_phase: ``[num_envs, 4]`` phase since each leg's swing start.
        desired_contact: ``[num_envs, 4]`` boolean stance target.
        desired_swing: ``[num_envs, 4]`` boolean swing target.
    """

    global_phase: torch.Tensor
    leg_phase: torch.Tensor
    desired_contact: torch.Tensor
    desired_swing: torch.Tensor

    @property
    def stance_count(self) -> torch.Tensor:
        """Number of desired stance legs per environment."""

        return self.desired_contact.to(torch.int64).sum(dim=1)

    @property
    def swing_count(self) -> torch.Tensor:
        """Number of desired swing legs per environment."""

        return self.desired_swing.to(torch.int64).sum(dim=1)


def _as_1d_tensor(value: float | torch.Tensor, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Convert scalar or tensor input to a one-dimensional float tensor."""

    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
    else:
        tensor = torch.tensor([value], dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 1:
        raise ValueError(f"Expected a scalar or 1-D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def _broadcast_like(value: float | torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    """Broadcast scalar or per-env tensor to match ``reference`` shape."""

    tensor = _as_1d_tensor(value, device=reference.device)
    if tensor.numel() == 1:
        return tensor.expand_as(reference)
    if tensor.shape != reference.shape:
        raise ValueError(f"{name} must be scalar or shape {tuple(reference.shape)}, got {tuple(tensor.shape)}.")
    return tensor


def advance_phase(
    global_phase: float | torch.Tensor,
    dt: float | torch.Tensor,
    frequency_hz: float | torch.Tensor,
) -> torch.Tensor:
    """Advance global phase by ``dt * frequency_hz`` and wrap to ``[0, 1)``."""

    phase = _as_1d_tensor(global_phase)
    dt_tensor = _broadcast_like(dt, phase, "dt")
    frequency = _broadcast_like(frequency_hz, phase, "frequency_hz")
    return torch.remainder(phase + dt_tensor * frequency, 1.0)


def compute_stand_schedule(global_phase: float | torch.Tensor) -> GaitSchedule:
    """Return an all-stance schedule for standing."""

    phase = _as_1d_tensor(global_phase)
    leg_phase = torch.zeros((phase.shape[0], len(LEG_NAMES)), dtype=torch.float32, device=phase.device)
    desired_contact = torch.ones_like(leg_phase, dtype=torch.bool)
    desired_swing = torch.zeros_like(leg_phase, dtype=torch.bool)
    return GaitSchedule(phase, leg_phase, desired_contact, desired_swing)


def compute_crawl_schedule(
    global_phase: float | torch.Tensor,
    duty_factor: float | torch.Tensor = 0.85,
    swing_start_offsets: tuple[float, float, float, float] | torch.Tensor = CRAWL_SWING_START_OFFSETS,
) -> GaitSchedule:
    """Return desired contacts for a static crawl gait.

    ``duty_factor`` is the fraction of a full cycle each leg should spend in
    stance.  The remaining ``1 - duty_factor`` interval is the leg's swing
    window, starting at the corresponding ``swing_start_offsets`` value.
    """

    phase = _as_1d_tensor(global_phase)
    duty = _broadcast_like(duty_factor, phase, "duty_factor")
    if torch.any((duty < 0.75) | (duty >= 1.0)):
        raise ValueError("crawl duty_factor must be in [0.75, 1.0) to guarantee at most one swing leg.")

    offsets = torch.as_tensor(swing_start_offsets, dtype=torch.float32, device=phase.device)
    if offsets.shape != (len(LEG_NAMES),):
        raise ValueError(f"swing_start_offsets must have shape ({len(LEG_NAMES)},), got {tuple(offsets.shape)}.")

    leg_phase = torch.remainder(phase[:, None] - offsets[None, :], 1.0)
    swing_fraction = (1.0 - duty).clamp_min(0.0)
    desired_swing = leg_phase < swing_fraction[:, None]
    desired_contact = torch.logical_not(desired_swing)
    return GaitSchedule(phase, leg_phase, desired_contact, desired_swing)


def render_contact_schedule(schedule: GaitSchedule) -> str:
    """Render a compact text table for a batch schedule.

    Contact legs are marked ``C`` and swing legs are marked ``S``.
    """

    lines = ["phase    FL FR RL RR    stance swing"]
    for env_id in range(schedule.global_phase.shape[0]):
        states = ["C" if bool(v) else "S" for v in schedule.desired_contact[env_id].tolist()]
        lines.append(
            f"{schedule.global_phase[env_id].item():.3f}    "
            f"{states[0]}  {states[1]}  {states[2]}  {states[3]}       "
            f"{int(schedule.stance_count[env_id].item())}      {int(schedule.swing_count[env_id].item())}"
        )
    return "\n".join(lines)

