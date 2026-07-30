"""Pure-Torch velocity-command sampling for Bennett Go2-11.

The deployment keyboard produces a small set of meaningful command modes
(straight, reverse, in-place yaw, and simultaneous translation/yaw).  Sampling
the three command axes independently gives very little probability to those
exact modes, so Go2-11 samples them explicitly while retaining continuous
command magnitudes inside each mode.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


COMMAND_MODE_NAMES = (
    "stand",
    "forward",
    "backward",
    "yaw_in_place",
    "forward_yaw",
    "backward_yaw",
    "lateral",
    "lateral_yaw",
)


def _validate_signed_range(name: str, value_range: tuple[float, float], minimum_abs: float) -> float:
    if value_range[0] > 0.0 or value_range[1] < 0.0:
        raise ValueError(f"{name} must span zero, got {value_range}.")
    maximum_abs = max(abs(float(value_range[0])), abs(float(value_range[1])))
    if not 0.0 <= minimum_abs <= maximum_abs:
        raise ValueError(f"{name} minimum_abs must be in [0, {maximum_abs}], got {minimum_abs}.")
    return maximum_abs


def sample_balanced_velocity_commands(
    num_commands: int,
    *,
    device: torch.device | str,
    mode_probabilities: Sequence[float],
    lin_vel_x_range: tuple[float, float],
    lin_vel_y_range: tuple[float, float],
    ang_vel_z_range: tuple[float, float],
    min_abs_lin_vel_x: float,
    min_abs_lin_vel_y: float,
    min_abs_ang_vel_z: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample keyboard-aligned commands and return ``(commands, mode_ids)``.

    The output command order is ``[vx, vy, yaw_rate]``.  Forward and backward
    have separate modes so their coverage can be tuned independently.
    """

    if num_commands < 0:
        raise ValueError(f"num_commands must be non-negative, got {num_commands}.")
    if len(mode_probabilities) != len(COMMAND_MODE_NAMES):
        raise ValueError(
            f"Expected {len(COMMAND_MODE_NAMES)} mode probabilities, got {len(mode_probabilities)}."
        )

    weights = torch.as_tensor(mode_probabilities, dtype=torch.float32, device=device)
    if not torch.isfinite(weights).all() or torch.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("mode_probabilities must be finite, non-negative, and have a positive sum.")
    weights = weights / weights.sum()

    max_x = _validate_signed_range("lin_vel_x_range", lin_vel_x_range, min_abs_lin_vel_x)
    max_y = _validate_signed_range("lin_vel_y_range", lin_vel_y_range, min_abs_lin_vel_y)
    max_yaw = _validate_signed_range("ang_vel_z_range", ang_vel_z_range, min_abs_ang_vel_z)

    mode_ids = torch.multinomial(weights, num_commands, replacement=True, generator=generator)
    commands = torch.zeros((num_commands, 3), dtype=torch.float32, device=device)

    def magnitude(minimum: float, maximum: float) -> torch.Tensor:
        return minimum + (maximum - minimum) * torch.rand(num_commands, device=device, generator=generator)

    def sign() -> torch.Tensor:
        return torch.where(
            torch.rand(num_commands, device=device, generator=generator) < 0.5,
            -torch.ones(num_commands, device=device),
            torch.ones(num_commands, device=device),
        )

    x_magnitude = magnitude(min_abs_lin_vel_x, max_x)
    y_magnitude = magnitude(min_abs_lin_vel_y, max_y)
    yaw_magnitude = magnitude(min_abs_ang_vel_z, max_yaw)
    yaw_signed = yaw_magnitude * sign()

    forward = mode_ids == 1
    backward = mode_ids == 2
    yaw_in_place = mode_ids == 3
    forward_yaw = mode_ids == 4
    backward_yaw = mode_ids == 5
    lateral = mode_ids == 6
    lateral_yaw = mode_ids == 7

    commands[forward | forward_yaw, 0] = x_magnitude[forward | forward_yaw]
    commands[backward | backward_yaw, 0] = -x_magnitude[backward | backward_yaw]
    commands[lateral | lateral_yaw, 1] = (y_magnitude * sign())[lateral | lateral_yaw]
    commands[yaw_in_place | forward_yaw | backward_yaw | lateral_yaw, 2] = yaw_signed[
        yaw_in_place | forward_yaw | backward_yaw | lateral_yaw
    ]
    return commands, mode_ids
