"""Balanced deployment-command sampling for Gravel1."""

from __future__ import annotations

from collections.abc import Sequence

import torch


COMMAND_MODE_NAMES = (
    "stand",
    "forward",
    "backward",
    "lateral_left",
    "lateral_right",
    "yaw_in_place",
    "forward_yaw",
    "backward_yaw",
    "lateral_yaw",
)


def _validate_signed_range(
    name: str, value_range: tuple[float, float], minimum_abs: float
) -> float:
    if value_range[0] > 0.0 or value_range[1] < 0.0:
        raise ValueError(f"{name} must span zero, got {value_range}.")
    maximum_abs = max(abs(float(value_range[0])), abs(float(value_range[1])))
    if not 0.0 <= minimum_abs <= maximum_abs:
        raise ValueError(
            f"{name} minimum_abs must be in [0, {maximum_abs}], got {minimum_abs}."
        )
    return maximum_abs


def sample_balanced_omnidirectional_commands(
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
    """Sample explicit stand, translation, yaw, and combined command modes."""

    if num_commands < 0:
        raise ValueError(f"num_commands must be non-negative, got {num_commands}.")
    if len(mode_probabilities) != len(COMMAND_MODE_NAMES):
        raise ValueError(
            f"Expected {len(COMMAND_MODE_NAMES)} mode probabilities, "
            f"got {len(mode_probabilities)}."
        )

    weights = torch.as_tensor(mode_probabilities, dtype=torch.float32, device=device)
    if (
        not torch.isfinite(weights).all()
        or torch.any(weights < 0.0)
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError(
            "mode_probabilities must be finite, non-negative, and have a positive sum."
        )
    weights = weights / weights.sum()

    max_x = _validate_signed_range(
        "lin_vel_x_range", lin_vel_x_range, min_abs_lin_vel_x
    )
    max_y = _validate_signed_range(
        "lin_vel_y_range", lin_vel_y_range, min_abs_lin_vel_y
    )
    max_yaw = _validate_signed_range(
        "ang_vel_z_range", ang_vel_z_range, min_abs_ang_vel_z
    )
    mode_ids = torch.multinomial(
        weights, num_commands, replacement=True, generator=generator
    )
    commands = torch.zeros((num_commands, 3), dtype=torch.float32, device=device)

    def magnitude(minimum: float, maximum: float) -> torch.Tensor:
        return minimum + (maximum - minimum) * torch.rand(
            num_commands, device=device, generator=generator
        )

    def signed_magnitude(minimum: float, maximum: float) -> torch.Tensor:
        sign = torch.where(
            torch.rand(num_commands, device=device, generator=generator) < 0.5,
            -torch.ones(num_commands, device=device),
            torch.ones(num_commands, device=device),
        )
        return magnitude(minimum, maximum) * sign

    x_magnitude = magnitude(min_abs_lin_vel_x, max_x)
    y_magnitude = magnitude(min_abs_lin_vel_y, max_y)
    yaw_signed = signed_magnitude(min_abs_ang_vel_z, max_yaw)
    lateral_sign = torch.where(
        torch.rand(num_commands, device=device, generator=generator) < 0.5,
        -torch.ones(num_commands, device=device),
        torch.ones(num_commands, device=device),
    )

    forward = mode_ids == 1
    backward = mode_ids == 2
    lateral_left = mode_ids == 3
    lateral_right = mode_ids == 4
    yaw_in_place = mode_ids == 5
    forward_yaw = mode_ids == 6
    backward_yaw = mode_ids == 7
    lateral_yaw = mode_ids == 8

    commands[forward | forward_yaw, 0] = x_magnitude[forward | forward_yaw]
    commands[backward | backward_yaw, 0] = -x_magnitude[
        backward | backward_yaw
    ]
    commands[lateral_left, 1] = y_magnitude[lateral_left]
    commands[lateral_right, 1] = -y_magnitude[lateral_right]
    commands[lateral_yaw, 1] = (
        y_magnitude[lateral_yaw] * lateral_sign[lateral_yaw]
    )
    commands[yaw_in_place | forward_yaw | backward_yaw | lateral_yaw, 2] = (
        yaw_signed[yaw_in_place | forward_yaw | backward_yaw | lateral_yaw]
    )
    return commands, mode_ids
