"""Balanced command term for Bennett diagonal-trot training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .command_sampling import COMMAND_MODE_NAMES, sample_balanced_velocity_commands

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BalancedTrotVelocityCommand(UniformVelocityCommand):
    """Sample the exact motion modes used by deployment controls."""

    cfg: "BalancedTrotVelocityCommandCfg"

    def __init__(self, cfg: "BalancedTrotVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._diagnostic_error_sum = torch.zeros((self.num_envs, 4), device=self.device)
        self._diagnostic_step_count = torch.zeros((self.num_envs, 4), device=self.device)

    def __str__(self) -> str:
        probabilities = ", ".join(
            f"{name}={probability:.3f}"
            for name, probability in zip(COMMAND_MODE_NAMES, self.cfg.mode_probabilities)
        )
        return super().__str__() + f"\n\tBalanced trot modes: {probabilities}"

    def _resample_command(self, env_ids: Sequence[int]):
        commands, mode_ids = sample_balanced_velocity_commands(
            len(env_ids),
            device=self.device,
            mode_probabilities=self.cfg.mode_probabilities,
            lin_vel_x_range=self.cfg.ranges.lin_vel_x,
            ang_vel_z_range=self.cfg.ranges.ang_vel_z,
            min_abs_lin_vel_x=self.cfg.min_abs_lin_vel_x,
            min_abs_ang_vel_z=self.cfg.min_abs_ang_vel_z,
        )
        self.vel_command_b[env_ids] = commands
        self.command_mode[env_ids] = mode_ids
        self.is_standing_env[env_ids] = mode_ids == 0
        self.is_heading_env[env_ids] = False

    def _update_metrics(self):
        super()._update_metrics()
        linear_x_error = torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        yaw_error = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        masks = (
            (self.command_mode == 2) | (self.command_mode == 5),
            self.command_mode == 3,
            (self.command_mode == 4) | (self.command_mode == 5),
            (self.command_mode == 4) | (self.command_mode == 5),
        )
        errors = (linear_x_error, yaw_error, linear_x_error, yaw_error)
        for index, (mask, error) in enumerate(zip(masks, errors)):
            mask_float = mask.to(torch.float32)
            self._diagnostic_error_sum[:, index] += error * mask_float
            self._diagnostic_step_count[:, index] += mask_float

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        metric_env_ids = slice(None) if env_ids is None else env_ids
        labels = ("error_backward_x", "error_yaw_in_place", "error_combined_x", "error_combined_yaw")
        diagnostics = {}
        for index, label in enumerate(labels):
            count = self._diagnostic_step_count[metric_env_ids, index].sum()
            error_sum = self._diagnostic_error_sum[metric_env_ids, index].sum()
            diagnostics[label] = (error_sum / count.clamp_min(1.0)).item()
        self._diagnostic_error_sum[metric_env_ids] = 0.0
        self._diagnostic_step_count[metric_env_ids] = 0.0
        extras = super().reset(env_ids)
        extras.update(diagnostics)
        return extras


@configclass
class BalancedTrotVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = BalancedTrotVelocityCommand

    # stand, forward, backward, yaw, forward+yaw, backward+yaw
    mode_probabilities: tuple[float, ...] = (0.12, 0.22, 0.18, 0.16, 0.18, 0.14)
    min_abs_lin_vel_x: float = 0.08
    min_abs_ang_vel_z: float = 0.18
