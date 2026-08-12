"""Velocity-command term with explicit real-deployment mode coverage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

from .command_sampling import (
    COMMAND_MODE_NAMES,
    sample_balanced_omnidirectional_commands,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BalancedOmnidirectionalVelocityCommand(UniformVelocityCommand):
    """Sample every hardware command mode without prescribing a gait."""

    cfg: "BalancedOmnidirectionalVelocityCommandCfg"

    def __init__(
        self,
        cfg: "BalancedOmnidirectionalVelocityCommandCfg",
        env: "ManagerBasedEnv",
    ):
        super().__init__(cfg, env)
        self.command_mode = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def __str__(self) -> str:
        probabilities = ", ".join(
            f"{name}={probability:.3f}"
            for name, probability in zip(
                COMMAND_MODE_NAMES, self.cfg.mode_probabilities
            )
        )
        return super().__str__() + f"\n\tBalanced command modes: {probabilities}"

    def _resample_command(self, env_ids: Sequence[int]):
        commands, mode_ids = sample_balanced_omnidirectional_commands(
            len(env_ids),
            device=self.device,
            mode_probabilities=self.cfg.mode_probabilities,
            lin_vel_x_range=self.cfg.ranges.lin_vel_x,
            lin_vel_y_range=self.cfg.ranges.lin_vel_y,
            ang_vel_z_range=self.cfg.ranges.ang_vel_z,
            min_abs_lin_vel_x=self.cfg.min_abs_lin_vel_x,
            min_abs_lin_vel_y=self.cfg.min_abs_lin_vel_y,
            min_abs_ang_vel_z=self.cfg.min_abs_ang_vel_z,
        )
        self.vel_command_b[env_ids] = commands
        self.command_mode[env_ids] = mode_ids
        self.is_standing_env[env_ids] = mode_ids == 0
        self.is_heading_env[env_ids] = False


@configclass
class BalancedOmnidirectionalVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = BalancedOmnidirectionalVelocityCommand

    # stand, forward, backward, left, right, yaw, forward+yaw,
    # backward+yaw, lateral+yaw.  Fore-aft motion is emphasized because the
    # 8-DoF Bennett has no hip-abduction actuator.
    mode_probabilities: tuple[float, ...] = (
        0.10,
        0.22,
        0.18,
        0.07,
        0.07,
        0.12,
        0.12,
        0.08,
        0.04,
    )
    min_abs_lin_vel_x: float = 0.08
    min_abs_lin_vel_y: float = 0.05
    min_abs_ang_vel_z: float = 0.15
