"""Validated 1-to-10 cm stair-height curriculum."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class validated_stair_level(ManagerTermBase):
    """Validate the current height while retaining the preceding skill."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._current_level = 0
        self._validated_pass_level = -1
        self._batch_trials = 0
        self._batch_successes = 0
        self._reported_trials = 0
        self._reported_successes = 0
        self._reported_success_rate = 0.0
        self._reported_level = -1
        self._consecutive_pass_batches = 0

    def reset(self, env_ids: Sequence[int] | None = None):
        # Batch evidence must persist across individual episode resets.
        pass

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_term_name: str,
        command_name: str,
        command_deadband: float,
        max_level: int,
        validation_episodes: int,
        required_success_rate: float,
        required_consecutive_pass_batches: int,
        previous_level_replay_fraction: float,
    ) -> dict[str, float]:
        if validation_episodes <= 0:
            raise ValueError("validation_episodes must be positive.")
        if not 0.0 < required_success_rate <= 1.0:
            raise ValueError("required_success_rate must be in (0, 1].")
        if required_consecutive_pass_batches <= 0:
            raise ValueError("required_consecutive_pass_batches must be positive.")
        if not 0.0 <= previous_level_replay_fraction < 1.0:
            raise ValueError("previous_level_replay_fraction must be in [0, 1).")

        terrain: TerrainImporter = env.scene.terrain
        if terrain.terrain_origins is None:
            raise RuntimeError("validated_stair_level requires generated terrain origins.")

        ids = self._as_index_tensor(env_ids, env.num_envs, env.device)
        if ids.numel() == 0:
            return self._state()

        command = env.command_manager.get_command(command_name)[ids]
        moving = command[:, 0] >= float(command_deadband)
        completed = env.episode_length_buf[ids] > 0
        previous_level = terrain.terrain_levels[ids]
        eligible = completed & moving & (previous_level == self._current_level)
        if torch.any(eligible):
            success = env.termination_manager.get_term(success_term_name)[ids]
            self._batch_trials += int(torch.count_nonzero(eligible).item())
            self._batch_successes += int(torch.count_nonzero(success & eligible).item())

        if self._batch_trials >= int(validation_episodes):
            success_rate = self._batch_successes / max(self._batch_trials, 1)
            self._reported_trials = self._batch_trials
            self._reported_successes = self._batch_successes
            self._reported_success_rate = success_rate
            self._reported_level = self._current_level
            if success_rate >= float(required_success_rate):
                self._consecutive_pass_batches += 1
            else:
                self._consecutive_pass_batches = 0
            if self._consecutive_pass_batches >= int(required_consecutive_pass_batches):
                self._validated_pass_level = self._current_level
                self._current_level = min(self._current_level + 1, int(max_level))
                self._consecutive_pass_batches = 0
            self._batch_trials = 0
            self._batch_successes = 0
        # Publish only completed validation batches.  Reporting the partial
        # accumulator on every asynchronous reset makes TensorBoard display a
        # noisy, non-comparable ratio and previously obscured the true gate.

        assigned_levels = torch.full_like(previous_level, self._current_level)
        if self._current_level > 0 and float(previous_level_replay_fraction) > 0.0:
            revisit_previous = torch.rand(ids.numel(), device=env.device) < float(
                previous_level_replay_fraction
            )
            assigned_levels = torch.where(
                revisit_previous,
                torch.full_like(assigned_levels, self._current_level - 1),
                assigned_levels,
            )

        terrain.terrain_levels[ids] = assigned_levels
        terrain.env_origins[ids] = terrain.terrain_origins[
            assigned_levels, terrain.terrain_types[ids]
        ]
        return self._state()

    def _state(self) -> dict[str, float]:
        return {
            "current_training_level": float(self._current_level),
            "current_step_height_cm": float(self._current_level + 1),
            "validated_pass_level": float(self._validated_pass_level),
            "validation_batch_level": float(self._reported_level),
            "validation_success_rate": float(self._reported_success_rate),
            "validation_successes": float(self._reported_successes),
            "validation_trials": float(self._reported_trials),
            "consecutive_pass_batches": float(self._consecutive_pass_batches),
        }

    @staticmethod
    def _as_index_tensor(
        env_ids: Sequence[int] | slice, num_envs: int, device: str
    ) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=device, dtype=torch.long)
