"""Batch-validated directional slope curriculum for standalone Slope4."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class validated_top_platform_level(ManagerTermBase):
    """Advance one global slope level after a statistically useful pass batch.

    Every reset environment is assigned to the same current level.  An episode
    counts as a success only when the dedicated termination term reports that
    all four feet reached and held the upper platform.  A failed validation
    batch repeats the same level; difficulty never falls and never advances
    from average travel distance.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._current_level = 0
        self._validated_pass_level = -1
        self._batch_trials = 0
        self._batch_successes = 0
        self._reported_trials = 0
        self._reported_successes = 0
        self._reported_success_rate = 0.0

    def reset(self, env_ids: Sequence[int] | None = None):
        # CurriculumManager.reset() is called after every episode reset.
        # Validation counters intentionally persist across those calls.
        pass

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        success_term_name: str,
        max_level: int,
        validation_episodes: int,
        required_success_rate: float,
    ) -> dict[str, float]:
        if validation_episodes <= 0:
            raise ValueError("validation_episodes must be positive.")
        if not 0.0 < required_success_rate <= 1.0:
            raise ValueError("required_success_rate must be in (0, 1].")

        terrain: TerrainImporter = env.scene.terrain
        if terrain.terrain_origins is None:
            raise RuntimeError("validated_top_platform_level requires generated terrain origins.")

        ids = self._as_index_tensor(env_ids, env.num_envs, env.device)
        if ids.numel() == 0:
            return self._state()

        # Curriculum compute runs before managers and episode buffers are reset,
        # so these values still describe the just-completed episode.
        completed = env.episode_length_buf[ids] > 0
        previous_level = terrain.terrain_levels[ids]
        eligible = completed & (previous_level == self._current_level)
        if torch.any(eligible):
            success = env.termination_manager.get_term(success_term_name)[ids]
            self._batch_trials += int(torch.count_nonzero(eligible).item())
            self._batch_successes += int(torch.count_nonzero(success & eligible).item())

        if self._batch_trials >= int(validation_episodes):
            success_rate = self._batch_successes / max(self._batch_trials, 1)
            self._reported_trials = self._batch_trials
            self._reported_successes = self._batch_successes
            self._reported_success_rate = success_rate

            if success_rate >= float(required_success_rate):
                self._validated_pass_level = self._current_level
                self._current_level = min(self._current_level + 1, int(max_level))

            # Each validation decision uses a fresh episode batch.  Failed
            # levels are therefore retried instead of being poisoned forever
            # by early exploration episodes.
            self._batch_trials = 0
            self._batch_successes = 0
        elif self._batch_trials > 0:
            self._reported_trials = self._batch_trials
            self._reported_successes = self._batch_successes
            self._reported_success_rate = self._batch_successes / self._batch_trials

        # Assign only resetting environments.  Environments still running an
        # older level finish that episode before joining the new global level.
        terrain.terrain_levels[ids] = self._current_level
        terrain.env_origins[ids] = terrain.terrain_origins[
            self._current_level, terrain.terrain_types[ids]
        ]
        return self._state()

    def _state(self) -> dict[str, float]:
        return {
            "current_training_level": float(self._current_level),
            "validated_pass_level": float(self._validated_pass_level),
            "validation_success_rate": float(self._reported_success_rate),
            "validation_successes": float(self._reported_successes),
            "validation_trials": float(self._reported_trials),
        }

    @staticmethod
    def _as_index_tensor(
        env_ids: Sequence[int] | slice,
        num_envs: int,
        device: str,
    ) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=device, dtype=torch.long)
