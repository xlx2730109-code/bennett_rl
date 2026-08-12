"""Terrain-level curriculum for Gravel1."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gravel_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    linear_reward_name: str = "track_lin_vel_xy_exp",
    yaw_reward_name: str = "track_ang_vel_z_exp",
    command_deadband: float = 0.025,
    promote_linear_score: float = 0.72,
    promote_yaw_score: float = 0.65,
    demote_linear_score: float = 0.45,
    demote_yaw_score: float = 0.40,
) -> dict[str, torch.Tensor]:
    """Advance terrain from full-episode tracking quality, not net distance.

    The stock distance rule is unsuitable here: two low-speed body-frame
    commands can cancel their world displacement even when both are tracked
    perfectly. Reward sums integrate tracking over the complete episode and
    remain valid for forward, reverse, lateral, yaw, and combined commands.
    """

    terrain: TerrainImporter = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    zero = torch.zeros((), device=env.device)
    if ids.numel() == 0:
        return {
            "mean_level": torch.mean(terrain.terrain_levels.float()),
            "promotion_rate": zero,
            "demotion_rate": zero,
            "linear_score": zero,
            "yaw_score": zero,
        }

    completed_episode = env.episode_length_buf[ids] > 0
    eval_ids = ids[completed_episode]
    promotion_rate = zero
    demotion_rate = zero
    mean_linear_score = zero
    mean_yaw_score = zero
    if eval_ids.numel() > 0:
        reward_sums = env.reward_manager._episode_sums
        if linear_reward_name not in reward_sums or yaw_reward_name not in reward_sums:
            raise KeyError(
                "Gravel curriculum requires active linear and yaw tracking rewards."
            )

        linear_cfg = env.reward_manager.get_term_cfg(linear_reward_name)
        yaw_cfg = env.reward_manager.get_term_cfg(yaw_reward_name)
        if linear_cfg.weight <= 0.0 or yaw_cfg.weight <= 0.0:
            raise ValueError("Tracking reward weights must be positive.")

        elapsed_s = (
            env.episode_length_buf[eval_ids].to(torch.float32)
            * float(env.step_dt)
        ).clamp_min(float(env.step_dt))
        linear_score = torch.clamp(
            reward_sums[linear_reward_name][eval_ids]
            / (float(linear_cfg.weight) * elapsed_s),
            min=0.0,
            max=1.0,
        )
        yaw_score = torch.clamp(
            reward_sums[yaw_reward_name][eval_ids]
            / (float(yaw_cfg.weight) * elapsed_s),
            min=0.0,
            max=1.0,
        )

        command = env.command_manager.get_command(command_name)[eval_ids]
        moving = torch.linalg.vector_norm(command[:, :3], dim=1) >= float(
            command_deadband
        )
        survived = env.reset_time_outs[eval_ids]
        move_up = (
            survived
            & moving
            & (linear_score >= float(promote_linear_score))
            & (yaw_score >= float(promote_yaw_score))
        )
        move_down = (~survived) | (
            moving
            & (
                (linear_score < float(demote_linear_score))
                | (yaw_score < float(demote_yaw_score))
            )
        )
        move_down &= ~move_up
        terrain.update_env_origins(eval_ids, move_up, move_down)

        promotion_rate = move_up.to(torch.float32).mean()
        demotion_rate = move_down.to(torch.float32).mean()
        mean_linear_score = linear_score.mean()
        mean_yaw_score = yaw_score.mean()

    return {
        "mean_level": torch.mean(terrain.terrain_levels.float()),
        "promotion_rate": promotion_rate,
        "demotion_rate": demotion_rate,
        "linear_score": mean_linear_score,
        "yaw_score": mean_yaw_score,
    }
