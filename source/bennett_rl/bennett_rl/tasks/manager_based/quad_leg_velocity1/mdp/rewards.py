"""Custom MDP terms for Bennett quadruped velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward swing feet for clearing the ground on flat terrain.

    This term complements ``feet_air_time``: air-time encourages a foot to leave contact,
    while this term rewards the feet that are currently out of contact for reaching a
    useful height. It is disabled for near-zero velocity commands.
    """

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > contact_threshold
    )
    swing_feet = ~contacts
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    clearance = torch.clamp(foot_height / max(target_height, 1.0e-6), min=0.0, max=1.0)

    swing_count = torch.clamp(torch.sum(swing_feet.float(), dim=1), min=1.0)
    reward = torch.sum(clearance * swing_feet.float(), dim=1) / swing_count
    moving = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    return reward * moving.float()
