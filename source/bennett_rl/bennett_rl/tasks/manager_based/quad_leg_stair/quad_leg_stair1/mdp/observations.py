"""Simulation-only privileged observations for Bennett Stair1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return four binary foot-contact states for the privileged critic."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return (torch.linalg.vector_norm(forces, dim=-1).amax(dim=1) > float(threshold)).to(torch.float32)


def normalized_terrain_level(env: ManagerBasedRLEnv, max_level: int) -> torch.Tensor:
    """Expose normalized curriculum level to the critic, never to the actor."""

    terrain: TerrainImporter = env.scene.terrain
    denominator = max(int(max_level), 1)
    return (terrain.terrain_levels.to(torch.float32) / float(denominator)).unsqueeze(1)

