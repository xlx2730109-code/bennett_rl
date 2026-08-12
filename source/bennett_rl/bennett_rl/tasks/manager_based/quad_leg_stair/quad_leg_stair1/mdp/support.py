"""Support-foot geometry helpers local to Bennett Stair1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def support_foot_height_w(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float,
    minimum_support_contacts: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean support-foot height and whether that estimate is reliable.

    Swing feet must not raise the terrain-height reference while crossing a
    riser.  A support estimate is used only when at least two configured feet
    are in contact; one-foot and flight phases are left to the base-contact and
    minimum-support terms instead of creating a false low-clearance failure.
    """

    if minimum_support_contacts <= 0:
        raise ValueError("minimum_support_contacts must be positive.")

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = torch.linalg.vector_norm(forces, dim=-1).amax(dim=1) > float(
        contact_threshold
    )
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if contacts.shape != foot_height.shape:
        raise RuntimeError(
            "Support contact and foot-body selections must have identical order and size."
        )

    contact_count = contacts.sum(dim=1)
    support_height = torch.sum(foot_height * contacts.to(foot_height.dtype), dim=1)
    support_height /= contact_count.clamp_min(1).to(foot_height.dtype)
    valid = contact_count >= int(minimum_support_contacts)
    return support_height, valid
