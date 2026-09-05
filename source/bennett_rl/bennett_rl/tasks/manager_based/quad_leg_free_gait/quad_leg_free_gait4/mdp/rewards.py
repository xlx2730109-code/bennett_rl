"""Gait-agnostic foot-clearance reward for the higher-clearance free-gait task.

free_gait1's ``swing_clearance`` was a *narrow Gaussian* (target 0.035, sigma
0.012): a peaked reward that gave ~zero gradient once the foot sat far from the
target, which is exactly why that term trained poorly. This is a different,
research-backed shape -- the one-sided saturating swing-phase clearance shortfall

    penalty = I_swing * (1 - exp{-k * max(0, h_min - h_foot)})

modelled on arXiv:2403.10723 (Towards Dynamic Quadrupedal Gaits). It is:

  * one-sided   -- only a foot that the policy elects to swing AND that sits below
                   the clearance threshold costs reward; a high foot costs nothing;
  * saturating  -- the penalty is bounded at ``weight`` and never grows unbounded,
                   so the policy is not pushed into a violent over-lift kick;
  * never sparse-- every bit of insufficient lift costs some reward, so the policy
                   always has gradient to lift the foot higher.

The target is deliberately generous (~2x the sim lift that transferred as a
dragging foot on hardware), which is the point: after the sim->real lift drop,
the real robot still clears the ground.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _moving_mask(env: ManagerBasedRLEnv, command_name: str, command_deadband: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :3], dim=1) >= float(command_deadband)


def _foot_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=1) > float(threshold)


def swing_foot_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    min_clearance: float,
    saturating_k: float,
    command_name: str,
    command_deadband: float,
) -> torch.Tensor:
    """One-sided saturating penalty for feet in swing that stay too low.

    Returns a *positive* penalty magnitude in [0, 1] per env (averaged over the
    feet in swing); a foot hovering below ``min_clearance`` scores toward 1 as
    ``shortfall`` grows. Attach it to the reward-set with a NEGATIVE weight so it
    becomes a penalty (e.g. ``weight=-0.15``).
    """
    contact = _foot_contacts(env, sensor_cfg, threshold)
    contact_float = contact.to(torch.float32)
    in_swing = (~contact).to(torch.float32)

    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # world foot Z
    stance_count = contact_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    stance_height = (foot_height * contact_float).sum(dim=1, keepdim=True) / stance_count
    rel_height = torch.clamp(foot_height - stance_height, min=0.0)

    shortfall = torch.clamp(float(min_clearance) - rel_height, min=0.0)
    penalty = (1.0 - torch.exp(-float(saturating_k) * shortfall)) * in_swing
    swing_count = in_swing.sum(dim=1).clamp_min(1.0)

    has_support = (contact_float.sum(dim=1) > 0.0).to(torch.float32)
    moving = _moving_mask(env, command_name, command_deadband).to(torch.float32)
    return moving * has_support * penalty.sum(dim=1) / swing_count
