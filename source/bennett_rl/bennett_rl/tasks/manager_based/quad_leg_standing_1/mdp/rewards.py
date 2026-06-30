from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


ACTUATED_JOINTS = [
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
]


def upright_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a level base using projected gravity x/y components."""
    asset: Articulation = env.scene[asset_cfg.name]
    orientation_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-orientation_error / max(sigma**2, 1.0e-6))


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    sigma: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward root height close to the standing target height."""
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = asset.data.root_pos_w[:, 2] - target_height
    return torch.exp(-torch.square(height_error) / max(sigma**2, 1.0e-6))


def zero_xy_lin_vel_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward near-zero horizontal base velocity. Used for reward only, not actor observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / max(sigma**2, 1.0e-6))


def zero_ang_vel_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward near-zero base angular velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    return torch.exp(-ang_vel_error / max(sigma**2, 1.0e-6))


def default_joint_pose_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
) -> torch.Tensor:
    """Reward actuated joints staying near the configured default standing pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.mean(torch.square(joint_error), dim=1) / max(sigma**2, 1.0e-6))


def all_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward all selected feet being in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    feet_in_contact = contacts > threshold
    return torch.mean(feet_in_contact.to(torch.float32), dim=1)


def missing_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return number of selected feet that are not currently contacting the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return torch.sum((contacts <= threshold).to(torch.float32), dim=1)


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_1", preserve_order=True),
    threshold: float = 1.0,
    max_value: float = 2.0,
) -> torch.Tensor:
    """Penalize horizontal foot velocity while the foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    contacts = contacts > threshold
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.clamp(torch.sum(torch.linalg.norm(body_vel_xy, dim=-1) * contacts, dim=1), max=max_value)


def bounded_joint_vel(
    env: ManagerBasedRLEnv,
    scale: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
) -> torch.Tensor:
    """Bounded joint velocity penalty in [0, num_joints].

    Squared joint velocity can spike to extreme values when a fresh standing policy jitters or falls,
    which destabilizes PPO before the policy has a chance to improve.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.tanh(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) / max(scale, 1.0e-6)), dim=1)


def bounded_joint_acc(
    env: ManagerBasedRLEnv,
    scale: float = 500.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
) -> torch.Tensor:
    """Bounded joint acceleration penalty in [0, num_joints]."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.tanh(torch.abs(asset.data.joint_acc[:, asset_cfg.joint_ids]) / max(scale, 1.0e-6)), dim=1)


def soft_torque_limit_l2(
    env: ManagerBasedRLEnv,
    soft_limit: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
) -> torch.Tensor:
    """Penalize only torque above the continuous soft limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque_excess = torch.clamp(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) - soft_limit, min=0.0)
    return torch.sum(torch.square(torque_excess), dim=1)


def torque_rms(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
) -> torch.Tensor:
    """Monitoring term: RMS actuator torque over the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sqrt(torch.mean(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1))


def torque_max(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
) -> torch.Tensor:
    """Monitoring term: maximum absolute actuator torque over the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.max(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1).values


def base_height_below(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root height is too low."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def base_tilt_over(
    env: ManagerBasedRLEnv,
    max_projected_gravity_xy: float = 0.68,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when roll/pitch tilt is large enough to be unrecoverable."""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return tilt > max_projected_gravity_xy
