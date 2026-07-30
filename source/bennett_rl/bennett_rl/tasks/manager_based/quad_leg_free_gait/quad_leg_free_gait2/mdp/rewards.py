"""Per-leg participation term for gait-free locomotion.

The term requires every leg to achieve useful clearance over time, but never
specifies a phase, foot order, diagonal pair, duty factor, or fixed frequency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class gait_free_leg_lift_starvation_l2(ManagerTermBase):
    """Penalize any individual leg that has not produced a valid lift recently."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        num_feet = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].shape[2]
        self._time_since_valid_lift = torch.zeros((env.num_envs, num_feet), device=env.device)

    def reset(self, env_ids: torch.Tensor):
        self._time_since_valid_lift[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        contact_threshold: float,
        valid_lift_height: float,
        slow_allowed_time: float,
        fast_allowed_time: float,
        min_equivalent_speed: float,
        max_equivalent_speed: float,
        yaw_equivalent_radius: float,
        command_name: str,
        command_deadband: float,
        max_normalized_excess: float,
    ) -> torch.Tensor:
        if slow_allowed_time <= 0.0 or fast_allowed_time <= 0.0:
            raise ValueError("Allowed lift times must be positive.")
        if slow_allowed_time < fast_allowed_time:
            raise ValueError("slow_allowed_time must be greater than or equal to fast_allowed_time.")
        if max_equivalent_speed <= min_equivalent_speed:
            raise ValueError("max_equivalent_speed must be greater than min_equivalent_speed.")

        command = env.command_manager.get_command(command_name)
        equivalent_speed = torch.sqrt(
            torch.square(command[:, 0])
            + torch.square(command[:, 1])
            + torch.square(float(yaw_equivalent_radius) * command[:, 2])
        )
        moving = equivalent_speed >= float(command_deadband)
        speed_fraction = torch.clamp(
            (equivalent_speed - float(min_equivalent_speed))
            / (float(max_equivalent_speed) - float(min_equivalent_speed)),
            min=0.0,
            max=1.0,
        )
        allowed_time = float(slow_allowed_time) + speed_fraction * (
            float(fast_allowed_time) - float(slow_allowed_time)
        )

        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        force_history = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        contact = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1) > float(contact_threshold)

        asset: Articulation = env.scene[asset_cfg.name]
        foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        contact_float = contact.to(torch.float32)
        stance_count = contact_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        stance_height = (foot_height * contact_float).sum(dim=1, keepdim=True) / stance_count
        relative_height = foot_height - stance_height
        has_support = contact.any(dim=1, keepdim=True)
        valid_lift = (
            torch.logical_not(contact)
            & (relative_height >= float(valid_lift_height))
            & has_support
            & moving.unsqueeze(1)
        )

        elapsed = self._time_since_valid_lift + float(env.step_dt)
        self._time_since_valid_lift.copy_(
            torch.where(moving.unsqueeze(1), elapsed, torch.zeros_like(elapsed))
        )
        self._time_since_valid_lift.masked_fill_(valid_lift, 0.0)

        normalized_excess = torch.relu(
            self._time_since_valid_lift / allowed_time.unsqueeze(1) - 1.0
        )
        normalized_excess = torch.clamp(normalized_excess, max=float(max_normalized_excess))
        return torch.mean(torch.square(normalized_excess), dim=1)
