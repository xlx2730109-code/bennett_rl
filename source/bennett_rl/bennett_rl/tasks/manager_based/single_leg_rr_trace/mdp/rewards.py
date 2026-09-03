from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _reference_offsets(
    env: ManagerBasedRLEnv,
    amplitude_rad: float = math.radians(20.0),
    max_speed_rad_s: float = math.radians(35.0),
    calf_phase_rad: float = math.pi / 2.0,
) -> torch.Tensor:
    """Return RR thigh/calf reference offsets from default pose."""
    frequency_hz = max_speed_rad_s / max(2.0 * math.pi * amplitude_rad, 1.0e-6)
    phase = 2.0 * math.pi * frequency_hz * env.episode_length_buf.to(torch.float32) * env.step_dt
    return amplitude_rad * torch.stack((torch.sin(phase), torch.sin(phase + calf_phase_rad)), dim=-1)


def single_leg_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sin/cos phase observation for the reference single-leg trace."""
    amplitude_rad = math.radians(20.0)
    max_speed_rad_s = math.radians(35.0)
    frequency_hz = max_speed_rad_s / max(2.0 * math.pi * amplitude_rad, 1.0e-6)
    phase = 2.0 * math.pi * frequency_hz * env.episode_length_buf.to(torch.float32) * env.step_dt
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def single_leg_reference_offsets(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reference RR thigh/calf offsets relative to the default pose, in radians."""
    return _reference_offsets(env)


def single_leg_tracking_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["RR_thigh", "RR_calf"]),
) -> torch.Tensor:
    """Actual minus reference RR thigh/calf offsets, in radians."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return joint_pos_rel - _reference_offsets(env)


def single_leg_track_reference_exp(
    env: ManagerBasedRLEnv,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["RR_thigh", "RR_calf"]),
) -> torch.Tensor:
    """Reward smooth reference tracking for the suspended RR single-leg trace."""
    error = single_leg_tracking_error(env, asset_cfg)
    return torch.exp(-torch.sum(torch.square(error), dim=1) / max(sigma**2, 1.0e-6))


def single_leg_action_track_reference_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.06,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Reward the processed RR action for following the sinusoidal reference offsets."""
    action_term: SingleLegPositionAction = env.action_manager.get_term(action_name)
    error = action_term.processed_actions - _reference_offsets(env)
    return torch.exp(-torch.sum(torch.square(error), dim=1) / max(sigma**2, 1.0e-6))


@configclass
class SingleLegPositionActionCfg(ActionTermCfg):
    """Configuration for two-dimensional RR single-leg joint-position action."""

    class_type: type[ActionTerm] = MISSING
    asset_name: str = MISSING
    controlled_joint_names: list[str] = MISSING
    hold_joint_names: list[str] = MISSING
    scale: float = math.radians(20.0)
    max_joint_speed: float = math.radians(35.0)


class SingleLegPositionAction(ActionTerm):
    """Map two policy actions to RR thigh/calf position targets and hold the other leg joints at default."""

    cfg: SingleLegPositionActionCfg

    def __init__(self, cfg: SingleLegPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._controlled_joint_ids, self._controlled_joint_names = self._asset.find_joints(
            cfg.controlled_joint_names, preserve_order=True
        )
        self._hold_joint_ids, self._hold_joint_names = self._asset.find_joints(cfg.hold_joint_names, preserve_order=True)
        self._raw_actions = torch.zeros(self.num_envs, len(self._controlled_joint_ids), device=self.device)
        self._desired_offsets = torch.zeros_like(self._raw_actions)
        self._applied_offsets = torch.zeros_like(self._raw_actions)
        self._position_targets = self._asset.data.default_joint_pos[:, self._hold_joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return len(self._controlled_joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._applied_offsets

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._desired_offsets[:] = self._raw_actions * self.cfg.scale
        max_delta = self.cfg.max_joint_speed * self._env.step_dt
        delta = torch.clamp(self._desired_offsets - self._applied_offsets, -max_delta, max_delta)
        self._applied_offsets += delta

        self._position_targets[:] = self._asset.data.default_joint_pos[:, self._hold_joint_ids]
        for action_index, joint_id in enumerate(self._controlled_joint_ids):
            hold_index = self._hold_joint_ids.index(joint_id)
            self._position_targets[:, hold_index] = (
                self._asset.data.default_joint_pos[:, joint_id] + self._applied_offsets[:, action_index]
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._position_targets, joint_ids=self._hold_joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._desired_offsets.zero_()
            self._applied_offsets.zero_()
            self._position_targets[:] = self._asset.data.default_joint_pos[:, self._hold_joint_ids]
        else:
            self._raw_actions[env_ids] = 0.0
            self._desired_offsets[env_ids] = 0.0
            self._applied_offsets[env_ids] = 0.0
            self._position_targets[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._hold_joint_ids]
