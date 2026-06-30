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


QUAD_TRACE_JOINTS = [
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
]

LEG_SIGNS = {
    "FL": 1.0,
    "FR": -1.0,
    "RL": -1.0,
    "RR": 1.0,
}


def _base_phase(
    env: ManagerBasedRLEnv,
    amplitude_rad: float = math.radians(20.0),
    max_speed_rad_s: float = math.radians(35.0),
) -> torch.Tensor:
    frequency_hz = max_speed_rad_s / max(2.0 * math.pi * amplitude_rad, 1.0e-6)
    return 2.0 * math.pi * frequency_hz * env.episode_length_buf.to(torch.float32) * env.step_dt


def _reference_offsets(
    env: ManagerBasedRLEnv,
    joint_names: list[str] = QUAD_TRACE_JOINTS,
    amplitude_rad: float = math.radians(20.0),
    max_speed_rad_s: float = math.radians(35.0),
    calf_phase_rad: float = math.pi / 2.0,
) -> torch.Tensor:
    """Return thigh/calf reference offsets from default pose for all selected joints."""
    phase = _base_phase(env, amplitude_rad, max_speed_rad_s)
    refs = []
    for joint_name in joint_names:
        leg_name, joint_kind = joint_name.split("_", maxsplit=1)
        joint_phase = phase
        if joint_kind == "calf":
            joint_phase = joint_phase + calf_phase_rad
        refs.append(LEG_SIGNS[leg_name] * amplitude_rad * torch.sin(joint_phase))
    return torch.stack(refs, dim=-1)


def _base_reference_offsets(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the two-dimensional RR-style base thigh/calf reference."""
    phase = _base_phase(env)
    return math.radians(20.0) * torch.stack((torch.sin(phase), torch.sin(phase + math.pi / 2.0)), dim=-1)


def _joint_names_from_cfg(asset_cfg: SceneEntityCfg) -> list[str]:
    if asset_cfg.joint_names is None:
        return QUAD_TRACE_JOINTS
    if isinstance(asset_cfg.joint_names, str):
        return [asset_cfg.joint_names]
    return list(asset_cfg.joint_names)


def quad_leg_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sin/cos phase observation shared by all four single-leg-equivalent traces."""
    phase = _base_phase(env)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def quad_leg_reference_offsets(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reference eight-joint offsets relative to the default pose, in radians."""
    return _reference_offsets(env)


def quad_leg_tracking_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=QUAD_TRACE_JOINTS, preserve_order=True),
) -> torch.Tensor:
    """Actual minus reference thigh/calf offsets, in radians."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return joint_pos_rel - _reference_offsets(env, _joint_names_from_cfg(asset_cfg))


def quad_leg_track_reference_exp(
    env: ManagerBasedRLEnv,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=QUAD_TRACE_JOINTS, preserve_order=True),
) -> torch.Tensor:
    """Average the original single-leg tracking reward over the four legs."""
    error = quad_leg_tracking_error(env, asset_cfg)
    leg_errors = error.reshape(error.shape[0], -1, 2)
    leg_rewards = torch.exp(-torch.sum(torch.square(leg_errors), dim=2) / max(sigma**2, 1.0e-6))
    return torch.mean(leg_rewards, dim=1)


def quad_leg_action_track_reference_exp(
    env: ManagerBasedRLEnv,
    sigma: float = 0.06,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Reward the two policy outputs for matching the base RR-style reference target."""
    action_term: QuadLegPositionAction = env.action_manager.get_term(action_name)
    error = action_term.processed_actions - _base_reference_offsets(env)
    return torch.exp(-torch.sum(torch.square(error), dim=1) / max(sigma**2, 1.0e-6))


@configclass
class QuadLegPositionActionCfg(ActionTermCfg):
    """Configuration for coordinated two-dimensional thigh/calf joint-position action."""

    class_type: type[ActionTerm] = MISSING
    asset_name: str = MISSING
    controlled_joint_names: list[str] = MISSING
    hold_joint_names: list[str] = MISSING
    scale: float = math.radians(20.0)
    max_joint_speed: float = math.radians(60.0)


class QuadLegPositionAction(ActionTerm):
    """Map two policy actions to a coordinated four-leg trot-like target.

    The policy learns the RR-style base thigh/calf offsets. The action term expands them to all
    controlled joints through LEG_SIGNS, keeping the policy action space as easy as the old single-leg task.
    """

    cfg: QuadLegPositionActionCfg

    def __init__(self, cfg: QuadLegPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._controlled_joint_ids, self._controlled_joint_names = self._asset.find_joints(
            cfg.controlled_joint_names, preserve_order=True
        )
        self._hold_joint_ids, self._hold_joint_names = self._asset.find_joints(cfg.hold_joint_names, preserve_order=True)
        self._joint_action_indices = torch.tensor(
            [0 if joint_name.split("_", maxsplit=1)[1] == "thigh" else 1 for joint_name in self._controlled_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self._joint_signs = torch.tensor(
            [LEG_SIGNS[joint_name.split("_", maxsplit=1)[0]] for joint_name in self._controlled_joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self._controlled_hold_indices = torch.tensor(
            [self._hold_joint_ids.index(joint_id) for joint_id in self._controlled_joint_ids],
            dtype=torch.long,
            device=self.device,
        )
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._desired_offsets = torch.zeros_like(self._raw_actions)
        self._applied_offsets = torch.zeros_like(self._raw_actions)
        self._position_targets = self._asset.data.default_joint_pos[:, self._hold_joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return 2

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
        expanded_offsets = self._applied_offsets[:, self._joint_action_indices] * self._joint_signs
        self._position_targets[:, self._controlled_hold_indices] = (
            self._asset.data.default_joint_pos[:, self._controlled_joint_ids] + expanded_offsets
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
