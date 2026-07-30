"""Task-success termination terms for standalone Slope4."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class top_platform_success(ManagerTermBase):
    """Finish an episode only after all four feet stay on the upper platform.

    Checking every foot prevents the old curriculum from advancing while the
    rear legs are still on the ramp.  The short hold rejects one-frame contact
    and position spikes at the ramp-to-platform transition.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._hold_time = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._hold_time.zero_()
        else:
            self._hold_time[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        top_platform_start_distance: float,
        foot_margin: float,
        hold_time_s: float,
        minimum_clearance: float,
        minimum_upright_cosine: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
        relative_foot_x = foot_pos[:, :, 0] - env.scene.env_origins[:, 0, None]

        all_feet_on_top = torch.amin(relative_foot_x, dim=1) >= (
            float(top_platform_start_distance) + float(foot_margin)
        )
        base_clearance = asset.data.root_pos_w[:, 2] - torch.mean(foot_pos[:, :, 2], dim=1)
        clearance_ok = base_clearance >= float(minimum_clearance)
        upright_ok = -asset.data.projected_gravity_b[:, 2] >= float(minimum_upright_cosine)
        success_now = all_feet_on_top & clearance_ok & upright_ok

        self._hold_time = torch.where(
            success_now,
            self._hold_time + env.step_dt,
            torch.zeros_like(self._hold_time),
        )
        return self._hold_time >= float(hold_time_s)
