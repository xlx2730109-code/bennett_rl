# TaskConfig 契约



"""Shared task-contract dataclass used by sim2sim.py and configs/*.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TaskConfig:
    task: str
    model: str                      # path to the MJCF (.xml)
    policy: str                     # path to the exported TorchScript policy.pt
    actuated_joints: list           # [8] joint names, training order
    default_joint_pos: np.ndarray   # [8] default/stand pose (obs + action offset)
    action_scale: float = 0.20
    clip_low: Optional[np.ndarray] = None   # [8] pose bounds
    clip_high: Optional[np.ndarray] = None  # [8]
    gait: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)  # fixed-base reference-trace task params
    step_dt: float = 0.02                     # control period 50 Hz
    phys_dt: float = 0.002                    # physics period 500 Hz
    obs_mode: str = "trot1"
    num_obs: int = 50
    num_actions: int = 8
    ep_len_s: float = 20.0
    init_base_pos: tuple = (0.0, 0.0, 0.36)
