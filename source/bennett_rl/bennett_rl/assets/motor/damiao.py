# Copyright (c) 2026, Bennett. All rights reserved.

"""DaMiao DM-J8006-2EC actuator model with a datasheet torque-speed envelope.

The real motor runs DM's MIT mode: the drive computes

    tau = kp * (q_des - q) + kd * (qdot_des - qdot) + tau_ff

and then clamps the result to what the hardware can actually deliver.  Two
physical limits act, and this class reproduces both, in the same order:

1. the configured phase-current limit -- a flat ``+-effort_limit`` clip;
2. the voltage / back-EMF torque-speed envelope of the official 24 V
   performance curve, built by
   :func:`bennett_rl.assets.motor.dm8006_envelope.build_envelope`.

``DelayedPDActuator`` is used as the base so CAN round-trip latency
(1-2 control steps at 50 Hz) can be layered on per configuration through
``min_delay`` / ``max_delay``; the defaults (0, 0) keep the model delay-free.

Usage (inside an articulation cfg, next to the existing ``DCMotorCfg`` groups):

    from bennett_rl.assets.motor.damiao import DamiaoMotorCfg

    "base_legs": DamiaoMotorCfg(
        joint_names_expr=[".*_thigh", ".*_calf"],
        effort_limit=8.0,           # driver current limit -> flat clip
        velocity_limit=19.8967,     # joint-side no-load speed @ 24 V
        stiffness=30.0, damping=2.0,
        # min_delay=1, max_delay=2,  # optional CAN latency randomization
    )

Scope notes (see also ``dm_j8006_2ec_v1_1_24v.yaml``): the envelope is the
24 V voltage limit; thermal derating between the 20 N-m peak and the 8 N-m
continuous rating is NOT modeled -- pick ``effort_limit`` per task the same
way the DCMotor baseline does.  Effort, speed and envelope are all joint-side
(after the 6:1 gearbox).
"""

from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from bennett_rl.assets.motor.dm8006_envelope import (
    CURVE_CSV,
    NO_LOAD_SPEED_RAD_S,
    PEAK_TORQUE_NM,
    RATED_TORQUE_NM,
    build_envelope,
    interp_envelope_torch,
)


@configclass
class DamiaoMotorCfg(DelayedPDActuatorCfg):
    """Configuration for the DM-J8006-2EC envelope actuator."""

    class_type: type = None  # filled after DamiaoMotor is defined (forward ref)

    curve_csv: Path = CURVE_CSV
    """Digitized 24 V performance sweep driving the envelope LUT."""

    peak_torque_nm: float = PEAK_TORQUE_NM
    """Stall peak torque anchor of the envelope (manual: 20 N-m)."""

    rated_torque_nm: float = RATED_TORQUE_NM
    """Continuous rating; sweep rows above it form the envelope's falling branch."""

    no_load_speed_rad_s: float = NO_LOAD_SPEED_RAD_S
    """Zero-torque envelope corner (manual: 190 rpm @ 24 V, joint-side)."""


class DamiaoMotor(DelayedPDActuator):
    """MIT-PD actuator clipped by the DM-J8006-2EC torque-speed envelope.

    ``compute`` follows :class:`~isaaclab.actuators.IdealPDActuator` (the MIT-PD
    law); ``_clip_effort`` applies the flat current-limit clip and then the
    datasheet envelope at the measured joint speed, four-quadrant symmetric.
    """

    cfg: DamiaoMotorCfg

    def __init__(self, cfg: DamiaoMotorCfg, *args, **kwargs):
        # The flat clip defaults to the peak rating when the task does not pick
        # a (usually conservative) current limit of its own.
        if cfg.effort_limit is None:
            cfg.effort_limit = cfg.peak_torque_nm
        super().__init__(cfg, *args, **kwargs)

        omega_grid, tau_grid = build_envelope(
            cfg.curve_csv,
            peak_torque_nm=cfg.peak_torque_nm,
            rated_torque_nm=cfg.rated_torque_nm,
            no_load_speed_rad_s=cfg.no_load_speed_rad_s,
        )
        self._no_load_speed = float(omega_grid[-1])
        self._tau_grid = torch.as_tensor(tau_grid, dtype=torch.float32, device=self._device)
        # measured joint speed stashed by compute() for _clip_effort
        self._joint_vel = torch.zeros_like(self.computed_effort)

    """
    Operations.
    """

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save the current speed so _clip_effort can apply the envelope
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    """
    Helper functions.
    """

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # voltage envelope at the measured |speed| ...
        tau_max = interp_envelope_torch(self._joint_vel.abs(), self._no_load_speed, self._tau_grid)
        # ... never above the configured current limit (flat +- clip)
        tau_max = torch.minimum(tau_max, self.effort_limit)
        return torch.clamp(effort, min=-tau_max, max=tau_max)


# the class object only exists now; close the config's forward reference
DamiaoMotorCfg.class_type = DamiaoMotor
