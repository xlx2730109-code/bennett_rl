# Copyright (c) 2026, Bennett. All rights reserved.

"""Motor datasheets, digitized curves, and actuator models for Bennett.

Deliberately light: only the numpy/torch envelope core is re-exported here so
offline tools can ``from bennett_rl.assets.motor import build_envelope``
without the simulation app.  The Isaac Lab actuator lives in
:mod:`bennett_rl.assets.motor.damiao` and must be imported from there directly
(importing it pulls in ``isaaclab``, which requires the app).
"""

from .dm8006_envelope import (
    CURVE_CSV,
    NO_LOAD_SPEED_RAD_S,
    PEAK_TORQUE_NM,
    RATED_SPEED_RAD_S,
    RATED_TORQUE_NM,
    build_envelope,
    dc_motor_envelope,
    interp_envelope_torch,
    load_sweep,
)

__all__ = [
    "CURVE_CSV",
    "NO_LOAD_SPEED_RAD_S",
    "PEAK_TORQUE_NM",
    "RATED_SPEED_RAD_S",
    "RATED_TORQUE_NM",
    "build_envelope",
    "dc_motor_envelope",
    "interp_envelope_torch",
    "load_sweep",
]
