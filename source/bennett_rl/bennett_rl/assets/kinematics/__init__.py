# Copyright (c) 2026, Bennett. All rights reserved.

"""Kinematic model and charts for the Bennett closed-chain leg.

Deliberately light: pure numpy + stdlib (URDF parsing).  No ``isaaclab``
dependency, so every script here runs offline with a plain interpreter.
The authoritative geometry source is parsed at runtime from
``assets/robots/Urdf_Bennett_3/urdf/Urdf_Bennett_3.urdf`` -- never a copy.
"""

from .bennett_kinematics import (
    BENNETT_URDF,
    BennettLeg,
    LEG_JOINT_LIMITS,
    V5_DEFAULT_POSE,
    load_leg_model,
)

__all__ = [
    "BENNETT_URDF",
    "BennettLeg",
    "LEG_JOINT_LIMITS",
    "V5_DEFAULT_POSE",
    "load_leg_model",
]
