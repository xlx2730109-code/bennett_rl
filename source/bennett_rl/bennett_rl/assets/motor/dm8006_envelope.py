# Copyright (c) 2026, Bennett. All rights reserved.

"""DM-J8006-2EC torque-speed envelope from the official 24 V curve (numpy core).

This module is intentionally free of Isaac Lab imports so it can be used by
the simulator actuator (:mod:`bennett_rl.assets.motor.damiao`), by offline
plotting tools, and in tests without launching the simulation app.

All values are joint-side (after the 6:1 gearbox), matching the deployment
contracts and the training actuator configurations.

Envelope construction
---------------------

The digitized sweep (``dm_j8006_24v_120rpm_curve.csv``) is a 24 V load sweep:
each row is a torque the motor held together with the speed it reached.  Only
the falling branch of that sweep (rows above the 8 N-m rating) carries
envelope information; the plateau rows near 120 rpm only re-measure the same
knee with digitization jitter and are dropped.  The remaining measured points
are joined with two datasheet anchors:

  (0 rad/s, peak torque)            stall peak from the manual (20 N-m)
  measured falling branch           13 -> 9 N-m over 73.4 -> 118.6 rpm
  (no-load speed, 0 N-m)            190 rpm no-load @ 24 V from the manual

and the torque is zero beyond the no-load speed.  The two spans without
measurements (stall -> 73 rpm, plateau -> 190 rpm) are therefore covered by
conservative straight chords anchored at both ends.  The envelope is
four-quadrant symmetric (mirror on |speed|), which matches the DM drive
behaving identically in both rotation directions.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

CURVE_CSV = Path(__file__).resolve().parent / "dm_j8006_24v_120rpm_curve.csv"
"""Digitized 24 V / ~120 rpm load sweep (official performance-curve PNG)."""

RATED_TORQUE_NM = 8.0
"""Continuous torque rating (manual); also the split between plateau and falling branch."""

RATED_SPEED_RAD_S = 12.5663706144
"""Rated speed at the continuous rating: 120 rpm (manual; also the sweep plateau)."""

PEAK_TORQUE_NM = 20.0
"""Short-time peak torque at stall (manual)."""

NO_LOAD_SPEED_RAD_S = 19.8967534727
"""24 V no-load output speed: 190 rpm (manual)."""

ENVELOPE_POINTS = 512
"""Resolution of the resampled uniform envelope grid."""


def load_sweep(csv_path: Path | str = CURVE_CSV) -> dict[str, np.ndarray]:
    """Load the digitized performance sweep as float columns.

    Raises:
        ValueError: If the torque column is not strictly increasing (the
            digitization contract the falling-branch extraction relies on).
    """
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    data = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in ("torque_nm", "speed_rpm")
    }
    if not np.all(np.diff(data["torque_nm"]) > 0.0):
        raise ValueError(f"torque_nm must be strictly increasing in {csv_path}")
    return data


def build_envelope(
    csv_path: Path | str = CURVE_CSV,
    *,
    peak_torque_nm: float = PEAK_TORQUE_NM,
    rated_torque_nm: float = RATED_TORQUE_NM,
    no_load_speed_rad_s: float = NO_LOAD_SPEED_RAD_S,
    num_points: int = ENVELOPE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the joint-side maximum-torque envelope ``tau_max(omega)``.

    Args:
        csv_path: Digitized sweep CSV (see :data:`CURVE_CSV`).
        peak_torque_nm: Stall peak anchor.  A value below the strongest
            measured sweep point cannot raise the envelope (voltage physics
            does not depend on the configured current limit); the stall anchor
            is clamped to the measurement in that case.
        rated_torque_nm: Continuous rating; sweep rows above it form the
            falling branch.
        no_load_speed_rad_s: Zero-torque corner (190 rpm @ 24 V).
        num_points: Resolution of the returned uniform-speed grid.

    Returns:
        ``(omega, tau_max)``: uniform ``omega`` grid from 0 to the no-load
        speed and the maximum available (motoring) torque at each speed.
        Beyond the no-load speed the envelope is zero.
    """
    sweep = load_sweep(csv_path)

    # Falling branch: every sweep row pulling more than the continuous rating.
    # Those speeds decrease with increasing torque, so invert to (speed, torque)
    # with both ascending speeds and descending torque.
    branch = sweep["torque_nm"] > rated_torque_nm
    speed = np.flip(sweep["speed_rpm"][branch]) * (2.0 * np.pi / 60.0)
    torque = np.flip(sweep["torque_nm"][branch])
    if not len(speed):
        raise ValueError(f"No sweep rows above {rated_torque_nm} N-m in {csv_path}")

    omega = np.concatenate(([0.0], speed, [no_load_speed_rad_s]))
    tau = np.concatenate(([max(peak_torque_nm, torque[0])], torque, [0.0]))
    if not np.all(np.diff(omega) > 0.0):
        raise ValueError("Envelope speed corners are not strictly increasing")
    if not np.all(np.diff(tau) < 0.0):
        raise ValueError("Envelope torque corners are not strictly decreasing")

    grid_omega = np.linspace(0.0, no_load_speed_rad_s, num_points)
    return grid_omega, np.interp(grid_omega, omega, tau)


def interp_envelope_torch(speed_abs, no_load_speed_rad_s: float, tau_grid):
    """Linear-envelope lookup on the uniform grid (torch mirror of ``numpy.interp``).

    ``speed_abs`` may be any tensor shape; speeds above the no-load speed clamp
    to zero torque.  Kept next to :func:`build_envelope` so the actuator and
    standalone tests exercise the exact same interpolation code path.
    """
    grid_size = tau_grid.shape[0]
    x = (speed_abs / no_load_speed_rad_s).clamp(0.0, 1.0) * (grid_size - 1)
    i0 = torch.clamp(x.long(), max=grid_size - 2)
    frac = x - i0.to(x.dtype)
    return tau_grid[i0] * (1.0 - frac) + tau_grid[i0 + 1] * frac


def dc_motor_envelope(
    speed_rad_s: np.ndarray,
    *,
    effort_limit_nm: float = 8.0,
    saturation_effort_nm: float = 20.0,
    velocity_limit_rad_s: float = NO_LOAD_SPEED_RAD_S,
) -> np.ndarray:
    """Positive torque envelope of the production Isaac Lab DCMotor baseline.

    Mirrors ``isaaclab.actuators.DCMotor`` (8 / 20 / 19.8968) for comparison
    plots: a flat clip at the continuous rating, then a straight line from the
    zero-speed saturation torque down to zero at the no-load speed.
    """
    speed = np.asarray(speed_rad_s, dtype=np.float64)
    linear = saturation_effort_nm * (1.0 - speed / velocity_limit_rad_s)
    return np.clip(np.minimum(effort_limit_nm, linear), 0.0, None)
