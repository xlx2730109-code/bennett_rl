"""Compare Isaac Lab DCMotor A/B envelopes with DM-J8006 24 V references.

The manufacturer CSV is a digitized 24 V, approximately 120 rpm load sweep.
It is plotted only as reference measurements and is not treated as a maximum
torque-speed envelope.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONTRACT = ROOT / "dm_j8006_2ec_v1_1_24v.yaml"
DEFAULT_CURVE = ROOT / "dm_j8006_24v_120rpm_curve.csv"
DEFAULT_OUTPUT = ROOT / "generated" / "dm_j8006_actuator_envelope_comparison.png"

MOTOR_A = {
    "label": "A: older/current 7 / 12 / 20",
    "effort_limit_nm": 7.0,
    "saturation_effort_nm": 12.0,
    "velocity_limit_rad_s": 20.0,
}


def _load_motor_b(path: Path) -> dict[str, float | str]:
    contract = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    baseline = contract["isaaclab_conservative_baseline"]
    return {
        "label": "B: datasheet candidate 8 / 20 / 19.8968",
        "effort_limit_nm": float(baseline["effort_limit_nm"]),
        "saturation_effort_nm": float(baseline["saturation_effort_nm"]),
        "velocity_limit_rad_s": float(baseline["velocity_limit_rad_s"]),
    }


def _load_curve(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in ("torque_nm", "speed_rpm")
    }


def available_positive_torque(
    speed_rad_s: np.ndarray | float,
    *,
    effort_limit_nm: float,
    saturation_effort_nm: float,
    velocity_limit_rad_s: float,
) -> np.ndarray:
    """Match Isaac Lab DCMotor's positive torque-speed hard limits."""

    speed = np.asarray(speed_rad_s, dtype=np.float64)
    torque_speed_limit = saturation_effort_nm * (
        1.0 - speed / velocity_limit_rad_s
    )
    return np.clip(
        np.minimum(effort_limit_nm, torque_speed_limit),
        a_min=0.0,
        a_max=None,
    )


def _motor_kwargs(motor: dict[str, float | str]) -> dict[str, float]:
    return {
        "effort_limit_nm": float(motor["effort_limit_nm"]),
        "saturation_effort_nm": float(motor["saturation_effort_nm"]),
        "velocity_limit_rad_s": float(motor["velocity_limit_rad_s"]),
    }


def plot_comparison(
    contract_path: Path,
    curve_path: Path,
    output_path: Path,
    show: bool,
) -> dict[str, float]:
    motor_b = _load_motor_b(contract_path)
    curve = _load_curve(curve_path)

    speed_rpm = np.linspace(0.0, 205.0, 1000)
    speed_rad_s = speed_rpm * (2.0 * math.pi / 60.0)
    rated_speed_rpm = 120.0
    rated_speed_rad_s = rated_speed_rpm * (2.0 * math.pi / 60.0)
    rated_torque_nm = 8.0

    torque_a = available_positive_torque(speed_rad_s, **_motor_kwargs(MOTOR_A))
    torque_b = available_positive_torque(speed_rad_s, **_motor_kwargs(motor_b))
    rated_a = float(
        available_positive_torque(rated_speed_rad_s, **_motor_kwargs(MOTOR_A))
    )
    rated_b = float(
        available_positive_torque(rated_speed_rad_s, **_motor_kwargs(motor_b))
    )

    fig, axis = plt.subplots(figsize=(11.0, 7.0), dpi=180)
    axis.plot(speed_rpm, torque_a, linewidth=2.5, label=str(MOTOR_A["label"]))
    axis.plot(speed_rpm, torque_b, linewidth=2.5, label=str(motor_b["label"]))

    valid_curve = (curve["torque_nm"] > 0.0) & (curve["speed_rpm"] > 0.0)
    axis.scatter(
        curve["speed_rpm"][valid_curve],
        curve["torque_nm"][valid_curve],
        s=28,
        marker="o",
        facecolors="none",
        edgecolors="#6a3d9a",
        label="Digitized 24 V load sweep (reference, not envelope)",
        zorder=4,
    )
    axis.scatter(
        [rated_speed_rpm],
        [rated_torque_nm],
        marker="*",
        s=180,
        color="black",
        label="Manual rated point: 8 Nm @ 120 rpm",
        zorder=5,
    )
    axis.axvline(190.0, color="0.4", linestyle="--", linewidth=1.2)
    axis.text(
        190.0,
        19.5,
        "24 V manual no-load speed: 190 rpm",
        rotation=90,
        va="top",
        ha="right",
        color="0.35",
    )
    axis.annotate(
        f"A @ 120 rpm = {rated_a:.2f} Nm",
        xy=(rated_speed_rpm, rated_a),
        xytext=(133.0, rated_a - 1.5),
        arrowprops={"arrowstyle": "->"},
    )
    axis.annotate(
        f"B @ 120 rpm = {rated_b:.2f} Nm",
        xy=(rated_speed_rpm, rated_b),
        xytext=(132.0, rated_b + 2.0),
        arrowprops={"arrowstyle": "->"},
    )

    axis.set_title("DM-J8006 24 V: Isaac Lab DCMotor A/B envelope comparison")
    axis.set_xlabel("Output speed (rpm)")
    axis.set_ylabel("Available positive output torque (N·m)")
    axis.set_xlim(0.0, 205.0)
    axis.set_ylim(0.0, 21.0)
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    if show:
        plt.show()
    plt.close(fig)

    return {
        "a_torque_at_120rpm_nm": rated_a,
        "b_torque_at_120rpm_nm": rated_b,
        "rated_torque_nm": rated_torque_nm,
        "a_rated_point_error_nm": rated_a - rated_torque_nm,
        "b_rated_point_error_nm": rated_b - rated_torque_nm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--curve", type=Path, default=DEFAULT_CURVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = plot_comparison(args.contract, args.curve, args.output, args.show)
    for key, value in metrics.items():
        print(f"[ENVELOPE] {key}={value:.6f}")
    print(f"[OUTPUT] {args.output}")
    print("[SCOPE] Manufacturer points are a load sweep, not a maximum envelope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
