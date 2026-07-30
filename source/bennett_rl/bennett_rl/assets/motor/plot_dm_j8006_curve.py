"""Reproduce and validate the digitized DM-J8006 24 V performance curve.

The CSV values were digitized from:
    24V 120RPM 8006电机性能曲线图.png

They are approximate plot readings, not the manufacturer's original raw samples.
Ibus is reconstructed as Pin / 24 V because its plotted line is hidden by the
Pin curve. The speed value at 5 Nm is reconstructed from Pout = torque * speed
because the blue and green curves overlap at that point. The plot is a 24 V,
approximately 120 rpm load sweep and must not be treated as a complete maximum
torque-speed envelope.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "dm_j8006_24v_120rpm_curve.csv"
DEFAULT_OUTPUT = ROOT / "generated" / "dm_j8006_24v_120rpm_reproduced.png"


def load_curve(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    required = {
        "torque_nm",
        "vbus_v",
        "ibus_a",
        "speed_rpm",
        "efficiency_pct",
        "input_power_w",
        "output_power_w",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in required
    }


def validate_curve(data: dict[str, np.ndarray]) -> dict[str, float]:
    torque = data["torque_nm"]
    speed = data["speed_rpm"]
    pin = data["input_power_w"]
    pout = data["output_power_w"]
    vbus = data["vbus_v"]
    ibus = data["ibus_a"]
    efficiency = data["efficiency_pct"]

    if not np.all(np.diff(torque) > 0.0):
        raise ValueError("torque_nm must be strictly increasing")
    if np.any(np.asarray(list(data.values())) < 0.0):
        raise ValueError("The digitized motoring-curve values must be non-negative")

    active = torque > 0.0
    calculated_pout = torque * speed * (2.0 * math.pi / 60.0)
    calculated_ibus = np.divide(pin, vbus, out=np.zeros_like(pin), where=vbus > 0.0)
    calculated_efficiency = np.divide(
        100.0 * pout,
        pin,
        out=np.zeros_like(pin),
        where=pin > 0.0,
    )

    return {
        "pout_rmse_w": float(
            np.sqrt(np.mean(np.square(pout[active] - calculated_pout[active])))
        ),
        "pout_max_abs_error_w": float(
            np.max(np.abs(pout[active] - calculated_pout[active]))
        ),
        "ibus_max_abs_error_a": float(
            np.max(np.abs(ibus[active] - calculated_ibus[active]))
        ),
        "efficiency_rmse_percentage_points": float(
            np.sqrt(
                np.mean(
                    np.square(
                        efficiency[active] - calculated_efficiency[active]
                    )
                )
            )
        ),
        "efficiency_max_abs_error_percentage_points": float(
            np.max(
                np.abs(efficiency[active] - calculated_efficiency[active])
            )
        ),
    }


def _add_left_axis(host: plt.Axes, offset: float, color: str) -> plt.Axes:
    axis = host.twinx()
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_position(("axes", offset))
    axis.spines["left"].set_visible(True)
    axis.yaxis.set_label_position("left")
    axis.yaxis.set_ticks_position("left")
    axis.tick_params(axis="y", colors=color)
    axis.yaxis.label.set_color(color)
    return axis


def _add_right_axis(host: plt.Axes, offset: float, color: str) -> plt.Axes:
    axis = host.twinx()
    axis.spines["right"].set_position(("axes", offset))
    axis.tick_params(axis="y", colors=color)
    axis.yaxis.label.set_color(color)
    return axis


def plot_curve(data: dict[str, np.ndarray], output: Path, show: bool) -> None:
    torque = data["torque_nm"]

    fig, speed_axis = plt.subplots(figsize=(15.0, 9.0), dpi=200)
    fig.subplots_adjust(left=0.27, right=0.73, top=0.90, bottom=0.15)

    vbus_axis = _add_left_axis(speed_axis, -0.27, "green")
    ibus_axis = _add_left_axis(speed_axis, -0.13, "#00bfbf")
    efficiency_axis = speed_axis.twinx()
    efficiency_axis.tick_params(axis="y", colors="red")
    efficiency_axis.yaxis.label.set_color("red")
    pin_axis = _add_right_axis(speed_axis, 1.14, "#bf00bf")
    pout_axis = _add_right_axis(speed_axis, 1.28, "#bfbf00")

    vbus_axis.plot(torque, data["vbus_v"], color="green", linewidth=2.5)
    ibus_axis.plot(torque, data["ibus_a"], color="#00bfbf", linewidth=2.5)
    speed_axis.plot(torque, data["speed_rpm"], color="blue", linewidth=2.5)
    efficiency_axis.plot(
        torque, data["efficiency_pct"], color="red", linewidth=2.5
    )
    pin_axis.plot(torque, data["input_power_w"], color="#bf00bf", linewidth=2.5)
    pout_axis.plot(torque, data["output_power_w"], color="#bfbf00", linewidth=2.5)

    speed_axis.set_title(
        "8006 Motor Performance Curve At 24V 120 RPM\n"
        "(digitized reproduction; not raw manufacturer samples)",
        fontsize=18,
    )
    speed_axis.set_xlabel("Torque (N·m)", fontsize=13)
    speed_axis.set_ylabel("Speed (rpm)", fontsize=13, color="blue")
    vbus_axis.set_ylabel("Vbus (V)", fontsize=13)
    ibus_axis.set_ylabel("Ibus (A)", fontsize=13)
    efficiency_axis.set_ylabel("Efficiency (%)", fontsize=13)
    pin_axis.set_ylabel("Pin (W)", fontsize=13)
    pout_axis.set_ylabel("Pout (W)", fontsize=13)

    speed_axis.set_xlim(-0.65, 13.65)
    speed_axis.set_ylim(-6.0, 126.0)
    vbus_axis.set_ylim(-1.0, 26.0)
    ibus_axis.set_ylim(-0.5, 13.5)
    efficiency_axis.set_ylim(-3.0, 72.0)
    pin_axis.set_ylim(-12.0, 315.0)
    pout_axis.set_ylim(-5.0, 124.0)

    speed_axis.set_xticks(np.arange(0.0, 14.0, 2.0))
    speed_axis.set_yticks(np.arange(0.0, 121.0, 20.0))
    vbus_axis.set_yticks(np.arange(0.0, 26.0, 5.0))
    ibus_axis.set_yticks(np.arange(0.0, 13.0, 2.0))
    efficiency_axis.set_yticks(np.arange(0.0, 71.0, 10.0))
    pin_axis.set_yticks(np.arange(0.0, 301.0, 50.0))
    pout_axis.set_yticks(np.arange(0.0, 121.0, 20.0))

    speed_axis.tick_params(axis="y", colors="blue")
    speed_axis.grid(axis="x", linestyle="--", color="0.7", alpha=0.8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_curve(args.csv)
    metrics = validate_curve(data)

    print("[SOURCE] Digitized from the manufacturer PNG; values are approximate.")
    print("[DERIVED] Ibus=Pin/24V; speed@5Nm reconstructed from mechanical Pout.")
    print("[SCOPE] 24 V, approximately 120 rpm load sweep; not a full envelope.")
    for name, value in metrics.items():
        print(f"[VALIDATION] {name}={value:.6f}")

    if not args.validate_only:
        plot_curve(data, args.output, args.show)
        print(f"[OUTPUT] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
