"""Plot Bennett torque evaluation summaries.

This module is intentionally lightweight because it is imported by
``eval_bennett_torque.py`` after the Isaac Lab rollout has finished.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _use_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _joint_names(joint_rows: list[dict]) -> list[str]:
    return [row["joint"] for row in joint_rows]


def _values(joint_rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in joint_rows]


def _nullable_values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) if row.get(key) is not None else 0.0 for row in rows]


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.clf()


def plot_overall(summary_payload: dict, plot_dir: Path, prefix: str) -> Path:
    """Plot overall torque, joint velocity, power, and tracking quality."""
    plt = _use_matplotlib()
    overall = summary_payload["overall"]
    metrics = [
        ("torque_rms", overall["torque"]["rms"], "Nm"),
        ("torque_p95", overall["torque"]["p95_abs"], "Nm"),
        ("torque_p99", overall["torque"]["p99_abs"], "Nm"),
        ("torque_max", overall["torque"]["max_abs"], "Nm"),
        ("speed_error", overall["mean_abs_speed_error_mps"], "m/s"),
        ("base_height", overall["mean_base_height_m"], "m"),
    ]
    labels = [item[0] for item in metrics]
    values = [item[1] for item in metrics]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(labels, values, color=["#4c78a8", "#4c78a8", "#4c78a8", "#e45756", "#72b7b2", "#54a24b"])
    ax.set_title("Bennett Torque Evaluation Summary")
    ax.set_ylabel("mixed units")
    ax.grid(axis="y", alpha=0.25)
    for bar, (_, value, unit) in zip(bars, metrics):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f} {unit}", ha="center", va="bottom")

    if overall.get("done_count_during_eval", 0):
        ax.text(
            0.99,
            0.95,
            f"resets during eval: {overall['done_count_during_eval']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#b27900",
        )

    path = plot_dir / f"{prefix}_overall.png"
    _finish(fig, path)
    return path


def plot_joint_torque(joint_rows: list[dict], plot_dir: Path, prefix: str) -> Path:
    """Plot per-joint RMS, p95, p99, and max absolute torque."""
    plt = _use_matplotlib()
    names = _joint_names(joint_rows)
    x = range(len(names))

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    width = 0.2
    series = [
        ("rms", _values(joint_rows, "torque_rms"), "#4c78a8"),
        ("p95", _values(joint_rows, "torque_p95_abs"), "#f58518"),
        ("p99", _values(joint_rows, "torque_p99_abs"), "#e45756"),
        ("max", _values(joint_rows, "torque_max_abs"), "#b279a2"),
    ]
    for offset, (label, vals, color) in zip((-1.5, -0.5, 0.5, 1.5), series):
        ax.bar([i + offset * width for i in x], vals, width=width, label=label, color=color)

    ax.axhline(8.0, color="#d62728", linestyle="--", linewidth=1.2, label="8 Nm limit")
    ax.set_title("Per-Joint Torque Statistics")
    ax.set_ylabel("absolute torque (Nm)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="upper left")

    path = plot_dir / f"{prefix}_joint_torque.png"
    _finish(fig, path)
    return path


def plot_design_reference(joint_rows: list[dict], plot_dir: Path, prefix: str) -> Path:
    """Plot p95/p99 torque as a motor-selection reference."""
    plt = _use_matplotlib()
    names = _joint_names(joint_rows)
    p95 = _values(joint_rows, "torque_p95_abs")
    p99 = _values(joint_rows, "torque_p99_abs")

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.plot(names, p95, marker="o", label="p95 torque", color="#4c78a8")
    ax.plot(names, p99, marker="o", label="p99 torque", color="#e45756")
    ax.axhline(8.0, color="#d62728", linestyle="--", linewidth=1.2, label="8 Nm limit")
    ax.set_title("Motor Selection Reference")
    ax.set_ylabel("absolute torque (Nm)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    path = plot_dir / f"{prefix}_design_reference.png"
    _finish(fig, path)
    return path


def plot_power(joint_rows: list[dict], plot_dir: Path, prefix: str) -> Path:
    """Plot per-joint mechanical power statistics."""
    plt = _use_matplotlib()
    names = _joint_names(joint_rows)
    x = range(len(names))

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    width = 0.22
    series = [
        ("mean", _values(joint_rows, "power_mean_abs"), "#54a24b"),
        ("rms", _values(joint_rows, "power_rms"), "#4c78a8"),
        ("p95", _values(joint_rows, "power_p95_abs"), "#f58518"),
        ("max", _values(joint_rows, "power_max_abs"), "#e45756"),
    ]
    for offset, (label, vals, color) in zip((-1.5, -0.5, 0.5, 1.5), series):
        ax.bar([i + offset * width for i in x], vals, width=width, label=label, color=color)

    ax.set_title("Per-Joint Mechanical Power")
    ax.set_ylabel("absolute mechanical power (W)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper left")

    path = plot_dir / f"{prefix}_power.png"
    _finish(fig, path)
    return path


def plot_foot_height(foot_rows: list[dict], plot_dir: Path, prefix: str) -> Path | None:
    """Plot per-foot swing-height statistics."""
    if not foot_rows:
        return None
    plt = _use_matplotlib()
    names = [row["foot"] for row in foot_rows]
    x = range(len(names))

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    width = 0.2
    series = [
        ("mean", _nullable_values(foot_rows, "swing_height_mean"), "#54a24b"),
        ("p95", _nullable_values(foot_rows, "swing_height_p95"), "#f58518"),
        ("p99", _nullable_values(foot_rows, "swing_height_p99"), "#e45756"),
        ("max", _nullable_values(foot_rows, "swing_height_max"), "#b279a2"),
    ]
    for offset, (label, vals, color) in zip((-1.5, -0.5, 0.5, 1.5), series):
        ax.bar([i + offset * width for i in x], vals, width=width, label=label, color=color)

    ax.set_title("Per-Foot Swing Height Statistics")
    ax.set_ylabel("foot height during swing (m)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper left")
    for row_id, row in enumerate(foot_rows):
        if int(row.get("swing_height_sample_count", 0)) == 0:
            ax.text(row_id, 0.01, "no swing samples", ha="center", va="bottom", rotation=90, color="#b27900")

    path = plot_dir / f"{prefix}_foot_swing_height.png"
    _finish(fig, path)
    return path


def plot_foot_x(foot_rows: list[dict], plot_dir: Path, prefix: str) -> Path | None:
    """Plot per-foot body-frame x-position statistics."""
    if not foot_rows or "all_x_min" not in foot_rows[0]:
        return None
    plt = _use_matplotlib()
    names = [row["foot"] for row in foot_rows]
    x = range(len(names))

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    width = 0.2
    series = [
        ("min", _nullable_values(foot_rows, "all_x_min"), "#4c78a8"),
        ("p50", _nullable_values(foot_rows, "all_x_p50"), "#54a24b"),
        ("max", _nullable_values(foot_rows, "all_x_max"), "#e45756"),
        ("swing max", _nullable_values(foot_rows, "swing_x_max"), "#b279a2"),
    ]
    for offset, (label, vals, color) in zip((-1.5, -0.5, 0.5, 1.5), series):
        ax.bar([i + offset * width for i in x], vals, width=width, label=label, color=color)

    ax.set_title("Per-Foot Body-Frame X Position")
    ax.set_ylabel("x position in yaw-aligned base frame (m)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper left")

    path = plot_dir / f"{prefix}_foot_x.png"
    _finish(fig, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an existing Bennett torque summary JSON.")
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    args = parser.parse_args()

    payload = json.loads(args.summary_json.read_text(encoding="utf-8"))
    plot_dir = args.output_dir or args.summary_json.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.summary_json.stem.replace("_summary", "")

    plot_overall(payload, plot_dir, prefix)
    plot_joint_torque(payload["joints"], plot_dir, prefix)
    plot_design_reference(payload["joints"], plot_dir, prefix)
    plot_power(payload["joints"], plot_dir, prefix)
    if "feet" in payload:
        plot_foot_height(payload["feet"], plot_dir, prefix)
        plot_foot_x(payload["feet"], plot_dir, prefix)
    print(f"Wrote plots to: {plot_dir}")


if __name__ == "__main__":
    main()
