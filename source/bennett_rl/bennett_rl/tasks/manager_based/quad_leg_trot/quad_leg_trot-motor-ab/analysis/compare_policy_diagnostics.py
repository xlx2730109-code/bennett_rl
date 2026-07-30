"""Create matched A/B tables and plots from Bennett diagnostic JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("velocity_x_rmse", "error_x", "rms"),
    ("yaw_rate_rmse", "error_yaw", "rms"),
    ("base_tilt_rms", "gravity_xy_norm", "rms"),
    ("base_ang_vel_xy_rms", "base_ang_vel_xy_norm", "rms"),
    ("joint_target_step_p95_deg", "joint_target_step_deg", "p95"),
    ("joint_acc_p95_rad_s2", "joint_acc_abs_rad_s2", "p95"),
    ("joint_torque_p95_nm", "joint_torque_abs_nm", "p95"),
    ("mechanical_power_mean_w", "mechanical_power_abs_w", "mean"),
    ("foot_force_p95_n", "foot_contact_force_n", "p95"),
    ("touchdown_force_p95_n", "touchdown_force_n", "p95"),
    ("contact_mismatch_mean", "contact_mismatch", "mean"),
    ("swing_height_mean_m", "swing_foot_height_m", "mean"),
    ("action_second_step_p95", "raw_action_second_step_abs", "p95"),
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_map(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {item["scenario"]: item for item in report["scenarios"]}


def _value(
    scenario: dict[str, object],
    metric: str,
    statistic: str,
) -> float | None:
    value = scenario["metrics"].get(metric, {}).get(statistic)
    return None if value is None else float(value)


def _write_scenario_table(
    a_map: dict[str, dict[str, object]],
    b_map: dict[str, dict[str, object]],
    scenarios: list[str],
    output: Path,
) -> None:
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["scenario", "metric", "statistic", "a", "b", "b_minus_a", "b_over_a"]
        )
        for scenario_name in scenarios:
            for label, metric, statistic in METRICS:
                a_value = _value(a_map[scenario_name], metric, statistic)
                b_value = _value(b_map[scenario_name], metric, statistic)
                if a_value is None or b_value is None:
                    writer.writerow(
                        [scenario_name, label, statistic, a_value, b_value, "", ""]
                    )
                    continue
                writer.writerow(
                    [
                        scenario_name,
                        label,
                        statistic,
                        a_value,
                        b_value,
                        b_value - a_value,
                        b_value / a_value if a_value != 0.0 else "",
                    ]
                )


def _aggregate(
    source: dict[str, dict[str, object]],
    scenarios: list[str],
    metric: str,
    statistic: str,
) -> float | None:
    values = [
        value
        for name in scenarios
        if (value := _value(source[name], metric, statistic)) is not None
    ]
    return None if not values else float(np.mean(values))


def _write_summary_table(
    a_map: dict[str, dict[str, object]],
    b_map: dict[str, dict[str, object]],
    movement_scenarios: list[str],
    output: Path,
) -> list[dict[str, float | str | None]]:
    rows: list[dict[str, float | str | None]] = []
    for label, metric, statistic in METRICS:
        a_value = _aggregate(a_map, movement_scenarios, metric, statistic)
        b_value = _aggregate(b_map, movement_scenarios, metric, statistic)
        rows.append(
            {
                "metric": label,
                "statistic": statistic,
                "a_mean_across_scenarios": a_value,
                "b_mean_across_scenarios": b_value,
                "b_minus_a": None
                if a_value is None or b_value is None
                else b_value - a_value,
                "b_over_a": None
                if a_value in (None, 0.0) or b_value is None
                else b_value / a_value,
            }
        )

    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _grouped_bar(
    scenarios: list[str],
    a_values: list[float],
    b_values: list[float],
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    x = np.arange(len(scenarios))
    width = 0.38
    fig, axis = plt.subplots(figsize=(12.0, 5.5), dpi=160)
    axis.bar(x - width / 2.0, a_values, width, label="A: 7/12/20")
    axis.bar(x + width / 2.0, b_values, width, label="B: 8/20/19.8968")
    axis.set_xticks(x, scenarios, rotation=25, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_scenario_metric(
    a_map: dict[str, dict[str, object]],
    b_map: dict[str, dict[str, object]],
    scenarios: list[str],
    metric: str,
    statistic: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    filtered: list[str] = []
    a_values: list[float] = []
    b_values: list[float] = []
    for name in scenarios:
        a_value = _value(a_map[name], metric, statistic)
        b_value = _value(b_map[name], metric, statistic)
        if a_value is None or b_value is None:
            continue
        filtered.append(name)
        a_values.append(a_value)
        b_values.append(b_value)
    _grouped_bar(filtered, a_values, b_values, ylabel, title, output)


def _write_markdown(
    report_a: dict[str, object],
    report_b: dict[str, object],
    scenarios: list[str],
    summary: list[dict[str, float | str | None]],
    output: Path,
) -> None:
    lines = [
        "# Trot motor A/B deterministic diagnostic",
        "",
        f"- A source: `{report_a['source_csv']}`",
        f"- B source: `{report_b['source_csv']}`",
        f"- Common scenarios: {', '.join(scenarios)}",
        f"- A rows used: {report_a['rows_used']}",
        f"- B rows used: {report_b['rows_used']}",
        "",
        "The report is descriptive. It does not declare a winner automatically,",
        "because lower is preferable for most errors/loads while swing height and",
        "contact behavior require task-specific interpretation.",
        "",
        "| Metric | A mean | B mean | B/A |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        a_value = row["a_mean_across_scenarios"]
        b_value = row["b_mean_across_scenarios"]
        ratio = row["b_over_a"]
        lines.append(
            f"| {row['metric']} | "
            f"{'n/a' if a_value is None else f'{a_value:.6g}'} | "
            f"{'n/a' if b_value is None else f'{b_value:.6g}'} | "
            f"{'n/a' if ratio is None else f'{ratio:.4f}'} |"
        )
    warnings = list(report_a.get("warnings", [])) + list(report_b.get("warnings", []))
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(a_json: Path, b_json: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    report_a = _load(a_json)
    report_b = _load(b_json)
    a_map = _scenario_map(report_a)
    b_map = _scenario_map(report_b)
    scenarios = sorted(set(a_map).intersection(b_map))
    if not scenarios:
        raise ValueError("A and B have no common scenarios")
    movement_scenarios = [name for name in scenarios if name != "stand"]

    _write_scenario_table(
        a_map,
        b_map,
        scenarios,
        output / "deterministic_scenario_metrics.csv",
    )
    summary = _write_summary_table(
        a_map,
        b_map,
        movement_scenarios,
        output / "deterministic_summary_metrics.csv",
    )
    _plot_scenario_metric(
        a_map,
        b_map,
        scenarios,
        "error_x",
        "rms",
        "RMSE (m/s)",
        "Body-frame x velocity tracking",
        output / "velocity_x_rmse.png",
    )
    _plot_scenario_metric(
        a_map,
        b_map,
        scenarios,
        "joint_target_step_deg",
        "p95",
        "P95 target step (deg / policy step)",
        "Joint-target smoothness",
        output / "joint_target_step_p95.png",
    )
    _plot_scenario_metric(
        a_map,
        b_map,
        scenarios,
        "touchdown_force_n",
        "p95",
        "P95 touchdown force (N)",
        "Touchdown force by scenario",
        output / "touchdown_force_p95.png",
    )
    _write_markdown(
        report_a,
        report_b,
        scenarios,
        summary,
        output / "REPORT.md",
    )
    print(f"[COMMON-SCENARIOS] {len(scenarios)}")
    print(f"[OUTPUT] {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-json", type=Path, required=True)
    parser.add_argument("--b-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare(args.a_json.resolve(), args.b_json.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
