"""Analyze CSV files produced by collect_bennett_policy_diagnostics.py.

The report is scenario-aware and focuses on locomotion behavior that aggregate
training reward hides: command tracking, body oscillation, action/target
smoothness, touchdown severity, contact timing, effort, and termination.
Existing CSV files are read-only. JSON is written only when explicitly asked.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


JOINT_NAMES = (
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
)
FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
EXPECTED_SCENARIOS = (
    "stand",
    "forward_slow",
    "forward_nominal",
    "backward_slow",
    "backward_nominal",
    "yaw_left",
    "yaw_right",
    "forward_yaw",
    "backward_yaw",
    "lateral_left",
)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "rms": None, "std": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "rms": math.sqrt(statistics.fmean(value * value for value in values)),
        "std": statistics.pstdev(values),
        "p95": _percentile(values, 95.0),
        "max": max(values),
    }


def _number(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _finite(value: float) -> bool:
    return math.isfinite(value)


@dataclass
class ScenarioAccumulator:
    name: str
    repeat: int
    contact_threshold_n: float
    rows: int = 0
    terminated: bool = False
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    phase_start_s: float = math.inf
    phase_end_s: float = -math.inf
    metrics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    previous_raw: list[float] | None = None
    previous_target: list[float] | None = None
    previous_raw_delta: list[float] | None = None
    previous_contact: list[bool] | None = None
    joint_target_steps: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in JOINT_NAMES}
    )

    def add(self, row: dict[str, str]) -> None:
        self.rows += 1
        self.terminated = self.terminated or _number(row, "done") > 0.5
        self.command = tuple(_number(row, name) for name in ("command_x", "command_y", "command_yaw"))
        phase_time = _number(row, "phase_time_s")
        if _finite(phase_time):
            self.phase_start_s = min(self.phase_start_s, phase_time)
            self.phase_end_s = max(self.phase_end_s, phase_time)

        actual = (
            _number(row, "base_lin_vel_x"),
            _number(row, "base_lin_vel_y"),
            _number(row, "base_ang_vel_z"),
        )
        for axis, value, command in zip(("x", "y", "yaw"), actual, self.command):
            if _finite(value):
                self.metrics[f"actual_{axis}"].append(value)
                self.metrics[f"error_{axis}"].append(value - command)

        height = _number(row, "root_height_m")
        lin_z = _number(row, "base_lin_vel_z")
        ang_x = _number(row, "base_ang_vel_x")
        ang_y = _number(row, "base_ang_vel_y")
        gravity_x = _number(row, "projected_gravity_x")
        gravity_y = _number(row, "projected_gravity_y")
        power = _number(row, "mechanical_power_abs_w")
        if _finite(height):
            self.metrics["root_height_m"].append(height)
        if _finite(lin_z):
            self.metrics["base_lin_vel_z_abs"].append(abs(lin_z))
        if _finite(ang_x) and _finite(ang_y):
            self.metrics["base_ang_vel_xy_norm"].append(math.hypot(ang_x, ang_y))
        if _finite(gravity_x) and _finite(gravity_y):
            self.metrics["gravity_xy_norm"].append(math.hypot(gravity_x, gravity_y))
        if _finite(power):
            self.metrics["mechanical_power_abs_w"].append(abs(power))

        raw = [_number(row, f"raw_action_{name}") for name in JOINT_NAMES]
        targets = [_number(row, f"joint_target_rad_{name}") for name in JOINT_NAMES]
        if self.previous_raw is not None:
            raw_delta = [value - old for value, old in zip(raw, self.previous_raw)]
            finite_delta = [abs(value) for value in raw_delta if _finite(value)]
            self.metrics["raw_action_step_abs"].extend(finite_delta)
            if finite_delta:
                self.metrics["raw_action_step_l2"].append(math.sqrt(sum(value * value for value in finite_delta)))
            if self.previous_raw_delta is not None:
                second = [abs(value - old) for value, old in zip(raw_delta, self.previous_raw_delta)]
                self.metrics["raw_action_second_step_abs"].extend(value for value in second if _finite(value))
            self.previous_raw_delta = raw_delta
        if self.previous_target is not None:
            for name, value, old in zip(JOINT_NAMES, targets, self.previous_target):
                if _finite(value) and _finite(old):
                    step_deg = math.degrees(abs(value - old))
                    self.metrics["joint_target_step_deg"].append(step_deg)
                    self.joint_target_steps[name].append(step_deg)
        self.previous_raw = raw
        self.previous_target = targets

        for name in JOINT_NAMES:
            acceleration = _number(row, f"joint_acc_rad_s2_{name}")
            torque = _number(row, f"joint_torque_nm_{name}")
            if _finite(acceleration):
                self.metrics["joint_acc_abs_rad_s2"].append(abs(acceleration))
            if _finite(torque):
                self.metrics["joint_torque_abs_nm"].append(abs(torque))

        contact = []
        for name in FOOT_NAMES:
            force = _number(row, f"foot_contact_force_n_{name}")
            velocity_z = _number(row, f"foot_vel_z_m_s_{name}")
            height = _number(row, f"foot_height_m_{name}")
            desired = _number(row, f"desired_contact_{name}") >= 0.5
            touching = _finite(force) and force >= self.contact_threshold_n
            contact.append(touching)
            if _finite(force):
                self.metrics["foot_contact_force_n"].append(force)
            self.metrics["contact_mismatch"].append(float(touching != desired))
            if not desired and _finite(height):
                self.metrics["swing_foot_height_m"].append(height)
            if self.previous_contact is not None and touching and not self.previous_contact[len(contact) - 1]:
                if _finite(force):
                    self.metrics["touchdown_force_n"].append(force)
                if _finite(velocity_z):
                    self.metrics["touchdown_descent_speed_m_s"].append(max(0.0, -velocity_z))
        self.previous_contact = contact

    def report(self) -> dict[str, object]:
        summaries = {name: _summary(values) for name, values in self.metrics.items()}
        return {
            "scenario": self.name,
            "repeat": self.repeat,
            "rows": self.rows,
            "duration_s": max(0.0, self.phase_end_s - self.phase_start_s) if self.rows > 1 else 0.0,
            "terminated": self.terminated,
            "command": list(self.command),
            "metrics": summaries,
            "joint_target_step_deg": {
                name: _summary(values) for name, values in self.joint_target_steps.items()
            },
        }


def analyze(path: Path, *, transient_s: float, contact_threshold_n: float) -> dict[str, object]:
    groups: dict[tuple[int, str], ScenarioAccumulator] = {}
    rows_read = 0
    rows_used = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {
            "scenario",
            "repeat",
            "scenario_phase",
            "phase_time_s",
            "command_x",
            "command_y",
            "command_yaw",
            "base_lin_vel_x",
            "base_ang_vel_z",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        for row in reader:
            rows_read += 1
            if row.get("scenario_phase") != "command" or _number(row, "phase_time_s") < transient_s:
                continue
            scenario = row.get("scenario", "unknown")
            repeat = int(_number(row, "repeat"))
            key = (repeat, scenario)
            if key not in groups:
                groups[key] = ScenarioAccumulator(scenario, repeat, contact_threshold_n)
            groups[key].add(row)
            rows_used += 1

    reports = [groups[key].report() for key in sorted(groups)]
    observed = {report["scenario"] for report in reports}
    missing_scenarios = [name for name in EXPECTED_SCENARIOS if name not in observed]
    warnings = []
    if missing_scenarios:
        warnings.append(f"partial diagnostic; missing scenarios: {', '.join(missing_scenarios)}")
    for report in reports:
        if report["terminated"]:
            warnings.append(f"{report['scenario']} repeat {report['repeat']} terminated")
    return {
        "schema_version": 1,
        "source_csv": str(path.resolve()),
        "transient_excluded_s": transient_s,
        "contact_threshold_n": contact_threshold_n,
        "rows_read": rows_read,
        "rows_used": rows_used,
        "scenarios": reports,
        "warnings": warnings,
    }


def _value(report: dict[str, object], metric: str, statistic: str) -> float | None:
    return report["metrics"].get(metric, {}).get(statistic)


def _format(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_report(result: dict[str, object]) -> None:
    print(f"source: {result['source_csv']}")
    print(f"rows: read={result['rows_read']} used={result['rows_used']}")
    print("scenario             cmd_x  vx_mean vx_rmse cmd_wz wz_mean wz_rmse target_d95 impact_p95 tilt_rms mismatch")
    for report in result["scenarios"]:
        command = report["command"]
        print(
            f"{report['scenario']:<20} "
            f"{command[0]:>6.2f} "
            f"{_format(_value(report, 'actual_x', 'mean')):>7} "
            f"{_format(_value(report, 'error_x', 'rms')):>7} "
            f"{command[2]:>6.2f} "
            f"{_format(_value(report, 'actual_yaw', 'mean')):>7} "
            f"{_format(_value(report, 'error_yaw', 'rms')):>7} "
            f"{_format(_value(report, 'joint_target_step_deg', 'p95'), 2):>10} "
            f"{_format(_value(report, 'touchdown_force_n', 'p95'), 1):>10} "
            f"{_format(_value(report, 'gravity_xy_norm', 'rms')):>8} "
            f"{_format(_value(report, 'contact_mismatch', 'mean')):>8}"
        )
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("--transient_s", type=float, default=0.5)
    parser.add_argument("--contact_threshold_n", type=float, default=5.0)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()
    if args.transient_s < 0.0 or args.contact_threshold_n < 0.0:
        parser.error("thresholds must be non-negative")
    result = analyze(
        args.source_csv.resolve(),
        transient_s=args.transient_s,
        contact_threshold_n=args.contact_threshold_n,
    )
    print_report(result)
    if args.output_json is not None:
        output = args.output_json.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing JSON: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json: {output}")


if __name__ == "__main__":
    main()
