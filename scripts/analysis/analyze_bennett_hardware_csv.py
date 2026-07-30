"""Summarize Bennett hardware telemetry without loading the full CSV into memory.

The production deployment writes one row per joint and policy update.  This
script supports both the legacy schema (``loop_count`` as the sample key) and
the newer schema (``policy_step``).  It reports command coverage, policy/target
step size, target-to-real tracking error, joint velocity, torque, feedback age,
and base IMU stability.  Existing logs are read-only; JSON output is written
only when ``--output_json`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


JOINT_ORDER = (
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
)


class Reservoir:
    """Deterministic bounded reservoir used for approximate percentiles."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = max(1, int(capacity))
        self.values: list[float] = []
        self.seen = 0
        self._random = random.Random(seed)

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
            return
        index = self._random.randrange(self.seen)
        if index < self.capacity:
            self.values[index] = value

    def percentile(self, quantile: float) -> float | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        position = (len(ordered) - 1) * float(quantile) / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass
class OnlineStats:
    """Streaming scalar statistics with approximate percentiles."""

    reservoir_size: int
    seed: int
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    samples: Reservoir = field(init=False)

    def __post_init__(self) -> None:
        self.samples = Reservoir(self.reservoir_size, self.seed)

    def add(self, value: float | str | None) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(number):
            return
        self.count += 1
        self.total += number
        self.total_square += number * number
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)
        self.samples.add(number)

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "rms": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "min": None,
                "max": None,
            }
        return {
            "count": self.count,
            "mean": self.total / self.count,
            "rms": math.sqrt(self.total_square / self.count),
            "p50": self.samples.percentile(50.0),
            "p95": self.samples.percentile(95.0),
            "p99": self.samples.percentile(99.0),
            "min": self.minimum,
            "max": self.maximum,
        }


def _float(row: dict[str, str], name: str) -> float | None:
    try:
        value = float(row.get(name, "nan"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _command(row: dict[str, str]) -> tuple[float, float, float]:
    return tuple(round(_float(row, name) or 0.0, 6) for name in ("command_x", "command_y", "command_yaw"))


def _stats_group(reservoir_size: int, seed: int) -> dict[str, OnlineStats]:
    names = (
        "raw_action",
        "action",
        "target_rad",
        "measured_rad",
        "stable_action_step",
        "stable_action_second_step",
        "stable_target_step_deg",
        "stable_target_second_step_deg",
        "tracking_error_deg",
        "joint_velocity_rad_s",
        "torque_nm",
        "feedback_age_s",
    )
    return {name: OnlineStats(reservoir_size, seed + index) for index, name in enumerate(names)}


def analyze(path: Path, *, reservoir_size: int, max_rows: int | None = None) -> dict[str, object]:
    joint_stats = {name: _stats_group(reservoir_size, 1000 * index) for index, name in enumerate(JOINT_ORDER)}
    previous: dict[
        str,
        tuple[
            float | None,
            float | None,
            tuple[float, float, float],
            float | None,
            float | None,
        ],
    ] = {}
    command_counts: Counter[tuple[float, float, float]] = Counter()
    stage_counts: Counter[str] = Counter()
    rate_limits: Counter[float] = Counter()
    base_stats = {
        "ang_vel_x": OnlineStats(reservoir_size, 81),
        "ang_vel_y": OnlineStats(reservoir_size, 82),
        "ang_vel_z": OnlineStats(reservoir_size, 83),
        "gravity_xy_norm": OnlineStats(reservoir_size, 84),
    }

    rows_read = 0
    policy_rows = 0
    policy_samples = 0
    first_elapsed: float | None = None
    last_elapsed: float | None = None
    last_sample_key: str | None = None
    sample_key_name: str | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        sample_key_name = "policy_step" if "policy_step" in reader.fieldnames else "loop_count"
        required = {
            sample_key_name,
            "startup_phase",
            "joint_name",
            "action",
            "target_sim_rad",
            "train_default_rad",
            "rel_rad",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row in reader:
            rows_read += 1
            if max_rows is not None and rows_read > max_rows:
                break
            if row.get("startup_phase") != "policy":
                continue
            policy_rows += 1
            elapsed = _float(row, "elapsed_s")
            if elapsed is not None:
                first_elapsed = elapsed if first_elapsed is None else first_elapsed
                last_elapsed = elapsed

            sample_key = row[sample_key_name]
            command = _command(row)
            if sample_key != last_sample_key:
                last_sample_key = sample_key
                policy_samples += 1
                command_counts[command] += 1
                stage_counts[row.get("diag_stage", "unknown")] += 1
                rate_limit = _float(row, "target_rate_limit_deg_s")
                if rate_limit is not None:
                    rate_limits[round(rate_limit, 6)] += 1
                for axis, field_name in zip(("x", "y", "z"), ("base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z")):
                    base_stats[f"ang_vel_{axis}"].add(_float(row, field_name))
                gravity_x = _float(row, "projected_gravity_x")
                gravity_y = _float(row, "projected_gravity_y")
                if gravity_x is not None and gravity_y is not None:
                    base_stats["gravity_xy_norm"].add(math.hypot(gravity_x, gravity_y))

            joint_name = row.get("joint_name", "")
            if joint_name not in joint_stats:
                continue
            metrics = joint_stats[joint_name]
            raw_action = _float(row, "raw_action")
            action = _float(row, "action")
            target = _float(row, "target_sim_rad")
            train_default = _float(row, "train_default_rad")
            real_relative = _float(row, "rel_rad")
            real = (
                train_default + real_relative
                if train_default is not None and real_relative is not None
                else None
            )
            metrics["raw_action"].add(raw_action)
            metrics["action"].add(action)
            metrics["target_rad"].add(target)
            metrics["measured_rad"].add(real)
            velocity = _float(row, "dq_policy_obs_rad_s")
            torque = _float(row, "tau_nm")
            metrics["joint_velocity_rad_s"].add(abs(velocity) if velocity is not None else None)
            metrics["torque_nm"].add(abs(torque) if torque is not None else None)
            metrics["feedback_age_s"].add(_float(row, "feedback_age_s"))
            if target is not None and real is not None:
                metrics["tracking_error_deg"].add(math.degrees(abs(target - real)))

            prior = previous.get(joint_name)
            action_delta = None
            target_delta = None
            if prior is not None and prior[2] == command:
                previous_action, previous_target, _, previous_action_delta, previous_target_delta = prior
                if action is not None and previous_action is not None:
                    action_delta = action - previous_action
                    metrics["stable_action_step"].add(abs(action_delta))
                    if previous_action_delta is not None:
                        metrics["stable_action_second_step"].add(abs(action_delta - previous_action_delta))
                if target is not None and previous_target is not None:
                    target_delta = target - previous_target
                    metrics["stable_target_step_deg"].add(math.degrees(abs(target_delta)))
                    if previous_target_delta is not None:
                        metrics["stable_target_second_step_deg"].add(
                            math.degrees(abs(target_delta - previous_target_delta))
                        )
            previous[joint_name] = (action, target, command, action_delta, target_delta)

    duration = None
    if first_elapsed is not None and last_elapsed is not None:
        duration = max(0.0, last_elapsed - first_elapsed)
    commands = [
        {"command": list(command), "samples": count, "fraction": count / max(policy_samples, 1)}
        for command, count in command_counts.most_common()
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "source_csv": str(path.resolve()),
        "sample_key": sample_key_name,
        "rows_read": rows_read,
        "policy_rows": policy_rows,
        "policy_samples": policy_samples,
        "duration_s": duration,
        "commands": commands,
        "diagnostic_stages": dict(stage_counts),
        "target_rate_limits_deg_s": dict(rate_limits),
        "base": {name: stats.as_dict() for name, stats in base_stats.items()},
        "joints": {
            joint_name: {metric_name: stats.as_dict() for metric_name, stats in metrics.items()}
            for joint_name, metrics in joint_stats.items()
        },
    }
    result["warnings"] = _warnings(result)
    return result


def _warnings(result: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    commands = result["commands"]
    assert isinstance(commands, list)
    command_values = [entry["command"] for entry in commands]
    if not any(abs(float(command[2])) > 1.0e-6 for command in command_values):
        warnings.append("No non-zero yaw command is present; turning cannot be evaluated from this log.")
    if not any(float(command[0]) < -1.0e-6 for command in command_values):
        warnings.append("No backward command is present; reverse motion cannot be evaluated from this log.")
    rate_limits = result["target_rate_limits_deg_s"]
    assert isinstance(rate_limits, dict)
    if any(float(value) >= 5000.0 for value in rate_limits):
        warnings.append("Target rate limiting is effectively disabled (>=5000 deg/s).")
    joints = result["joints"]
    assert isinstance(joints, dict)
    for joint_name, metrics in joints.items():
        target_p95 = metrics["stable_target_step_deg"]["p95"]
        tracking_rms = metrics["tracking_error_deg"]["rms"]
        if target_p95 is not None and float(target_p95) > 3.0:
            warnings.append(f"{joint_name}: stable target-step p95 exceeds 3 deg per policy update.")
        if tracking_rms is not None and float(tracking_rms) > 5.0:
            warnings.append(f"{joint_name}: target-to-real tracking RMS exceeds 5 deg.")
    return warnings


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _degrees(value: object) -> float | None:
    return None if value is None else math.degrees(float(value))


def print_summary(result: dict[str, object]) -> None:
    print(f"source: {result['source_csv']}")
    print(
        f"policy samples: {result['policy_samples']}  policy rows: {result['policy_rows']}  "
        f"duration: {_fmt(result['duration_s'])} s"
    )
    print("command coverage:")
    for entry in result["commands"][:20]:
        print(f"  {entry['command']}: {entry['samples']} ({100.0 * entry['fraction']:.2f}%)")
    print("joint diagnostics:")
    for joint_name in JOINT_ORDER:
        metrics = result["joints"][joint_name]
        print(
            f"  {joint_name:9s} target_range={_fmt(_degrees(metrics['target_rad']['min']))}/"
            f"{_fmt(_degrees(metrics['target_rad']['max']))} deg  "
            f"target_step[p95/max]={_fmt(metrics['stable_target_step_deg']['p95'])}/"
            f"{_fmt(metrics['stable_target_step_deg']['max'])} deg  "
            f"target_d2[p95]={_fmt(metrics['stable_target_second_step_deg']['p95'])} deg  "
            f"tracking[rms/p95]={_fmt(metrics['tracking_error_deg']['rms'])}/"
            f"{_fmt(metrics['tracking_error_deg']['p95'])} deg  "
            f"torque[rms/max]={_fmt(metrics['torque_nm']['rms'])}/{_fmt(metrics['torque_nm']['max'])} Nm"
        )
    if result["warnings"]:
        print("warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Deployment telemetry CSV")
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--reservoir_size", type=int, default=50_000)
    parser.add_argument("--max_rows", type=int, default=None, help="Optional quick-audit row limit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.reservoir_size < 100:
        raise ValueError("--reservoir_size must be at least 100")
    result = analyze(args.csv, reservoir_size=args.reservoir_size, max_rows=args.max_rows)
    print_summary(result)
    if args.output_json is not None:
        output = args.output_json.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing JSON: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json: {output}")


if __name__ == "__main__":
    main()
