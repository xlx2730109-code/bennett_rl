"""Audit a Bennett RSL-RL run for early-collapse and termination pathologies.

This is a read-only TensorBoard event-file check.  It is deliberately small
enough to run while training or in a checkpoint gate; it never edits a run.
Exit code 2 means that the configured collapse signature was detected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CORE_TAGS = {
    "reward": "Train/mean_reward",
    "episode_length": "Train/mean_episode_length",
    "root_height": "Episode_Termination/root_height",
    "base_contact": "Episode_Termination/base_contact",
    "timeout": "Episode_Termination/time_out",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="RSL-RL run directory containing TensorBoard event files")
    parser.add_argument("--tail-points", type=int, default=3, help="Number of final logged points used by the gate")
    parser.add_argument(
        "--collapse-episode-steps",
        type=float,
        default=10.0,
        help="Maximum final mean episode length for the collapse signature",
    )
    parser.add_argument(
        "--collapse-root-rate",
        type=float,
        default=0.8,
        help="Minimum final root-height termination rate for the collapse signature",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional new JSON report path")
    args = parser.parse_args()
    if args.tail_points < 1:
        parser.error("--tail-points must be positive")
    if args.collapse_episode_steps <= 0.0:
        parser.error("--collapse-episode-steps must be positive")
    if not 0.0 <= args.collapse_root_rate <= 1.0:
        parser.error("--collapse-root-rate must be in [0, 1]")
    return args


def _scalar_series(accumulator: EventAccumulator, tag: str) -> list[dict[str, float | int]]:
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [{"step": event.step, "value": event.value} for event in accumulator.Scalars(tag)]


def _tail_values(series: list[dict[str, float | int]], count: int) -> list[float]:
    return [float(point["value"]) for point in series[-count:]]


def main() -> int:
    args = _parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        raise NotADirectoryError(run)
    if not any(run.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"No TensorBoard event file found in: {run}")

    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
    accumulator.Reload()
    series = {name: _scalar_series(accumulator, tag) for name, tag in CORE_TAGS.items()}

    missing = [CORE_TAGS[name] for name in ("episode_length", "root_height") if not series[name]]
    if missing:
        raise RuntimeError(f"Required scalar tags are missing: {', '.join(missing)}")

    tail_count = min(args.tail_points, len(series["episode_length"]), len(series["root_height"]))
    episode_tail = _tail_values(series["episode_length"], tail_count)
    root_tail = _tail_values(series["root_height"], tail_count)
    collapse_detected = (
        tail_count == args.tail_points
        and all(value <= args.collapse_episode_steps for value in episode_tail)
        and all(value >= args.collapse_root_rate for value in root_tail)
    )

    summary: dict[str, object] = {
        "run": str(run),
        "status": "collapse_detected" if collapse_detected else "no_collapse_signature",
        "gate": {
            "tail_points": args.tail_points,
            "collapse_episode_steps": args.collapse_episode_steps,
            "collapse_root_rate": args.collapse_root_rate,
        },
        "tail": {},
    }
    for name, values in series.items():
        if not values:
            continue
        tail = values[-args.tail_points :]
        summary["tail"][name] = {
            "tag": CORE_TAGS[name],
            "last_step": int(values[-1]["step"]),
            "last": float(values[-1]["value"]),
            "values": [float(point["value"]) for point in tail],
        }

    print(f"run: {run}")
    print(f"status: {summary['status']}")
    for name, values in series.items():
        if values:
            point = values[-1]
            print(f"{name:>14}: step={point['step']} last={float(point['value']):.6g}")
        else:
            print(f"{name:>14}: missing")
    print(
        "collapse gate: "
        f"last {args.tail_points} episode_length <= {args.collapse_episode_steps:g} and "
        f"root_height termination >= {args.collapse_root_rate:g}"
    )

    if args.json is not None:
        output = args.json.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"json: {output}")

    return 2 if collapse_detected else 0


if __name__ == "__main__":
    sys.exit(main())
