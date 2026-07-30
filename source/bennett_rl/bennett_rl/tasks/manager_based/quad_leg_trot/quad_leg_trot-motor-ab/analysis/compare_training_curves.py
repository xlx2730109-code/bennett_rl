"""Compare common TensorBoard scalars from paired motor A/B training runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAG_PATTERN = re.compile(
    r"(mean_reward|mean_episode_length|track_|torque|dof_acc|action_rate|"
    r"second_difference|trot_|touchdown|termination)",
    re.IGNORECASE,
)


def _event_file(run: Path) -> Path:
    files = sorted(run.glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run}")
    return files[-1]


def _load(run: Path) -> tuple[EventAccumulator, set[str]]:
    accumulator = EventAccumulator(
        str(_event_file(run)),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    return accumulator, tags


def _series(accumulator: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    events = accumulator.Scalars(tag)
    return (
        np.asarray([event.step for event in events], dtype=np.int64),
        np.asarray([event.value for event in events], dtype=np.float64),
    )


def _tail_mean(values: np.ndarray, fraction: float = 0.2) -> float:
    count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.mean(values[-count:]))


def compare(run_a: Path, run_b: Path, output: Path, requested_tags: list[str]) -> None:
    acc_a, tags_a = _load(run_a)
    acc_b, tags_b = _load(run_b)
    common = sorted(tags_a.intersection(tags_b))

    if requested_tags:
        missing = sorted(set(requested_tags).difference(common))
        if missing:
            raise ValueError(f"Requested tags missing from one or both runs: {missing}")
        selected = requested_tags
    else:
        selected = [tag for tag in common if DEFAULT_TAG_PATTERN.search(tag)]

    if not selected:
        raise ValueError("No matching common scalar tags were found")

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "training_scalar_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["tag", "a_last", "b_last", "a_tail20_mean", "b_tail20_mean", "b_minus_a"]
        )
        for tag in selected:
            _, values_a = _series(acc_a, tag)
            _, values_b = _series(acc_b, tag)
            mean_a = _tail_mean(values_a)
            mean_b = _tail_mean(values_b)
            writer.writerow(
                [tag, values_a[-1], values_b[-1], mean_a, mean_b, mean_b - mean_a]
            )

    for index in range(0, len(selected), 9):
        batch = selected[index : index + 9]
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
        for axis, tag in zip(axes.flat, batch):
            steps_a, values_a = _series(acc_a, tag)
            steps_b, values_b = _series(acc_b, tag)
            axis.plot(steps_a, values_a, label="A: 7/12/20", linewidth=1.2)
            axis.plot(steps_b, values_b, label="B: 8/20/19.8968", linewidth=1.2)
            axis.set_title(tag, fontsize=9)
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(batch) :]:
            axis.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"training_curves_{index // 9 + 1:02d}.png", dpi=160)
        plt.close(fig)

    print(f"[A] {run_a}")
    print(f"[B] {run_b}")
    print(f"[TAGS] {len(selected)}")
    print(f"[SUMMARY] {summary_path}")
    print(f"[PLOTS] {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument("--tag", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare(args.run_a, args.run_b, args.output, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
