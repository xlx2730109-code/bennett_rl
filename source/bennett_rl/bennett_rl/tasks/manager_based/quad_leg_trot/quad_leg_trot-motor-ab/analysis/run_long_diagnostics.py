"""Run matched long deterministic diagnostics for motor A and B checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
TASK_A = "Isaac-BennettRL-Flat-QuadLeg-Trot-MotorAB-A-Play-v0"
TASK_B = "Isaac-BennettRL-Flat-QuadLeg-Trot-MotorAB-B-Play-v0"


def _project_root() -> Path:
    for parent in HERE.parents:
        if (parent / "scripts" / "rsl_rl" / "collect_bennett_policy_diagnostics.py").is_file():
            return parent
    raise FileNotFoundError("Unable to locate the Bennett project root")


def _run(command: list[str], cwd: Path) -> None:
    print("[RUN] " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _checkpoint(run: Path, checkpoint: str) -> Path:
    path = (run / checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run(args: argparse.Namespace) -> None:
    root = _project_root()
    collector = root / "scripts" / "rsl_rl" / "collect_bennett_policy_diagnostics.py"
    analyzer = root / "scripts" / "analysis" / "analyze_bennett_policy_diagnostics.py"
    comparator = HERE / "compare_policy_diagnostics.py"
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    variants = (
        ("a", TASK_A, _checkpoint(args.run_a.resolve(), args.checkpoint)),
        ("b", TASK_B, _checkpoint(args.run_b.resolve(), args.checkpoint)),
    )
    for name, task, checkpoint in variants:
        csv_path = output / f"{name}.csv"
        command = [
            str(args.python),
            "-u",
            "-B",
            str(collector),
            "--task",
            task,
            "--checkpoint",
            str(checkpoint),
            "--output_csv",
            str(csv_path),
            "--num_envs",
            "1",
            "--seed",
            str(args.seed),
            "--settle_s",
            str(args.settle_s),
            "--command_s",
            str(args.command_s),
            "--recovery_s",
            str(args.recovery_s),
            "--repeats",
            str(args.repeats),
            "--headless",
        ]
        _run(command, root)
        _run(
            [
                str(args.python),
                "-B",
                str(analyzer),
                str(csv_path),
                "--transient_s",
                str(args.transient_s),
                "--output_json",
                str(output / f"{name}.json"),
            ],
            root,
        )

    _run(
        [
            str(args.python),
            "-B",
            str(comparator),
            "--a-json",
            str(output / "a.json"),
            "--b-json",
            str(output / "b.json"),
            "--output",
            str(output / "comparison"),
        ],
        root,
    )
    print(f"[COMPLETE] {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--checkpoint", default="model_600.pt")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--command-s", type=float, default=5.0)
    parser.add_argument("--recovery-s", type=float, default=1.0)
    parser.add_argument("--transient-s", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "output" / "deterministic_model_600_long",
    )
    args = parser.parse_args()
    if min(args.settle_s, args.command_s, args.recovery_s, args.transient_s) < 0.0:
        parser.error("durations must be non-negative")
    if args.command_s <= args.transient_s:
        parser.error("--command-s must be greater than --transient-s")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    return args


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
