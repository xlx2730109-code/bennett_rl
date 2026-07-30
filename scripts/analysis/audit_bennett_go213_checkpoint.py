"""Fail-fast deterministic action audit for Bennett Go2-13 checkpoints.

This is a model-only guard for the policy-distribution failures seen in the
Go2-13 experiments.  It never opens Isaac Sim and never edits a run.  It
supports both the restored 50-D Go2-11 observation contract and the retired
58-D filtered-action contract, probes the actor mean over keyboard-aligned
commands and a full crawl cycle, then fails if the learned Gaussian std or
deterministic clip fraction exceeds the configured limits.

This guard is intentionally not a locomotion-quality verdict.  A checkpoint
that passes must still complete a deterministic Isaac rollout/video before
Sim2Real use.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


SUPPORTED_OBSERVATION_DIMS = (50, 58)
GAIT_FREQUENCY_HZ = 0.55
GAIT_DUTY_FACTOR = 0.78
GAIT_SWING_HEIGHT_M = 0.065
GAIT_OFFSETS = torch.tensor((0.0, 0.5, 0.75, 0.25), dtype=torch.float32)
SCENARIOS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_slow": (0.06, 0.0, 0.0),
    "forward_nominal": (0.14, 0.0, 0.0),
    "backward_slow": (-0.06, 0.0, 0.0),
    "backward_nominal": (-0.14, 0.0, 0.0),
    "yaw_left": (0.0, 0.0, 0.25),
    "yaw_right": (0.0, 0.0, -0.25),
    "forward_yaw": (0.10, 0.0, 0.25),
    "backward_yaw": (-0.10, 0.0, 0.25),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Go2-13 run directory")
    parser.add_argument("--checkpoint", type=Path, help="Specific model_*.pt; defaults to the latest checkpoint")
    parser.add_argument("--all-checkpoints", action="store_true", help="Report every model_*.pt in the run")
    parser.add_argument("--phase-samples", type=int, default=100)
    parser.add_argument("--max-std", type=float, default=1.0)
    parser.add_argument("--max-clip-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.phase_samples < 4:
        parser.error("--phase-samples must be at least 4")
    if args.max_std <= 0.0:
        parser.error("--max-std must be positive")
    if not 0.0 <= args.max_clip_fraction <= 1.0:
        parser.error("--max-clip-fraction must be in [0, 1]")
    return args


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    return int(match.group(1))


def _load_action_clip(run: Path) -> float:
    agent_path = run / "params" / "agent.yaml"
    if not agent_path.is_file():
        raise FileNotFoundError(agent_path)
    agent_cfg = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
    clip = agent_cfg.get("clip_actions")
    if clip is None or float(clip) <= 0.0:
        raise ValueError(f"params/agent.yaml must contain a positive clip_actions, got {clip!r}")
    return float(clip)


def _policy_observation(
    phase: float,
    command: tuple[float, float, float],
    observation_dim: int,
) -> torch.Tensor:
    """Build a nominal restored or filtered Go2-13 observation."""

    if observation_dim not in SUPPORTED_OBSERVATION_DIMS:
        raise ValueError(f"Unsupported Go2-13 observation dimension: {observation_dim}")
    phase_start = 33 if observation_dim == 50 else 41
    leg_phase_start = phase_start + 2
    contacts_start = leg_phase_start + 8
    gait_params_start = contacts_start + 4

    observation = torch.zeros(observation_dim, dtype=torch.float32)
    observation[5] = -1.0  # projected gravity
    observation[6:9] = torch.tensor(command)

    moving = math.sqrt(sum(value * value for value in command)) >= 0.025
    if moving:
        phase_rad = 2.0 * math.pi * phase
        observation[phase_start : phase_start + 2] = torch.tensor((math.sin(phase_rad), math.cos(phase_rad)))
        leg_phase = torch.remainder(torch.tensor(phase) - GAIT_OFFSETS, 1.0)
        leg_phase_rad = 2.0 * math.pi * leg_phase
        observation[leg_phase_start : leg_phase_start + 8] = torch.stack(
            (torch.sin(leg_phase_rad), torch.cos(leg_phase_rad)), dim=-1
        ).reshape(-1)
        observation[contacts_start : contacts_start + 4] = (
            leg_phase >= (1.0 - GAIT_DUTY_FACTOR)
        ).to(torch.float32)
        observation[gait_params_start : gait_params_start + 3] = torch.tensor(
            (GAIT_FREQUENCY_HZ, GAIT_DUTY_FACTOR, GAIT_SWING_HEIGHT_M)
        )
    else:
        observation[phase_start : phase_start + 2] = torch.tensor((0.0, 1.0))
        observation[leg_phase_start : leg_phase_start + 8] = torch.tensor((0.0, 1.0) * 4)
        observation[contacts_start : contacts_start + 4] = 1.0
        observation[gait_params_start : gait_params_start + 3] = torch.tensor((0.0, 1.0, 0.0))
    return observation


def _activation(name: str, value: torch.Tensor) -> torch.Tensor:
    normalized = name.lower()
    if normalized == "elu":
        return F.elu(value)
    if normalized == "relu":
        return F.relu(value)
    if normalized == "tanh":
        return torch.tanh(value)
    if normalized == "selu":
        return F.selu(value)
    raise ValueError(f"Unsupported actor activation for audit: {name!r}")


def _actor_forward(
    state: dict[str, torch.Tensor],
    observation: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    linear_indices = sorted(
        int(match.group(1))
        for key in state
        if (match := re.fullmatch(r"actor\.(\d+)\.weight", key)) is not None
    )
    if not linear_indices:
        raise KeyError("Checkpoint has no actor.<index>.weight tensors")

    value = observation
    for position, index in enumerate(linear_indices):
        value = F.linear(value, state[f"actor.{index}.weight"], state[f"actor.{index}.bias"])
        if position + 1 < len(linear_indices):
            value = _activation(activation, value)
    return value


def _action_std(state: dict[str, torch.Tensor]) -> torch.Tensor:
    if "std" in state:
        return state["std"]
    if "log_std" in state:
        return torch.exp(state["log_std"])
    raise KeyError("Checkpoint has neither std nor log_std")


def _audit_checkpoint(
    checkpoint_path: Path,
    action_clip: float,
    activation: str,
    phase_samples: int,
) -> dict[str, float | int | str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"{checkpoint_path} has no model_state_dict")

    actor_input_key = min(
        (key for key in state if re.fullmatch(r"actor\.\d+\.weight", key)),
        key=lambda key: int(key.split(".")[1]),
    )
    observation_dim = int(state[actor_input_key].shape[1])
    if observation_dim not in SUPPORTED_OBSERVATION_DIMS:
        raise ValueError(
            f"{checkpoint_path.name}: expected one of {SUPPORTED_OBSERVATION_DIMS} actor observations, "
            f"got {observation_dim}"
        )

    phases = [index / phase_samples for index in range(phase_samples)]
    observations = torch.stack(
        [
            _policy_observation(phase, command, observation_dim)
            for command in SCENARIOS.values()
            for phase in phases
        ]
    )
    with torch.no_grad():
        raw_actions = _actor_forward(state, observations, activation)

    std = _action_std(state)
    clipped_actions = torch.clamp(raw_actions, -action_clip, action_clip)
    clip_fraction = torch.mean((torch.abs(raw_actions) > action_clip).to(torch.float32))

    scenario_actions = clipped_actions.reshape(len(SCENARIOS), phase_samples, -1)
    moving_actions = scenario_actions[1:]
    phase_span = moving_actions.amax(dim=1) - moving_actions.amin(dim=1)

    return {
        "checkpoint": checkpoint_path.name,
        "step": _checkpoint_step(checkpoint_path),
        "std_mean": float(std.mean()),
        "std_max": float(std.max()),
        "raw_abs_mean": float(torch.abs(raw_actions).mean()),
        "raw_abs_max": float(torch.abs(raw_actions).max()),
        "clip_fraction": float(clip_fraction),
        "effective_phase_span_mean": float(phase_span.mean()),
    }


def main() -> int:
    args = _parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        raise NotADirectoryError(run)

    agent_cfg = yaml.safe_load((run / "params" / "agent.yaml").read_text(encoding="utf-8"))
    activation = str(agent_cfg["policy"]["activation"])
    action_clip = _load_action_clip(run)

    if args.checkpoint is not None:
        checkpoints = [args.checkpoint.resolve()]
    else:
        checkpoints = sorted(run.glob("model_*.pt"), key=_checkpoint_step)
        if not checkpoints:
            raise FileNotFoundError(f"No model_*.pt checkpoints in {run}")
        if not args.all_checkpoints:
            checkpoints = [checkpoints[-1]]

    print(
        f"run={run}\taction_clip={action_clip:g}\tmax_std={args.max_std:g}\t"
        f"max_clip_fraction={args.max_clip_fraction:.2%}"
    )
    reports = []
    for checkpoint in checkpoints:
        report = _audit_checkpoint(checkpoint, action_clip, activation, args.phase_samples)
        reports.append(report)
        print(
            f"{report['checkpoint']:>13s} "
            f"std_mean={report['std_mean']:.4f} std_max={report['std_max']:.4f} "
            f"raw_abs_mean={report['raw_abs_mean']:.4f} raw_abs_max={report['raw_abs_max']:.4f} "
            f"clip={report['clip_fraction']:.2%} "
            f"effective_phase_span_mean={report['effective_phase_span_mean']:.4f}"
        )

    latest = reports[-1]
    failures = []
    if float(latest["std_max"]) > args.max_std:
        failures.append(f"std_max {latest['std_max']:.4f} > {args.max_std:.4f}")
    if float(latest["clip_fraction"]) > args.max_clip_fraction:
        failures.append(
            f"clip_fraction {float(latest['clip_fraction']):.2%} > {args.max_clip_fraction:.2%}"
        )
    if failures:
        print("status=FAIL\t" + "; ".join(failures))
        return 2

    print("status=PASS_MODEL_GUARD")
    print("note=This model-only guard does not replace deterministic Isaac rollout/video validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
