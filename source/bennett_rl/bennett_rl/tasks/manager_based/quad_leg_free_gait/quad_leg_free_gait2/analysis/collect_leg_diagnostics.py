"""Collect deterministic per-leg FreeGait diagnostics from one Isaac Lab environment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-BennettRL-Flat-QuadLeg-FreeGait2-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output_csv", type=Path, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--settle_s", type=float, default=0.5)
parser.add_argument("--command_s", type=float, default=2.0)
parser.add_argument("--recovery_s", type=float, default=0.5)
parser.add_argument("--valid_lift_height", type=float, default=0.020)
parser.add_argument("--contact_threshold", type=float, default=1.0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.command_s <= 0.0:
    parser.error("--command_s must be positive")
if min(args_cli.settle_s, args_cli.recovery_s) < 0.0:
    parser.error("settle and recovery durations must be non-negative")
if args_cli.valid_lift_height <= 0.0 or args_cli.contact_threshold <= 0.0:
    parser.error("lift height and contact threshold must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import bennett_rl.tasks  # noqa: F401


FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FOOT_LABELS = ("FL", "FR", "RL", "RR")
SCENARIOS = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward", (0.18, 0.0, 0.0)),
    ("backward", (-0.18, 0.0, 0.0)),
    ("lateral_left", (0.0, 0.12, 0.0)),
    ("lateral_right", (0.0, -0.12, 0.0)),
    ("yaw_left", (0.0, 0.0, 0.35)),
    ("yaw_right", (0.0, 0.0, -0.35)),
)

CSV_HEADER = [
    "scenario",
    "phase",
    "phase_time_s",
    "command_x",
    "command_y",
    "command_yaw",
    "done",
    "base_vel_x",
    "base_vel_y",
    "base_yaw_rate",
]
for label in FOOT_LABELS:
    CSV_HEADER.extend(
        [
            f"{label}_relative_height_m",
            f"{label}_contact",
            f"{label}_contact_force_n",
            f"{label}_stance_slide_m_s",
            f"{label}_current_contact_time_s",
            f"{label}_current_air_time_s",
            f"{label}_valid_lift_event",
        ]
    )


def _set_command(raw_env, command: tuple[float, float, float]) -> None:
    term = raw_env.command_manager.get_term("base_velocity")
    value = torch.tensor(command, dtype=torch.float32, device=term.command.device).unsqueeze(0)
    term.vel_command_b[:] = value
    moving = torch.linalg.vector_norm(value, dim=1) >= 1.0e-6
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = torch.logical_not(moving)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False


def _phase(
    step: int, dt: float, command: tuple[float, float, float]
) -> tuple[str, float, tuple[float, float, float]]:
    settle_steps = int(round(args_cli.settle_s / dt))
    command_steps = int(round(args_cli.command_s / dt))
    if step < settle_steps:
        return "settle", step * dt, (0.0, 0.0, 0.0)
    if step < settle_steps + command_steps:
        return "command", (step - settle_steps) * dt, command
    return "recovery", (step - settle_steps - command_steps) * dt, (0.0, 0.0, 0.0)


def _summary_rows(records: list[dict[str, float | int | str]]) -> list[list[float | int | str]]:
    rows: list[list[float | int | str]] = []
    for scenario, _ in SCENARIOS:
        scenario_records = [
            record for record in records if record["scenario"] == scenario and record["phase"] == "command"
        ]
        for label in FOOT_LABELS:
            heights = np.asarray(
                [record[f"{label}_relative_height_m"] for record in scenario_records], dtype=float
            )
            contacts = np.asarray([record[f"{label}_contact"] for record in scenario_records], dtype=float)
            slides = np.asarray(
                [
                    record[f"{label}_stance_slide_m_s"]
                    for record in scenario_records
                    if record[f"{label}_contact"]
                ],
                dtype=float,
            )
            contact_times = np.asarray(
                [record[f"{label}_current_contact_time_s"] for record in scenario_records], dtype=float
            )
            lift_events = int(
                sum(int(record[f"{label}_valid_lift_event"]) for record in scenario_records)
            )
            rows.append(
                [
                    scenario,
                    label,
                    lift_events,
                    float(np.max(heights)) if heights.size else float("nan"),
                    float(np.percentile(heights, 95)) if heights.size else float("nan"),
                    float(np.mean(contacts)) if contacts.size else float("nan"),
                    float(np.max(contact_times)) if contact_times.size else float("nan"),
                    float(np.percentile(slides, 95)) if slides.size else 0.0,
                ]
            )
    return rows


def main() -> None:
    output_path = args_cli.output_csv.resolve()
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path} or {summary_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args_cli.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.log_dir = str(checkpoint.parent)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = gym_env.unwrapped
    robot = raw_env.scene["robot"]
    sensor = raw_env.scene.sensors["contact_forces"]
    foot_ids, resolved_feet = robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
    sensor_foot_ids, resolved_sensor_feet = sensor.find_bodies(list(FOOT_NAMES), preserve_order=True)
    if tuple(resolved_feet) != FOOT_NAMES or tuple(resolved_sensor_feet) != FOOT_NAMES:
        raise RuntimeError(
            f"Foot order mismatch: robot={resolved_feet}, sensor={resolved_sensor_feet}"
        )

    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=raw_env.device)
    policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
    if policy_nn is None:
        raise RuntimeError("Unable to find the RSL-RL policy module")

    dt = float(raw_env.step_dt)
    total_steps = int(round((args_cli.settle_s + args_cli.command_s + args_cli.recovery_s) / dt))
    records: list[dict[str, float | int | str]] = []
    try:
        for scenario, scenario_command in SCENARIOS:
            env.reset()
            policy_nn.reset(torch.ones(1, dtype=torch.bool, device=raw_env.device))
            previous_valid_lift = torch.zeros(len(FOOT_NAMES), dtype=torch.bool, device=raw_env.device)
            for step in range(total_steps):
                phase, phase_time, command = _phase(step, dt, scenario_command)
                _set_command(raw_env, command)
                obs = env.get_observations()
                with torch.no_grad():
                    action = policy(obs)
                _, _, done, _ = env.step(action)
                policy_nn.reset(done)

                foot_height = robot.data.body_pos_w[0, foot_ids, 2]
                foot_velocity_xy = robot.data.body_lin_vel_w[0, foot_ids, :2]
                force = torch.linalg.vector_norm(
                    sensor.data.net_forces_w[0, sensor_foot_ids], dim=1
                )
                contact = force > float(args_cli.contact_threshold)
                contact_float = contact.to(torch.float32)
                stance_height = torch.sum(foot_height * contact_float) / contact_float.sum().clamp_min(1.0)
                relative_height = foot_height - stance_height
                valid_lift = (
                    torch.logical_not(contact)
                    & (relative_height >= float(args_cli.valid_lift_height))
                    & contact.any()
                    & (phase == "command")
                )
                lift_event = valid_lift & torch.logical_not(previous_valid_lift)
                previous_valid_lift = valid_lift
                contact_time = sensor.data.current_contact_time[0, sensor_foot_ids]
                air_time = sensor.data.current_air_time[0, sensor_foot_ids]
                slide = torch.linalg.vector_norm(foot_velocity_xy, dim=1) * contact_float

                record: dict[str, float | int | str] = {
                    "scenario": scenario,
                    "phase": phase,
                    "phase_time_s": phase_time,
                    "command_x": command[0],
                    "command_y": command[1],
                    "command_yaw": command[2],
                    "done": int(done[0].item()),
                    "base_vel_x": float(robot.data.root_lin_vel_b[0, 0].item()),
                    "base_vel_y": float(robot.data.root_lin_vel_b[0, 1].item()),
                    "base_yaw_rate": float(robot.data.root_ang_vel_b[0, 2].item()),
                }
                for index, label in enumerate(FOOT_LABELS):
                    record.update(
                        {
                            f"{label}_relative_height_m": float(relative_height[index].item()),
                            f"{label}_contact": int(contact[index].item()),
                            f"{label}_contact_force_n": float(force[index].item()),
                            f"{label}_stance_slide_m_s": float(slide[index].item()),
                            f"{label}_current_contact_time_s": float(contact_time[index].item()),
                            f"{label}_current_air_time_s": float(air_time[index].item()),
                            f"{label}_valid_lift_event": int(lift_event[index].item()),
                        }
                    )
                records.append(record)
    finally:
        env.close()

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(records)

    summary_header = [
        "scenario",
        "leg",
        "valid_lift_events",
        "max_relative_height_m",
        "p95_relative_height_m",
        "contact_ratio",
        "max_continuous_contact_s",
        "p95_stance_slide_m_s",
    ]
    summary_rows = _summary_rows(records)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(summary_header)
        writer.writerows(summary_rows)

    print(f"[LEG-DIAGNOSTIC] samples={len(records)} raw={output_path}")
    print(f"[LEG-DIAGNOSTIC] summary={summary_path}")
    for row in summary_rows:
        scenario, leg, events, max_height, _, contact_ratio, max_contact, slide_p95 = row
        if scenario != "stand":
            print(
                f"[LEG] {scenario:14s} {leg} events={events:2d} "
                f"max_h={100.0 * max_height:5.2f}cm contact={contact_ratio:5.1%} "
                f"max_contact={max_contact:4.2f}s slide_p95={slide_p95:5.3f}m/s"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
