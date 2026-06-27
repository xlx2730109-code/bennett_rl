# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate Bennett active-joint torque statistics at a fixed velocity command."""
#出关节电机均方根数据、图表等

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate Bennett torque statistics with a fixed velocity command.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to evaluate.")
parser.add_argument("--speed", type=float, default=0.5, help="Fixed forward velocity command in m/s.")
parser.add_argument("--yaw_rate", type=float, default=0.0, help="Fixed yaw velocity command in rad/s.")
parser.add_argument("--duration_s", type=float, default=20.0, help="Evaluation duration after warmup.")
parser.add_argument("--warmup_s", type=float, default=3.0, help="Warmup duration ignored in the final statistics.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory for CSV/JSON results.")
parser.add_argument("--output_prefix", type=str, default="bennett_torque_eval", help="Output file prefix.")
parser.add_argument(
    "--foot_contact_threshold",
    type=float,
    default=1.0,
    help="Contact force threshold used to decide whether a foot is in swing phase.",
)
parser.add_argument(
    "--plot",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Generate PPT-ready PNG plots after writing CSV/JSON.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


FOOT_BODY_NAMES = ["FL_1", "FR_1", "RL_1", "RR_1"]


def _set_fixed_velocity_command(base_env, speed: float, yaw_rate: float) -> None:
    """Keep the command term fixed even if the command manager resamples."""
    try:
        command_term = base_env.command_manager.get_term("base_velocity")
        command_term.vel_command_b[:, 0] = speed
        command_term.vel_command_b[:, 1] = 0.0
        command_term.vel_command_b[:, 2] = yaw_rate
    except Exception:
        pass


def _get_active_joint_ids_and_names(robot, preferred_names: list[str] | None = None) -> tuple[list[int], list[str]]:
    """Return actuated joint ids/names based on the robot actuator configuration."""
    active_indices: list[int] = []
    active_names: list[str] = []
    for actuator in robot.actuators.values():
        joint_indices = actuator.joint_indices
        if isinstance(joint_indices, slice):
            start, stop, step = joint_indices.indices(robot.num_joints)
            active_indices.extend(range(start, stop, step))
        elif hasattr(joint_indices, "detach"):
            active_indices.extend(joint_indices.detach().cpu().tolist())
        else:
            active_indices.extend(list(joint_indices))
        active_names.extend(actuator.joint_names)
    if preferred_names:
        active_by_name = {name: idx for idx, name in zip(active_indices, active_names)}
        if all(name in active_by_name for name in preferred_names):
            active_names = list(preferred_names)
            active_indices = [active_by_name[name] for name in active_names]
    return active_indices, active_names


def _get_torque_tensor(robot) -> torch.Tensor:
    """Return the best available joint torque tensor."""
    if hasattr(robot.data, "applied_torque"):
        return robot.data.applied_torque
    if hasattr(robot.data, "computed_torque"):
        return robot.data.computed_torque
    if hasattr(robot.data, "computed_torques"):
        return robot.data.computed_torques
    return robot.data.applied_efforts


def _scalar_stats(values: torch.Tensor) -> dict[str, float]:
    """Compute scalar statistics from a 1-D tensor."""
    values = values.reshape(-1)
    return {
        "mean_abs": torch.mean(torch.abs(values)).item(),
        "rms": torch.sqrt(torch.mean(torch.square(values))).item(),
        "p95_abs": torch.quantile(torch.abs(values), 0.95).item(),
        "p99_abs": torch.quantile(torch.abs(values), 0.99).item(),
        "max_abs": torch.max(torch.abs(values)).item(),
    }


def _height_stats(values: torch.Tensor) -> dict[str, float | int | None]:
    """Compute non-absolute height statistics from a 1-D tensor."""
    values = values.reshape(-1)
    if values.numel() == 0:
        return {
            "sample_count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "sample_count": int(values.numel()),
        "mean": torch.mean(values).item(),
        "p50": torch.quantile(values, 0.50).item(),
        "p95": torch.quantile(values, 0.95).item(),
        "p99": torch.quantile(values, 0.99).item(),
        "max": torch.max(values).item(),
    }


def _position_stats(values: torch.Tensor) -> dict[str, float | int | None]:
    """Compute signed position statistics from a 1-D tensor."""
    values = values.reshape(-1)
    if values.numel() == 0:
        return {
            "sample_count": 0,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
            "range": None,
        }
    value_min = torch.min(values).item()
    value_max = torch.max(values).item()
    return {
        "sample_count": int(values.numel()),
        "mean": torch.mean(values).item(),
        "p05": torch.quantile(values, 0.05).item(),
        "p50": torch.quantile(values, 0.50).item(),
        "p95": torch.quantile(values, 0.95).item(),
        "min": value_min,
        "max": value_max,
        "range": value_max - value_min,
    }


def _range_from_rows(rows: list[dict], key: str) -> float | None:
    """Return max-min for a nullable numeric field."""
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return max(values) - min(values)


def _max_from_rows(rows: list[dict], key: str) -> float | None:
    """Return max for a nullable numeric field."""
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return max(values)


def _min_from_rows(rows: list[dict], key: str) -> float | None:
    """Return min for a nullable numeric field."""
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return min(values)


def _safe_file_prefix(prefix: str) -> str:
    """Return a filename-safe prefix on Windows/Linux/macOS.

    Windows treats ':' as an NTFS alternate-data-stream separator, so a prefix
    such as '12:50' silently creates hidden streams instead of normal files.
    """
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", prefix.strip())
    safe = safe.strip(" .-_")
    return safe or "bennett_torque_eval"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.speed, args_cli.speed)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.yaw_rate, args_cli.yaw_rate)
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else Path(log_dir) / "eval_torque"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped
    step_dt = base_env.step_dt

    robot = base_env.scene["robot"]
    preferred_joint_names = None
    if hasattr(env_cfg.actions, "joint_pos") and hasattr(env_cfg.actions.joint_pos, "joint_names"):
        preferred_joint_names = env_cfg.actions.joint_pos.joint_names
    active_joint_ids, active_joint_names = _get_active_joint_ids_and_names(robot, preferred_joint_names)
    if not active_joint_ids:
        raise RuntimeError("No active joints were found from robot.actuators.")
    foot_body_ids, foot_body_names = robot.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    if len(foot_body_ids) != len(FOOT_BODY_NAMES):
        raise RuntimeError(f"Could not resolve all Bennett foot bodies: {FOOT_BODY_NAMES}")
    contact_sensor = base_env.scene.sensors["contact_forces"]
    contact_foot_body_ids, contact_foot_body_names = contact_sensor.find_bodies(FOOT_BODY_NAMES, preserve_order=True)
    if contact_foot_body_names != foot_body_names:
        raise RuntimeError(
            "Foot body order mismatch between robot and contact sensor: "
            f"robot={foot_body_names}, contact_sensor={contact_foot_body_names}"
        )

    rsl_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(rsl_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=rsl_env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))

    warmup_steps = int(args_cli.warmup_s / step_dt)
    eval_steps = int(args_cli.duration_s / step_dt)
    total_steps = warmup_steps + eval_steps

    _set_fixed_velocity_command(base_env, args_cli.speed, args_cli.yaw_rate)
    obs = rsl_env.get_observations()

    torque_samples = []
    vel_samples = []
    power_samples = []
    speed_error_samples = []
    yaw_error_samples = []
    base_height_samples = []
    foot_height_samples = []
    foot_x_samples = []
    foot_swing_mask_samples = []
    done_count = 0

    for step in range(total_steps):
        start_time = time.time()
        _set_fixed_velocity_command(base_env, args_cli.speed, args_cli.yaw_rate)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = rsl_env.step(actions)
            if policy_nn is not None and hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        _set_fixed_velocity_command(base_env, args_cli.speed, args_cli.yaw_rate)

        if step >= warmup_steps:
            raw_torques = _get_torque_tensor(robot)
            torques = raw_torques[:, active_joint_ids].detach().clone()
            joint_vels = robot.data.joint_vel[:, active_joint_ids].detach().clone()
            torque_samples.append(torques.cpu())
            vel_samples.append(joint_vels.cpu())
            power_samples.append(torch.abs(torques * joint_vels).cpu())

            speed_error = torch.abs(robot.data.root_lin_vel_b[:, 0] - args_cli.speed)
            yaw_error = torch.abs(robot.data.root_ang_vel_b[:, 2] - args_cli.yaw_rate)
            speed_error_samples.append(speed_error.detach().cpu())
            yaw_error_samples.append(yaw_error.detach().cpu())
            base_height_samples.append(robot.data.root_pos_w[:, 2].detach().cpu())

            foot_height = robot.data.body_pos_w[:, foot_body_ids, 2] - base_env.scene.env_origins[:, 2].unsqueeze(1)
            foot_rel_pos_w = robot.data.body_pos_w[:, foot_body_ids, :] - robot.data.root_pos_w[:, None, :]
            foot_yaw = yaw_quat(robot.data.root_quat_w)[:, None, :].expand(-1, foot_rel_pos_w.shape[1], -1)
            foot_rel_pos_b = quat_apply_inverse(
                foot_yaw.reshape(-1, 4), foot_rel_pos_w.reshape(-1, 3)
            ).reshape_as(foot_rel_pos_w)
            contact_force = contact_sensor.data.net_forces_w_history[:, :, contact_foot_body_ids, :].norm(dim=-1).max(dim=1)[0]
            foot_swing_mask = contact_force <= args_cli.foot_contact_threshold
            foot_height_samples.append(foot_height.detach().cpu())
            foot_x_samples.append(foot_rel_pos_b[:, :, 0].detach().cpu())
            foot_swing_mask_samples.append(foot_swing_mask.detach().cpu())
            done_count += int(torch.sum(dones).item())

        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    torque_data = torch.cat(torque_samples, dim=0)
    vel_data = torch.cat(vel_samples, dim=0)
    power_data = torch.cat(power_samples, dim=0)
    foot_height_data = torch.cat(foot_height_samples, dim=0)
    foot_x_data = torch.cat(foot_x_samples, dim=0)
    foot_swing_mask_data = torch.cat(foot_swing_mask_samples, dim=0)

    joint_rows = []
    for joint_id, joint_name in enumerate(active_joint_names):
        row = {"joint": joint_name}
        row.update({f"torque_{k}": v for k, v in _scalar_stats(torque_data[:, joint_id]).items()})
        row.update({f"velocity_{k}": v for k, v in _scalar_stats(vel_data[:, joint_id]).items()})
        row.update({f"power_{k}": v for k, v in _scalar_stats(power_data[:, joint_id]).items()})
        joint_rows.append(row)

    foot_rows = []
    for foot_id, foot_name in enumerate(foot_body_names):
        all_heights = foot_height_data[:, foot_id]
        swing_heights = all_heights[foot_swing_mask_data[:, foot_id]]
        all_x = foot_x_data[:, foot_id]
        swing_x = all_x[foot_swing_mask_data[:, foot_id]]
        stance_x = all_x[~foot_swing_mask_data[:, foot_id]]
        row = {"foot": foot_name}
        row.update({f"swing_height_{k}": v for k, v in _height_stats(swing_heights).items()})
        row.update({f"all_height_{k}": v for k, v in _height_stats(all_heights).items()})
        row.update({f"swing_x_{k}": v for k, v in _position_stats(swing_x).items()})
        row.update({f"stance_x_{k}": v for k, v in _position_stats(stance_x).items()})
        row.update({f"all_x_{k}": v for k, v in _position_stats(all_x).items()})
        foot_rows.append(row)

    overall = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "num_envs": args_cli.num_envs,
        "speed_mps": args_cli.speed,
        "yaw_rate_radps": args_cli.yaw_rate,
        "warmup_s": args_cli.warmup_s,
        "duration_s": args_cli.duration_s,
        "step_dt": step_dt,
        "eval_steps": eval_steps,
        "active_joint_names": active_joint_names,
        "foot_body_names": foot_body_names,
        "foot_contact_threshold": args_cli.foot_contact_threshold,
        "torque": _scalar_stats(torque_data),
        "velocity": _scalar_stats(vel_data),
        "power": _scalar_stats(power_data),
        "mean_abs_speed_error_mps": torch.mean(torch.cat(speed_error_samples)).item(),
        "mean_abs_yaw_error_radps": torch.mean(torch.cat(yaw_error_samples)).item(),
        "mean_base_height_m": torch.mean(torch.cat(base_height_samples)).item(),
        "foot_height": {
            "swing_height_p95_range_m": _range_from_rows(foot_rows, "swing_height_p95"),
            "swing_height_p99_range_m": _range_from_rows(foot_rows, "swing_height_p99"),
            "swing_height_max_range_m": _range_from_rows(foot_rows, "swing_height_max"),
            "swing_height_max_m": _max_from_rows(foot_rows, "swing_height_max"),
        },
        "foot_x": {
            "all_x_min_m": _min_from_rows(foot_rows, "all_x_min"),
            "all_x_max_m": _max_from_rows(foot_rows, "all_x_max"),
            "rear_all_x_max_m": _max_from_rows(foot_rows[2:], "all_x_max") if len(foot_rows) >= 4 else None,
            "rear_swing_x_max_m": _max_from_rows(foot_rows[2:], "swing_x_max") if len(foot_rows) >= 4 else None,
        },
        "done_count_during_eval": done_count,
    }

    output_prefix = _safe_file_prefix(args_cli.output_prefix)
    if output_prefix != args_cli.output_prefix:
        print(f"[WARN] Sanitized output_prefix from {args_cli.output_prefix!r} to {output_prefix!r}.")

    csv_path = output_dir / f"{output_prefix}_joints.csv"
    feet_csv_path = output_dir / f"{output_prefix}_feet.csv"
    json_path = output_dir / f"{output_prefix}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(joint_rows[0].keys()))
        writer.writeheader()
        writer.writerows(joint_rows)

    with feet_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(foot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(foot_rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"overall": overall, "joints": joint_rows, "feet": foot_rows}, f, indent=2)

    print(f"[INFO] Wrote joint CSV: {csv_path}")
    print(f"[INFO] Wrote foot CSV: {feet_csv_path}")
    print(f"[INFO] Wrote summary JSON: {json_path}")
    print(json.dumps(overall, indent=2))

    rsl_env.close()

    if args_cli.plot:
        from plot_bennett_torque_eval import (
            plot_design_reference,
            plot_foot_height,
            plot_foot_x,
            plot_joint_torque,
            plot_overall,
            plot_power,
        )

        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {"overall": overall, "joints": joint_rows, "feet": foot_rows}
        plot_overall(summary_payload, plot_dir, output_prefix)
        plot_joint_torque(joint_rows, plot_dir, output_prefix)
        plot_design_reference(joint_rows, plot_dir, output_prefix)
        plot_power(joint_rows, plot_dir, output_prefix)
        plot_foot_height(foot_rows, plot_dir, output_prefix)
        plot_foot_x(foot_rows, plot_dir, output_prefix)
        print(f"[INFO] Wrote plots to: {plot_dir}")
        for path in sorted(plot_dir.glob(f"{output_prefix}_*.png")):
            print(f"[INFO] Plot: {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
