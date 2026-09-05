"""Collect deterministic Bennett policy diagnostics from one Isaac Lab environment.

This script is intentionally separate from ``play.py`` and ``train.py``.  It
loads a checkpoint, runs a fixed command matrix, and writes policy-rate data
needed for Sim2Real analysis: observations, raw/applied actions, joint targets,
joint state/torque, base motion, foot motion, and contact force.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-BennettRL-Flat-Go2-10-Play-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_csv", type=Path, default=None)
parser.add_argument("--settle_s", type=float, default=1.5)
parser.add_argument("--command_s", type=float, default=5.0)
parser.add_argument("--recovery_s", type=float, default=1.5)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--eval_action_clip", type=float, default=3.5)
parser.add_argument(
    "--zero_actions",
    action="store_true",
    default=False,
    help="Step the environment with zero actions instead of policy output (mechanical baseline diagnostic).",
)
parser.add_argument(
    "--keep_terrain_visual_material",
    action="store_true",
    help="Keep the configured (possibly remote) terrain visual material. Diagnostics disable it by default.",
)
parser.add_argument(
    "--use_remote_ground_plane",
    action="store_true",
    help=(
        "Use the task's stock Nucleus-hosted ground-plane USD. By default diagnostics replace it "
        "with an equivalent locally generated flat mesh so collection also works offline."
    ),
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")
if args_cli.num_envs != 1:
    parser.error("diagnostic collection requires --num_envs 1")
if min(args_cli.settle_s, args_cli.command_s, args_cli.recovery_s) < 0.0:
    parser.error("scenario durations must be non-negative")
if args_cli.command_s <= 0.0 or args_cli.repeats < 1:
    parser.error("--command_s and --repeats must be positive")
if args_cli.eval_action_clip <= 0.0:
    parser.error("--eval_action_clip must be positive")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import bennett_rl.tasks  # noqa: F401


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
# Full omnidirectional matrix for free_gait3 (training ranges vx +-0.35,
# vy +-0.25, wz +-0.60): six directions x three speeds + a stand baseline.
def _matrix():
    speeds = {
        "slow": 0.25, "mid": 0.57, "fast": 0.91,   # fractions of the training range
    }
    dirs = {
        "forward":       lambda s: (0.35 * s, 0.0, 0.0),
        "backward":      lambda s: (-0.35 * s, 0.0, 0.0),
        "lateral_left":  lambda s: (0.0, 0.25 * s, 0.0),
        "lateral_right": lambda s: (0.0, -0.25 * s, 0.0),
        "yaw_left":      lambda s: (0.0, 0.0, 0.60 * s),
        "yaw_right":     lambda s: (0.0, 0.0, -0.60 * s),
    }
    rows = [("stand", (0.0, 0.0, 0.0))]
    for d_name, d_fn in dirs.items():
        for s_name, s_frac in speeds.items():
            rows.append((f"{d_name}_{s_name}", d_fn(s_frac)))
    return tuple(rows)


SCENARIOS = _matrix()


def _vector_columns(prefix: str, names: tuple[str, ...]) -> list[str]:
    return [f"{prefix}_{name}" for name in names]


CSV_HEADER = [
    "scenario",
    "repeat",
    "scenario_phase",
    "phase_time_s",
    "scenario_time_s",
    "step",
    "episode_step",
    "command_x",
    "command_y",
    "command_yaw",
    "reward",
    "done",
    "gait_phase",
    "root_height_m",
    "base_lin_vel_x",
    "base_lin_vel_y",
    "base_lin_vel_z",
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
    "projected_gravity_x",
    "projected_gravity_y",
    "projected_gravity_z",
    "mechanical_power_abs_w",
] + (
    _vector_columns("raw_action", JOINT_NAMES)
    + _vector_columns("applied_action", JOINT_NAMES)
    + _vector_columns("joint_target_rad", JOINT_NAMES)
    + _vector_columns("joint_pos_rad", JOINT_NAMES)
    + _vector_columns("joint_vel_rad_s", JOINT_NAMES)
    + _vector_columns("joint_acc_rad_s2", JOINT_NAMES)
    + _vector_columns("joint_torque_nm", JOINT_NAMES)
    + _vector_columns("foot_height_m", FOOT_NAMES)
    + _vector_columns("foot_vel_z_m_s", FOOT_NAMES)
    + _vector_columns("foot_contact_force_n", FOOT_NAMES)
    + _vector_columns("desired_contact", FOOT_NAMES)
)


def _tensor_values(value: torch.Tensor) -> list[float]:
    return value.detach().to("cpu", dtype=torch.float32).reshape(-1).tolist()


def _policy_observation(obs) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs
    if "policy" not in obs:
        raise KeyError(f"observation has no 'policy' group: {list(obs.keys())}")
    return obs["policy"]


def _set_velocity_command(raw_env, command: tuple[float, float, float]) -> None:
    term = raw_env.command_manager.get_term("base_velocity")
    value = torch.tensor(command, dtype=torch.float32, device=term.command.device).unsqueeze(0)
    term.vel_command_b[:] = value
    moving = torch.linalg.vector_norm(value, dim=1) >= 1.0e-6
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = torch.logical_not(moving)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False


def _phase_for_step(step: int, dt: float) -> tuple[str, float, tuple[float, float, float]]:
    settle_steps = int(round(args_cli.settle_s / dt))
    command_steps = int(round(args_cli.command_s / dt))
    if step < settle_steps:
        return "settle", step * dt, (0.0, 0.0, 0.0)
    if step < settle_steps + command_steps:
        return "command", (step - settle_steps) * dt, ()
    return "recovery", (step - settle_steps - command_steps) * dt, (0.0, 0.0, 0.0)


def _find_ids(robot, contact_sensor) -> tuple[list[int], list[int], list[int]]:
    joint_ids, resolved_joints = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
    body_ids, resolved_feet = robot.find_bodies(list(FOOT_NAMES), preserve_order=True)
    contact_body_ids, resolved_contact_feet = contact_sensor.find_bodies(list(FOOT_NAMES), preserve_order=True)
    if tuple(resolved_joints) != JOINT_NAMES:
        raise RuntimeError(f"joint order mismatch: {resolved_joints}")
    if tuple(resolved_feet) != FOOT_NAMES:
        raise RuntimeError(f"foot order mismatch: {resolved_feet}")
    if tuple(resolved_contact_feet) != FOOT_NAMES:
        raise RuntimeError(f"contact-sensor foot order mismatch: {resolved_contact_feet}")
    return (
        [int(index) for index in joint_ids],
        [int(index) for index in body_ids],
        [int(index) for index in contact_body_ids],
    )


def _safe_joint_target(robot, joint_ids: list[int]) -> torch.Tensor:
    target = getattr(robot.data, "joint_pos_target", None)
    if target is None:
        return torch.full((len(joint_ids),), float("nan"), device=robot.device)
    return target[0, joint_ids]


def _output_path() -> Path:
    if args_cli.output_csv is not None:
        return args_cli.output_csv.resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_task = args_cli.task.replace(":", "_").replace("/", "_")
    return (Path("outputs") / "bennett_policy_diagnostics" / f"{safe_task}_{stamp}.csv").resolve()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    # A stock ``terrain_type="plane"`` loads NVIDIA's remote
    # ``default_environment.usd``.  That makes a safety diagnostic depend on
    # network/cache state.  A one-cell procedural mesh is the same z=0 flat
    # collision surface and keeps the collection path fully local.
    if not args_cli.use_remote_ground_plane and env_cfg.scene.terrain.terrain_type == "plane":
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=args_cli.seed,
            curriculum=False,
            size=(8.0, 8.0),
            num_rows=1,
            num_cols=1,
            color_scheme="none",
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
        env_cfg.scene.terrain.max_init_terrain_level = None
    # The stock locomotion scene uses a Nucleus-hosted MDL.  It is visual only,
    # but can block an otherwise local/headless diagnostic when Nucleus is not
    # reachable.  Physics material, collision, and terrain geometry are kept.
    if not args_cli.keep_terrain_visual_material:
        env_cfg.scene.terrain.visual_material = None
    for event_name in ("base_external_force_torque", "push_robot"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(checkpoint)
    output_path = _output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing diagnostic CSV: {output_path}")

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = gym_env.unwrapped
    dt = float(raw_env.step_dt)
    total_steps = int(round((args_cli.settle_s + args_cli.command_s + args_cli.recovery_s) / dt))
    robot = raw_env.scene["robot"]
    contact_sensor = raw_env.scene.sensors["contact_forces"]
    joint_ids, foot_ids, contact_foot_ids = _find_ids(robot, contact_sensor)

    env = RslRlVecEnvWrapper(gym_env, clip_actions=args_cli.eval_action_clip)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Only OnPolicyRunner is supported, got {agent_cfg.class_name}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=raw_env.device)
    policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
    if policy_nn is None:
        raise RuntimeError("Unable to find the RSL-RL policy module")

    control_source = "zero_actions" if args_cli.zero_actions else "policy"
    print(f"[DIAGNOSTIC] task={args_cli.task} checkpoint={checkpoint}")
    print(f"[DIAGNOSTIC] control_source={control_source}")
    print(f"[DIAGNOSTIC] dt={dt:.6f}s scenarios={len(SCENARIOS)} repeats={args_cli.repeats}")
    print(f"[DIAGNOSTIC] output={output_path}")

    global_step = 0
    try:
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_HEADER)
            for repeat in range(args_cli.repeats):
                for scenario_name, scenario_command in SCENARIOS:
                    env.reset()
                    reset_done = torch.ones(1, dtype=torch.bool, device=raw_env.device)
                    policy_nn.reset(reset_done)
                    scenario_failed = False
                    for scenario_step in range(total_steps):
                        phase_name, phase_time, phase_command = _phase_for_step(scenario_step, dt)
                        command = scenario_command if phase_name == "command" else phase_command
                        _set_velocity_command(raw_env, command)
                        obs = env.get_observations()
                        policy_obs = _policy_observation(obs)
                        obs_dim = int(policy_obs.shape[-1])
                        if obs_dim not in (33, 50):
                            raise RuntimeError(f"Expected 33 (plain) or 50 (trot) policy observations, got {tuple(policy_obs.shape)}")
                        # Keep simulator stepping outside ``inference_mode``.
                        # Isaac Lab updates cached state tensors in-place during
                        # reset; turning those tensors into inference tensors
                        # makes the next scenario reset fail.
                        with torch.no_grad():
                            raw_action = (
                                torch.zeros((raw_env.num_envs, raw_env.action_manager.total_action_dim), device=raw_env.device)
                                if args_cli.zero_actions
                                else policy(obs)
                            )
                        next_obs, reward, done, _ = env.step(raw_action)
                        policy_nn.reset(done)

                        applied_action = raw_env.action_manager.action[0]
                        joint_pos = robot.data.joint_pos[0, joint_ids]
                        joint_vel = robot.data.joint_vel[0, joint_ids]
                        joint_acc = robot.data.joint_acc[0, joint_ids]
                        joint_torque = robot.data.applied_torque[0, joint_ids]
                        joint_target = _safe_joint_target(robot, joint_ids)
                        foot_height = robot.data.body_pos_w[0, foot_ids, 2] - raw_env.scene.env_origins[0, 2]
                        foot_vel_z = robot.data.body_lin_vel_w[0, foot_ids, 2]
                        foot_force = torch.linalg.vector_norm(
                            contact_sensor.data.net_forces_w[0, contact_foot_ids], dim=1
                        )
                        mechanical_power = torch.sum(torch.abs(joint_torque * joint_vel)).item()
                        if obs_dim == 50:
                            # trot: gait block lives at fixed offsets in the observation
                            phase_sin = float(policy_obs[0, 33])
                            phase_cos = float(policy_obs[0, 34])
                            gait_phase = math.atan2(phase_sin, phase_cos) / (2.0 * math.pi)
                            if gait_phase < 0.0:
                                gait_phase += 1.0
                            desired_contacts = _tensor_values(policy_obs[0, 43:47])
                        else:
                            # plain (free_gait): no clock -> no phase; derive the
                            # contact state from the measured vertical foot force.
                            gait_phase = float("nan")
                            desired_contacts = [
                                1.0 if f > 5.0 else 0.0 for f in foot_force.detach().tolist()
                            ]

                        row = [
                            scenario_name,
                            repeat,
                            phase_name,
                            phase_time,
                            scenario_step * dt,
                            global_step,
                            int(raw_env.episode_length_buf[0].item()),
                            *command,
                            float(reward[0].item()),
                            int(done[0].item()),
                            gait_phase,
                            float(robot.data.root_pos_w[0, 2].item()),
                            *_tensor_values(robot.data.root_lin_vel_b[0]),
                            *_tensor_values(robot.data.root_ang_vel_b[0]),
                            *_tensor_values(robot.data.projected_gravity_b[0]),
                            mechanical_power,
                            *_tensor_values(raw_action[0]),
                            *_tensor_values(applied_action),
                            *_tensor_values(joint_target),
                            *_tensor_values(joint_pos),
                            *_tensor_values(joint_vel),
                            *_tensor_values(joint_acc),
                            *_tensor_values(joint_torque),
                            *_tensor_values(foot_height),
                            *_tensor_values(foot_vel_z),
                            *_tensor_values(foot_force),
                            *desired_contacts,
                        ]
                        if len(row) != len(CSV_HEADER):
                            raise RuntimeError(f"CSV row length {len(row)} != header length {len(CSV_HEADER)}")
                        writer.writerow(row)
                        global_step += 1
                        if bool(done[0].item()):
                            scenario_failed = True
                            break
                    stream.flush()
                    status = "terminated" if scenario_failed else "complete"
                    print(f"[DIAGNOSTIC] repeat={repeat} scenario={scenario_name} status={status}")
                    if scenario_failed:
                        continue
    finally:
        env.close()
    print(f"[DIAGNOSTIC] wrote {global_step} rows to {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
