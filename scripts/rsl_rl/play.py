# RSL-RL 播放/导出入口

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from endpoint_trajectory_export import export_endpoint_trajectory  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Length of the recorded video in environment steps. Overrides --video_seconds when set.",
)
parser.add_argument(
    "--video_seconds",
    type=float,
    default=30.0,
    help="Length of each recorded video in simulated seconds when --video_length is not set. Defaults to 30 seconds.",
)
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=None,
    help="Optional video output directory. Defaults to <checkpoint_dir>/videos/play.",
)
parser.add_argument(
    "--disable_export",
    action="store_true",
    default=False,
    help="Do not export or overwrite policy.pt and policy.onnx during playback.",
)
parser.add_argument(
    "--endpoint_trail",
    action="store_true",
    default=False,
    help="Draw the selected robot body as a green endpoint marker with a position trail.",
)
parser.add_argument(
    "--endpoint_body",
    type=str,
    default=None,
    help="Exact robot body used by --endpoint_trail. Defaults to RR_foot for V4, otherwise RR_2.",
)
parser.add_argument(
    "--endpoint_bodies",
    type=str,
    nargs="+",
    default=None,
    help="One or more exact robot bodies to draw with separate coloured trails.",
)
parser.add_argument(
    "--endpoint_trail_points",
    type=int,
    default=250,
    help="Maximum number of saved endpoint trail points.",
)
parser.add_argument(
    "--endpoint_trail_stride",
    type=int,
    default=None,
    help="Save one trail point every N environment steps. Defaults to approximately every 0.016 seconds.",
)
parser.add_argument(
    "--endpoint_radius",
    type=float,
    default=0.016,
    help="Radius in metres of the current green endpoint marker.",
)
parser.add_argument(
    "--endpoint_trail_radius",
    type=float,
    default=0.002,
    help="Radius in metres of each green trail marker.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--keyboard", action="store_true", default=False, help="Enable keyboard control.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--play_episode_length_s",
    type=float,
    default=80.0,
    help="Override episode timeout length in seconds for play only. Defaults to 80 seconds for all play tasks.",
)
parser.add_argument(
    "--speed",
    type=float,
    default=None,
    help="Fix base_velocity lin_vel_x command to this speed during play, if the task has that command.",
)
parser.add_argument(
    "--keyboard_x_sensitivity",
    type=float,
    default=None,
    help="Keyboard forward/backward velocity sensitivity in m/s. Defaults to the command range or 0.5.",
)
parser.add_argument(
    "--keyboard_y_sensitivity",
    type=float,
    default=None,
    help="Keyboard lateral velocity sensitivity in m/s. Defaults to the command range or 0.3.",
)
parser.add_argument(
    "--keyboard_yaw_sensitivity",
    type=float,
    default=None,
    help="Keyboard yaw-rate sensitivity in rad/s. Defaults to the command range or 0.8.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task == "Isaac-Velocity-Rough-Bennett_Test4-v0":
    args_cli.task = "Isaac-Velocity-Rough-Bennett_Test4-Play-v0"
    print("[INFO] Redirected Bennett Test4 rough play to Isaac-Velocity-Rough-Bennett_Test4-Play-v0.")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import bennett_rl.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    def _range_abs_max(value_range, fallback: float) -> float:
        value = max(abs(float(value_range[0])), abs(float(value_range[1])))
        return value if value > 1.0e-6 else fallback

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if not hasattr(env_cfg, "episode_length_s"):
        raise ValueError("--play_episode_length_s requires an environment config with episode_length_s.")
    env_cfg.episode_length_s = args_cli.play_episode_length_s
    print(f"[INFO] Fixed play episode length to {args_cli.play_episode_length_s:.2f} s.")
    if args_cli.speed is not None:
        if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "base_velocity"):
            raise ValueError("--speed requires an environment config with commands.base_velocity.")
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.speed, args_cli.speed)
        print(f"[INFO] Fixed play command lin_vel_x to {args_cli.speed:.3f} m/s.")

    keyboard_controller = None
    if args_cli.keyboard:
        if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "base_velocity"):
            raise ValueError("--keyboard requires an environment config with commands.base_velocity.")
        if not hasattr(env_cfg, "observations") or not hasattr(env_cfg.observations.policy, "velocity_commands"):
            raise ValueError("--keyboard requires a policy observation named velocity_commands.")

        env_cfg.scene.num_envs = 1
        env_cfg.commands.base_velocity.debug_vis = False
        command_ranges = env_cfg.commands.base_velocity.ranges
        x_sensitivity = (
            args_cli.keyboard_x_sensitivity
            if args_cli.keyboard_x_sensitivity is not None
            else _range_abs_max(command_ranges.lin_vel_x, 0.5)
        )
        y_sensitivity = (
            args_cli.keyboard_y_sensitivity
            if args_cli.keyboard_y_sensitivity is not None
            else _range_abs_max(command_ranges.lin_vel_y, 0.3)
        )
        yaw_sensitivity = (
            args_cli.keyboard_yaw_sensitivity
            if args_cli.keyboard_yaw_sensitivity is not None
            else _range_abs_max(command_ranges.ang_vel_z, 0.8)
        )
        keyboard_controller = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=x_sensitivity,
                v_y_sensitivity=y_sensitivity,
                omega_z_sensitivity=yaw_sensitivity,
            )
        )
        # Keep all command-conditioned observations consistent.  Crawl policies use not only
        # velocity_commands, but also phase/contact/gait observations that read commands.base_velocity.
        # The actual command tensor is synchronized into the command manager inside the play loop.
        for key_name in ("PAGE_UP", "PAGEUP", "PGUP"):
            keyboard_controller._INPUT_KEY_MAPPING[key_name] = torch.tensor(
                [1.0, 0.0, 0.0], dtype=torch.float32
            ).numpy() * x_sensitivity
        for key_name in ("PAGE_DOWN", "PAGEDOWN", "PGDN"):
            keyboard_controller._INPUT_KEY_MAPPING[key_name] = torch.tensor(
                [-1.0, 0.0, 0.0], dtype=torch.float32
            ).numpy() * x_sensitivity
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        print("[INFO] Enabled keyboard command control for commands.base_velocity:")
        print(f"       x={x_sensitivity:.3f} m/s, y={y_sensitivity:.3f} m/s, yaw={yaw_sensitivity:.3f} rad/s")
        print("       forward/backward: PgUp/PgDn, Up/Down, or Numpad 8/2")
        print(keyboard_controller)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # #sim to sim 关节顺序
    # # ===================== 加在这里 =====================
    # print("\n" + "="*60)
    # print("【终极映射字典】请把下面这个完整的列表复制发过来：")
    # try:
    #     print(env.unwrapped.scene["robot"].data.joint_names)
    # except:
    #     pass
    # print("="*60 + "\n")
    # simulation_app.close() # 安全关闭 Isaac Lab
    # exit(0)
    # # ====================================================

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # fetch simulation step dt for video FPS calculation and real-time throttling
    dt = env.unwrapped.step_dt

    raw_env = env.unwrapped

    endpoint_robot = None
    endpoint_markers: list[VisualizationMarkers] = []
    endpoint_body_ids: list[int] = []
    endpoint_trails: list[list[torch.Tensor]] = []
    endpoint_trajectory_samples: list[tuple[float, int, torch.Tensor]] = []
    endpoint_trajectory_segment = 0
    endpoint_output_dir = log_dir
    endpoint_trail_stride = args_cli.endpoint_trail_stride
    if args_cli.endpoint_trail:
        if raw_env.num_envs != 1:
            raise ValueError("--endpoint_trail requires exactly one environment. Use --num_envs 1.")
        if args_cli.endpoint_trail_points < 1:
            raise ValueError("--endpoint_trail_points must be at least 1.")
        if endpoint_trail_stride is None:
            endpoint_trail_stride = max(1, int(round(0.016 / dt)))
        if endpoint_trail_stride < 1:
            raise ValueError("--endpoint_trail_stride must be at least 1.")
        if args_cli.endpoint_radius <= 0.0 or args_cli.endpoint_trail_radius <= 0.0:
            raise ValueError("Endpoint marker radii must be greater than zero.")

        endpoint_robot = raw_env.scene["robot"]
        if args_cli.endpoint_body is not None and args_cli.endpoint_bodies is not None:
            raise ValueError("Use either --endpoint_body or --endpoint_bodies, not both.")
        endpoint_body_names = args_cli.endpoint_bodies
        if endpoint_body_names is None:
            endpoint_body_name = args_cli.endpoint_body
            if endpoint_body_name is None:
                endpoint_body_name = "RR_foot" if "V4-50Hz" in args_cli.task else "RR_2"
            endpoint_body_names = [endpoint_body_name]

        resolved_body_ids, resolved_endpoint_body_names = endpoint_robot.find_bodies(
            endpoint_body_names, preserve_order=True
        )
        if len(resolved_body_ids) != len(endpoint_body_names):
            raise RuntimeError(
                f"Failed to resolve all endpoint bodies {endpoint_body_names}: {resolved_endpoint_body_names}"
            )
        endpoint_body_ids = list(resolved_body_ids)
        endpoint_trails = [[] for _ in endpoint_body_ids]

        trail_colours = (
            (0.0, 1.0, 0.0),
            (0.0, 0.55, 1.0),
            (1.0, 0.75, 0.0),
            (1.0, 0.0, 0.65),
        )
        for body_index, _ in enumerate(endpoint_body_ids):
            colour = trail_colours[body_index % len(trail_colours)]
            marker_cfg = POSITION_GOAL_MARKER_CFG.copy()
            marker_cfg.prim_path = f"/Visuals/RslRlEndpointTrail/Body{body_index}"
            marker_cfg.markers["target_far"].radius = args_cli.endpoint_trail_radius
            marker_cfg.markers["target_far"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=colour)
            marker_cfg.markers["target_near"].radius = args_cli.endpoint_radius
            marker_cfg.markers["target_near"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=colour)
            endpoint_markers.append(VisualizationMarkers(marker_cfg))
        print(
            f"[INFO] Enabled endpoint trails: bodies={resolved_endpoint_body_names}, "
            f"points={args_cli.endpoint_trail_points}, stride={endpoint_trail_stride}."
        )

    # wrap for video recording
    if args_cli.video:
        if args_cli.video_length is None:
            # RecordVideo omits the initial wrapper step, so add one step to obtain
            # the requested number of output frames (for example, 1500 frames at 50 Hz for 30 s).
            args_cli.video_length = max(1, int(round(args_cli.video_seconds / dt)) + 1)
            print(
                f"[INFO] Auto video_length={args_cli.video_length} steps "
                f"for {args_cli.video_seconds:g}s videos (step_dt={dt})."
            )
        endpoint_output_dir = (
            os.path.abspath(args_cli.video_output_dir)
            if args_cli.video_output_dir is not None
            else os.path.join(
                log_dir,
                "videos",
                f"play_with_trail_{int(round(1.0 / dt))}hz" if args_cli.endpoint_trail else "play",
            )
        )
        video_kwargs = {
            "video_folder": endpoint_output_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "fps": int(1 / dt),
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    def _sync_keyboard_base_velocity_command():
        if keyboard_controller is None:
            return
        command_term = raw_env.command_manager.get_term("base_velocity")
        command = keyboard_controller.advance().to(command_term.command.device).unsqueeze(0)
        if command_term.command.shape[0] != 1:
            raise RuntimeError("--keyboard play expects exactly one environment.")
        command_term.vel_command_b[:] = command
        moving = torch.linalg.norm(command[:, :2], dim=1) > 1.0e-6
        moving = torch.logical_or(moving, torch.abs(command[:, 2]) > 1.0e-6)
        if hasattr(command_term, "is_standing_env"):
            command_term.is_standing_env[:] = torch.logical_not(moving)
        if hasattr(command_term, "is_heading_env"):
            command_term.is_heading_env[:] = False

    def _update_endpoint_trail(step_index: int, dones: torch.Tensor | None = None):
        nonlocal endpoint_trajectory_segment
        if endpoint_robot is None or not endpoint_markers or not endpoint_body_ids:
            return
        if dones is not None and bool(torch.any(dones).item()):
            for endpoint_trail in endpoint_trails:
                endpoint_trail.clear()
            endpoint_trajectory_segment += 1

        endpoint_positions_w = endpoint_robot.data.body_pos_w[0, endpoint_body_ids].clone()
        endpoint_trajectory_samples.append(
            (step_index * dt, endpoint_trajectory_segment, endpoint_positions_w.detach().cpu())
        )

        for endpoint_position_w, endpoint_marker, endpoint_trail in zip(
            endpoint_positions_w, endpoint_markers, endpoint_trails, strict=True
        ):
            endpoint_pos_w = endpoint_position_w.unsqueeze(0)
            if step_index % endpoint_trail_stride == 0:
                endpoint_trail.append(endpoint_pos_w)
                del endpoint_trail[: max(0, len(endpoint_trail) - args_cli.endpoint_trail_points)]

            marker_positions = torch.cat([*endpoint_trail, endpoint_pos_w], dim=0)
            marker_indices = torch.zeros(
                marker_positions.shape[0], device=endpoint_robot.device, dtype=torch.long
            )
            marker_indices[-1] = 1
            endpoint_marker.visualize(marker_positions, marker_indices=marker_indices)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    if not args_cli.disable_export:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # reset environment
    _sync_keyboard_base_velocity_command()
    obs = env.get_observations()
    timestep = 0
    _update_endpoint_trail(timestep)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            _sync_keyboard_base_velocity_command()
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            _update_endpoint_trail(timestep + 1, dones)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        # Note: always run in real-time when recording video to ensure correct playback speed
        sleep_time = dt - (time.time() - start_time)
        if (args_cli.real_time or args_cli.video) and sleep_time > 0:
            time.sleep(sleep_time)

    if endpoint_trajectory_samples:
        csv_path, html_path = export_endpoint_trajectory(
            endpoint_output_dir, resolved_endpoint_body_names, endpoint_trajectory_samples
        )
        print(f"[INFO] Exported endpoint trajectory CSV: {csv_path}")
        print(f"[INFO] Exported interactive 3-D trajectory: {html_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
