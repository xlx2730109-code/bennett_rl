# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
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
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: keyboard_controller.advance().unsqueeze(0).to(env.device),
        )
        print("[INFO] Enabled keyboard command control for policy velocity_commands:")
        print(f"       x={x_sensitivity:.3f} m/s, y={y_sensitivity:.3f} m/s, yaw={yaw_sensitivity:.3f} rad/s")
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

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
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
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
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

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
