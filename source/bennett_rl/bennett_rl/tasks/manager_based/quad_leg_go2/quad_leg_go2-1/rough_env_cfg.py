# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from bennett_rl.assets.robots.bennett import BENNETT_CFG_V1  # isort: skip

from . import mdp as go2_mdp


ACTIVE_JOINTS = [
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
]

FOOT_BODIES = ".*_1"
BASE_BODY = "base"


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Use Bennett hardware/asset while keeping this task as the Go2-style reward experiment.
        # old: self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_cfg = BENNETT_CFG_V1.copy()
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
        robot_cfg.actuators["base_legs"].effort_limit = 10.0
        robot_cfg.actuators["base_legs"].saturation_effort = 12.0
        robot_cfg.actuators["base_legs"].stiffness = 40.0
        robot_cfg.actuators["base_legs"].damping = 2.5
        self.scene.robot = robot_cfg
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.joint_names = ACTIVE_JOINTS
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.preserve_order = True

        # Low-speed straight-line Bennett bring-up.
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Sim2Real-oriented 33D policy observation: no direct base linear velocity.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=ACTIVE_JOINTS, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=ACTIVE_JOINTS, preserve_order=True
        )

        # event
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.push_robot = None
        # old: self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params = {
            # old: "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        # old: self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = FOOT_BODIES
        self.rewards.feet_air_time.params["threshold"] = 0.22
        self.rewards.feet_air_time.weight = 0.35
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.12
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.3
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.action_rate_l2.weight = -0.04
        self.rewards.base_height = RewardTermCfg(
            func=go2_mdp.base_height_exp,
            weight=0.6,
            params={
                "target_height": 0.32,
                "sigma": 0.06,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY
        self.terminations.low_base_height = DoneTerm(
            func=go2_mdp.base_height_below,
            params={"minimum_height": 0.22, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.excessive_tilt = DoneTerm(
            func=go2_mdp.base_tilt_over,
            params={"max_projected_gravity_xy": math.sin(0.75), "asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# python .\scripts\rsl_rl\play.py --task=logs\rsl_rl\quad_leg_go2 --video  --checkpoint
