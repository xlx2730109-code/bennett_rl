# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V2

from .mdp import rewards as trot_rewards


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

FOOT_BODIES = ["FL_1", "FR_1", "RL_1", "RR_1"]
TROT_CYCLE_TIME = 0.56
TROT_DUTY_FACTOR = 0.56
TROT_FRONT_FOOT_HALF_WIDTH = 0.15
TROT_REAR_FOOT_HALF_WIDTH = 0.15
TROT_STRIDE_LENGTH = 0.10
TROT_TARGET_X_BY_FOOT = (0.23, 0.23, -0.23, -0.23)


@configclass
class BennettTrot1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        robot_cfg = BENNETT_CFG_V2.copy()
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot = robot_cfg

        self.decimation = 4
        self.actions.joint_pos.joint_names = ACTIVE_JOINTS
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.preserve_order = False

        self.observations.policy.base_lin_vel = None
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS)
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS)
        self.observations.policy.gait_phase = ObsTerm(
            func=trot_rewards.gait_phase_sin_cos,
            params={"cycle_time": TROT_CYCLE_TIME},
        )

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self._configure_events()
        self._configure_commands()
        self._configure_rewards()
        self._configure_terminations()

    def _configure_events(self):
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.push_robot = None
        self.events.base_com = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.0, 0.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    def _configure_commands(self):
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 0.8
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 2.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

    def _configure_rewards(self):
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_1"
        self.rewards.feet_air_time.params["threshold"] = 0.2
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_2", ".*_thigh", ".*_calf"]),
                "threshold": 1.0,
            },
        )
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.dof_pos_limits.weight = 0.0
        self.rewards.track_straight_line_y_l2 = RewTerm(
            func=trot_rewards.track_straight_line_y_l2,
            weight=-0.6,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot"),
                "max_lateral_error": 0.45,
            },
        )
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-5.0,
            params={"target_height": 0.28},
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_1"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_1"),
            },
        )
        self.rewards.trot_contact_phase = RewTerm(
            func=trot_rewards.trot_contact_phase,
            weight=0.6,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "cycle_time": TROT_CYCLE_TIME,
                "duty_factor": TROT_DUTY_FACTOR,
                "contact_force_threshold": 1.0,
            },
        )
        self.rewards.swing_foot_clearance = RewTerm(
            func=trot_rewards.swing_foot_clearance,
            weight=0.25,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "cycle_time": TROT_CYCLE_TIME,
                "duty_factor": TROT_DUTY_FACTOR,
                "target_height": 0.04,
                "contact_force_threshold": 1.0,
            },
        )
        self.rewards.trot_swing_foot_x_trajectory_l2 = RewTerm(
            func=trot_rewards.trot_swing_foot_x_trajectory_l2,
            weight=-0.35,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "cycle_time": TROT_CYCLE_TIME,
                "duty_factor": TROT_DUTY_FACTOR,
                "stride_length": TROT_STRIDE_LENGTH,
                "target_x_by_foot": TROT_TARGET_X_BY_FOOT,
            },
        )
        self.rewards.foot_lateral_position_l2 = RewTerm(
            func=trot_rewards.foot_lateral_position_l2,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "target_abs_y": TROT_FRONT_FOOT_HALF_WIDTH,
                "target_y_by_foot": (
                    TROT_FRONT_FOOT_HALF_WIDTH,
                    -TROT_FRONT_FOOT_HALF_WIDTH,
                    TROT_REAR_FOOT_HALF_WIDTH,
                    -TROT_REAR_FOOT_HALF_WIDTH,
                ),
            },
        )

    def _configure_terminations(self):
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        self.terminations.base_contact.params["threshold"] = 1.0
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.16},
        )


@configclass
class BennettTrot1FlatEnvCfg_PLAY(BennettTrot1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None




# python .\scripts\zero_agent.py --task Isaac-BennettRL-Flat-Trot1-v0