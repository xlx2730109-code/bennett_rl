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

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V1

from .mdp import rewards as walk_rewards


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
WALK_CYCLE_TIME = 2.2
WALK_DUTY_FACTOR = 0.76
WALK_SWING_OFFSETS = (0.25, 0.75, 0.0, 0.5)
WALK_TARGET_AIR_TIME = WALK_CYCLE_TIME * (1.0 - WALK_DUTY_FACTOR)


@configclass
class BennettWalk3FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        robot_cfg = BENNETT_CFG_V1.copy()
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.init_state.pos = (0.0, 0.0, 0.35)
        robot_cfg.actuators["base_legs"].effort_limit = 10.0
        robot_cfg.actuators["base_legs"].saturation_effort = 12.0
        robot_cfg.actuators["base_legs"].stiffness = 36.0
        robot_cfg.actuators["base_legs"].damping = 2.8
        self.scene.robot = robot_cfg

        self.decimation = 4
        self.actions.joint_pos.joint_names = ACTIVE_JOINTS
        self.actions.joint_pos.scale = 0.22
        self.actions.joint_pos.use_default_offset = True
        self.actions.joint_pos.preserve_order = False

        self.observations.policy.base_lin_vel = None
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS)
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS)
        self.observations.policy.gait_phase = ObsTerm(
            func=walk_rewards.gait_phase_sin_cos,
            params={"cycle_time": WALK_CYCLE_TIME},
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
        self.events.physics_material.params["static_friction_range"] = (0.65, 1.05)
        self.events.physics_material.params["dynamic_friction_range"] = (0.45, 0.85)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.8, 2.0)
        self.events.base_com.params["asset_cfg"].body_names = "base"
        self.events.base_com.params["com_range"] = {
            "x": (-0.015, 0.015),
            "y": (-0.015, 0.015),
            "z": (-0.005, 0.005),
        }
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.0, 0.0)
        self.events.push_robot.interval_range_s = (12.0, 18.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.08, 0.08),
        }
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
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 0.4
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.18)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

    def _configure_rewards(self):
        self.rewards.track_lin_vel_xy_exp.weight = 1.6
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.18
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.dof_torques_l2.weight = -0.0003
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.action_rate_l2.weight = -0.06
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_1"
        self.rewards.feet_air_time.params["threshold"] = 0.12
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_2", ".*_thigh", ".*_calf"]),
                "threshold": 1.0,
            },
        )
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.pitch_l1 = RewTerm(
            func=walk_rewards.pitch_l1,
            weight=-5.0,
        )
        self.rewards.dof_pos_limits.weight = 0.0
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-7.0,
            params={"target_height": 0.36},
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.12,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_1"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_1"),
            },
        )
        self.rewards.walk_contact_phase = RewTerm(
            func=walk_rewards.walk_contact_phase,
            weight=1.4,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "cycle_time": WALK_CYCLE_TIME,
                "duty_factor": WALK_DUTY_FACTOR,
                "swing_offsets": WALK_SWING_OFFSETS,
                "contact_force_threshold": 1.0,
            },
        )
        self.rewards.walk_swing_foot_clearance = RewTerm(
            func=walk_rewards.walk_swing_foot_clearance,
            weight=0.45,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "cycle_time": WALK_CYCLE_TIME,
                "duty_factor": WALK_DUTY_FACTOR,
                "swing_offsets": WALK_SWING_OFFSETS,
                "target_height": 0.065,
                "contact_force_threshold": 1.0,
            },
        )
        self.rewards.foot_lateral_position_l2 = RewTerm(
            func=walk_rewards.foot_lateral_position_l2,
            weight=-0.9,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "target_abs_y": 0.25,
                "target_y_by_foot": (0.25, -0.25, 0.28, -0.28),
            },
        )
        self.rewards.walk_air_time_target_l2 = RewTerm(
            func=walk_rewards.feet_air_time_target_l2,
            weight=-1.4,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "target_air_time": WALK_TARGET_AIR_TIME,
            },
        )

    def _configure_terminations(self):
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        self.terminations.base_contact.params["threshold"] = 1.0
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.22},
        )


@configclass
class BennettWalk3FlatEnvCfg_PLAY(BennettWalk3FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
