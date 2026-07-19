# 硬蹬地推进，脚底打滑、蹬地僵硬，机身高度不高

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import mdp

##
# Pre-defined configs
##
from bennett_rl.assets.robots.bennett import BENNETT_CFG_V4  # isort: skip


ACTUATED_JOINTS = [
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
BASE_BODY = ["base"]
TARGET_BASE_HEIGHT = 0.32
MIN_BASE_HEIGHT = 0.20
LOW_SPEED_RANGE = (0.04, 0.18)
COMMAND_DEADBAND = 0.025
LOW_GAIT_FREQUENCY_HZ = 0.65
LOW_GAIT_DUTY_FACTOR = 0.80
LOW_GAIT_SWING_HEIGHT = 0.055
FRICTION_STATIC_RANGE = (0.6, 1.3)
FRICTION_DYNAMIC_RANGE = (0.5, 1.1)
BASE_MASS_SCALE_RANGE = (0.90, 1.10)
BASE_COM_RANGE = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)}
ACTUATOR_STIFFNESS_SCALE_RANGE = (0.80, 1.20)
ACTUATOR_DAMPING_SCALE_RANGE = (0.70, 1.30)


@configclass
class QuadLegGo23RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        robot_cfg = BENNETT_CFG_V4.copy()
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.spawn.articulation_props.fix_root_link = False
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
        robot_cfg.actuators["base_legs"].effort_limit = 8.0
        robot_cfg.actuators["base_legs"].saturation_effort = 20.0
        robot_cfg.actuators["base_legs"].stiffness = 30.0
        robot_cfg.actuators["base_legs"].damping = 2.0
        self.scene.robot = robot_cfg

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Bennett has 8 actuated joints. Keep the order identical to deployment.
        self.actions.joint_pos.joint_names = ACTUATED_JOINTS
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.preserve_order = True

        # Low-speed straight-line command range for Bennett hardware bring-up.
        self.commands.base_velocity.resampling_time_range = (5.0, 8.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = LOW_SPEED_RANGE
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Sim2Real observation cleanup: Bennett hardware does not directly provide
        # base linear velocity or terrain height scan.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
        )
        self.observations.policy.crawl_phase = ObsTerm(
            func=mdp.commanded_crawl_global_phase_sin_cos,
            params={
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.observations.policy.crawl_leg_phase = ObsTerm(
            func=mdp.commanded_crawl_leg_phase_sin_cos,
            params={
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.observations.policy.desired_contacts = ObsTerm(
            func=mdp.commanded_crawl_desired_contacts,
            params={
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.observations.policy.gait_params = ObsTerm(
            func=mdp.commanded_crawl_gait_params,
            params={
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "swing_height": LOW_GAIT_SWING_HEIGHT,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # Domain randomization follows the proven crawl setup, but keeps the
        # Go2-style velocity task structure.
        self.events.physics_material.params["static_friction_range"] = FRICTION_STATIC_RANGE
        self.events.physics_material.params["dynamic_friction_range"] = FRICTION_DYNAMIC_RANGE
        self.events.physics_material.params["restitution_range"] = (0.0, 0.02)
        self.events.physics_material.params["num_buckets"] = 64
        self.events.physics_material.params["make_consistent"] = True
        self.events.add_base_mass.params["mass_distribution_params"] = BASE_MASS_SCALE_RANGE
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.add_base_mass.params["recompute_inertia"] = True
        self.events.add_base_mass.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_com.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_com.params["com_range"] = BASE_COM_RANGE
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
                "stiffness_distribution_params": ACTUATOR_STIFFNESS_SCALE_RANGE,
                "damping_distribution_params": ACTUATOR_DAMPING_SCALE_RANGE,
                "operation": "scale",
            },
        )
        self.events.base_external_force_torque.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.5, 0.5)
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        }
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.10, 0.10)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        }
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            params={"velocity_range": {"x": (-0.45, 0.45), "y": (-0.35, 0.35), "yaw": (-0.25, 0.25)}},
            interval_range_s=(6.0, 12.0),
        )

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = FOOT_BODIES
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.10
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-6.0,
            params={"target_height": TARGET_BASE_HEIGHT},
        )
        self.rewards.lateral_yaw_vel_l2 = RewTerm(
            func=mdp.lateral_yaw_vel_l2,
            weight=-1.0,
        )
        self.rewards.crawl_contact_match = RewTerm(
            func=mdp.crawl_contact_match,
            weight=0.6,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "threshold": 1.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.missing_stance_contacts = RewTerm(
            func=mdp.crawl_missing_stance_contacts,
            weight=-0.8,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "threshold": 1.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.extra_swing_contacts = RewTerm(
            func=mdp.crawl_extra_swing_contacts,
            weight=-1.2,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "threshold": 1.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.swing_foot_clearance = RewTerm(
            func=mdp.crawl_swing_foot_clearance,
            weight=0.9,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "target_height": LOW_GAIT_SWING_HEIGHT,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.stance_feet_slide = RewTerm(
            func=mdp.crawl_stance_feet_slide,
            weight=-0.12,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "threshold": 1.0,
                "max_value": 2.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.stand_base_still = RewTerm(
            func=mdp.stand_still_base_vel_l2,
            weight=-1.0,
            params={"command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.stand_joint_still = RewTerm(
            func=mdp.stand_still_joint_vel_l2,
            weight=-0.04,
            params={
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
            },
        )
        self.rewards.stand_default_pose = RewTerm(
            func=mdp.stand_still_joint_pose_exp,
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "sigma": 0.12,
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
            },
        )

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": MIN_BASE_HEIGHT},
        )


@configclass
class QuadLegGo23RoughEnvCfg_PLAY(QuadLegGo23RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
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

