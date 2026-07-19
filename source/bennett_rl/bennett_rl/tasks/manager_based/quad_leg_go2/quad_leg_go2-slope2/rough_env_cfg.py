import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import mdp
from .slope_terrain import DirectionalSlopeTerrainCfg

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V4

ACTUATED_JOINTS = ["FL_thigh", "FL_calf", "FR_thigh", "FR_calf", "RL_thigh", "RL_calf", "RR_thigh", "RR_calf"]
FOOT_BODIES = ["FL_1", "FR_1", "RL_1", "RR_1"]
BASE_BODY = ["base"]
TARGET_BASE_HEIGHT = 0.27
MIN_ROOT_HEIGHT = 0.15
LOW_SPEED_RANGE = (0.35, 0.80)
COMMAND_DEADBAND = 0.025
LOW_GAIT_FREQUENCY_HZ = 0.55
LOW_GAIT_DUTY_FACTOR = 0.78
LOW_GAIT_SWING_HEIGHT = 0.055
FRICTION_STATIC_RANGE = (0.6, 1.3)
FRICTION_DYNAMIC_RANGE = (0.5, 1.1)
BASE_MASS_SCALE_RANGE = (0.90, 1.10)
BASE_COM_RANGE = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)}
ACTUATOR_STIFFNESS_SCALE_RANGE = (0.80, 1.20)
ACTUATOR_DAMPING_SCALE_RANGE = (0.70, 1.30)
# Slope constants
SLOPE_ANGLES = (0.06, 0.10, 0.14, 0.18, 0.22, 0.27, 0.32, 0.38)
SLOPE_TERRAIN_SIZE = (8.0, 3.4)


@configclass
class QuadLegGo2Slope2EnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # === Robot ===
        robot_cfg = BENNETT_CFG_V4.copy()
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.spawn.articulation_props.fix_root_link = False
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
        robot_cfg.init_state.joint_pos.update({
            "FL_thigh": +0.08, "FR_thigh": -0.08,
            "RL_thigh": +0.08, "RR_thigh": -0.08,
            "FL_calf": -0.16, "FR_calf": -0.16,
            "RL_calf": -0.16, "RR_calf": -0.16,
        })
        robot_cfg.actuators["base_legs"].effort_limit = 7.0
        robot_cfg.actuators["base_legs"].saturation_effort = 12.0
        robot_cfg.actuators["base_legs"].stiffness = 28.0
        robot_cfg.actuators["base_legs"].damping = 2.0
        self.scene.robot = robot_cfg

        self.episode_length_s = 20.0

        # === Slope terrain (from V8 approach) ===
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=SLOPE_TERRAIN_SIZE,
            border_width=20.0,
            num_rows=1,
            num_cols=len(SLOPE_ANGLES),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                f"slope_{i:02d}": DirectionalSlopeTerrainCfg(
                    proportion=1.0 / len(SLOPE_ANGLES),
                    slope_angle_range=(a, a),
                    approach_length=1.3,
                    top_platform_length=1.0,
                    spawn_x=0.55,
                    lane_width=2.8,
                    transition_length=0.25,
                    transition_segments=8,
                )
                for i, a in enumerate(SLOPE_ANGLES)
            },
        )
        self.scene.terrain.max_init_terrain_level = 0

        # === Actions ===
        self.actions.joint_pos.joint_names = ACTUATED_JOINTS
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.preserve_order = True

        # === Commands: forward-only ===
        self.commands.base_velocity.resampling_time_range = (5.0, 8.0)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.heading_control_stiffness = 0.8
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = LOW_SPEED_RANGE
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # === Observations (from -7) ===
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
            params={"frequency_hz": LOW_GAIT_FREQUENCY_HZ, "command_name": "base_velocity",
                    "command_deadband": COMMAND_DEADBAND},
        )
        self.observations.policy.crawl_leg_phase = ObsTerm(
            func=mdp.commanded_crawl_leg_phase_sin_cos,
            params={"frequency_hz": LOW_GAIT_FREQUENCY_HZ, "duty_factor": LOW_GAIT_DUTY_FACTOR,
                    "command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.observations.policy.desired_contacts = ObsTerm(
            func=mdp.commanded_crawl_desired_contacts,
            params={"frequency_hz": LOW_GAIT_FREQUENCY_HZ, "duty_factor": LOW_GAIT_DUTY_FACTOR,
                    "command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.observations.policy.gait_params = ObsTerm(
            func=mdp.commanded_crawl_gait_params,
            params={"frequency_hz": LOW_GAIT_FREQUENCY_HZ, "duty_factor": LOW_GAIT_DUTY_FACTOR,
                    "swing_height": LOW_GAIT_SWING_HEIGHT,
                    "command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # === Events (from -7) ===
        self.events.physics_material.params["static_friction_range"] = FRICTION_STATIC_RANGE
        self.events.physics_material.params["dynamic_friction_range"] = FRICTION_DYNAMIC_RANGE
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.physics_material.params["num_buckets"] = 64
        self.events.physics_material.params["make_consistent"] = True
        self.events.add_base_mass.params["mass_distribution_params"] = BASE_MASS_SCALE_RANGE
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.add_base_mass.params["recompute_inertia"] = True
        self.events.add_base_mass.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_com.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_com.params["com_range"] = BASE_COM_RANGE
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
                    "stiffness_distribution_params": ACTUATOR_STIFFNESS_SCALE_RANGE,
                    "damping_distribution_params": ACTUATOR_DAMPING_SCALE_RANGE, "operation": "scale"},
        )
        self.events.base_external_force_torque.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.5, 0.5)
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
            "position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0),
        }
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.03, 0.03)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        }
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity, mode="interval",
            params={"velocity_range": {"x": (-0.45, 0.45), "y": (-0.35, 0.35), "yaw": (-0.25, 0.25)}},
            interval_range_s=(6.0, 12.0),
        )

        # === Rewards (from -7 + slope additions) ===
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=FOOT_BODIES, preserve_order=True
        )
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.8
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.10
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.dof_acc_l2.weight = -7.0e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-3.0,
            params={"target_height": TARGET_BASE_HEIGHT},
        )
        # Slope: align pitch with terrain, keep roll level
        self.rewards.slope_aligned_orientation_l2 = RewTerm(
            func=mdp.slope_aligned_orientation_l2, weight=-1.2,
            params={"slope_angles": SLOPE_ANGLES, "approach_length": 1.3,
                    "top_platform_length": 1.0, "spawn_x": 0.55, "transition_length": 0.25,
                    "slope_follow_ratio": 0.85, "roll_weight": 1.0},
        )
        # Straight-line lateral deviation penalty
        self.rewards.track_straight_line_y_l2 = RewTerm(
            func=mdp.track_straight_line_y_l2, weight=-0.8,
            params={"command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND, "max_lateral_error": 0.45},
        )
        self.rewards.lateral_yaw_vel_l2 = None
        self.rewards.base_ang_vel_xy_l2 = RewTerm(func=mdp.base_ang_vel_xy_l2, weight=-0.4)
        self.rewards.crawl_contact_match = RewTerm(
            func=mdp.crawl_contact_match, weight=0.3,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                    "frequency_hz": LOW_GAIT_FREQUENCY_HZ, "duty_factor": LOW_GAIT_DUTY_FACTOR,
                    "threshold": 1.0, "command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.missing_stance_contacts = RewTerm(
            func=mdp.crawl_missing_stance_contacts, weight=-0.45,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                    "frequency_hz": LOW_GAIT_FREQUENCY_HZ, "duty_factor": LOW_GAIT_DUTY_FACTOR,
                    "threshold": 1.0, "command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.extra_swing_contacts = RewTerm(
            func=mdp.crawl_extra_swing_contacts, weight=-0.7,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                    "frequency_hz": LOW_GAIT_FREQUENCY_HZ, "duty_factor": LOW_GAIT_DUTY_FACTOR,
                    "threshold": 1.0, "command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.fr_swing_touchdown_events = RewTerm(
            func=mdp.crawl_swing_touchdown_events, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                    "frequency_hz": LOW_GAIT_FREQUENCY_HZ, "duty_factor": LOW_GAIT_DUTY_FACTOR,
                    "foot_index": 1, "command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.swing_foot_clearance = RewTerm(
            func=mdp.crawl_swing_foot_clearance, weight=0.30,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                    "frequency_hz": LOW_GAIT_FREQUENCY_HZ, "duty_factor": LOW_GAIT_DUTY_FACTOR,
                    "target_height": LOW_GAIT_SWING_HEIGHT,
                    "command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.stance_feet_slide = RewTerm(
            func=mdp.crawl_stance_feet_slide, weight=-0.35,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                    "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                    "frequency_hz": LOW_GAIT_FREQUENCY_HZ, "duty_factor": LOW_GAIT_DUTY_FACTOR,
                    "threshold": 1.0, "max_value": 2.0,
                    "command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.stand_base_still = RewTerm(
            func=mdp.stand_still_base_vel_l2, weight=-1.0,
            params={"command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.stand_joint_still = RewTerm(
            func=mdp.stand_still_joint_vel_l2, weight=-0.04,
            params={"command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
        )
        self.rewards.stand_default_pose = RewTerm(
            func=mdp.stand_still_joint_pose_exp, weight=0.5,
            params={"command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND,
                    "sigma": 0.12,
                    "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
        )

        # === Terminations ===
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": MIN_ROOT_HEIGHT},
        )


@configclass
class QuadLegGo2Slope2EnvCfg_PLAY(QuadLegGo2Slope2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = len(SLOPE_ANGLES)
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = len(SLOPE_ANGLES)
            self.scene.terrain.terrain_generator.curriculum = True
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
