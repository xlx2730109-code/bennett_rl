"""Bennett Go2-Slope1: slope climbing training config.

Uses BENNETT_CFG_V6 (from Go2-11) with directional slope terrain for
climbing training.  Observations are Sim2Real-compatible (same as Go2-11).
"""

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

##
# Pre-defined configs
##
from bennett_rl.assets.robots.bennett import BENNETT_CFG_V6  # isort: skip


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

FOOT_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
BASE_BODY = ["base"]
TARGET_BASE_HEIGHT = 0.38
MIN_BASE_HEIGHT = 0.15
LOW_SPEED_RANGE = (0.04, 0.18)
COMMAND_DEADBAND = 0.025
LOW_GAIT_FREQUENCY_HZ = 0.55
LOW_GAIT_DUTY_FACTOR = 0.78
LOW_GAIT_SWING_HEIGHT = 0.065
CONTACT_TRANSITION_FRACTION = 0.04

# Domain randomisation ranges (from Go2-11)
FRICTION_STATIC_RANGE = (0.6, 1.3)
FRICTION_DYNAMIC_RANGE = (0.5, 1.1)
BASE_MASS_SCALE_RANGE = (0.90, 1.10)
BASE_COM_RANGE = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)}
ACTUATOR_STIFFNESS_SCALE_RANGE = (0.80, 1.20)
ACTUATOR_DAMPING_SCALE_RANGE = (0.70, 1.30)
JOINT_TARGET_LIMITS = {
    ".*_thigh": (-0.80, 0.80),
    ".*_calf": (-0.90, 0.55),
}

# Slope terrain parameters
SLOPE_ANGLES = tuple(math.radians(angle) for angle in (0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.8, 5.4, 6.0))
SLOPE_TERRAIN_SIZE = (5.0, 2.8)
SLOPE_APPROACH_LENGTH = 1.20
SLOPE_TOP_PLATFORM_LENGTH = 1.00
SLOPE_SPAWN_X = 0.65
SLOPE_TRANSITION_LENGTH = 0.25


@configclass
class QuadLegGo2Slope1EnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ---------- Robot config (from Go2-11, BENNETT_CFG_V6) ----------
        robot_cfg = BENNETT_CFG_V6.copy()
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.spawn.articulation_props.fix_root_link = False
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
        robot_cfg.init_state.joint_pos.update(
            {
                "FL_thigh": +0.08,
                "FR_thigh": -0.08,
                "RL_thigh": +0.08,
                "RR_thigh": -0.08,
                "FL_calf": -0.16,
                "FR_calf": -0.16,
                "RL_calf": -0.16,
                "RR_calf": -0.16,
            }
        )
        robot_cfg.actuators["base_legs"].effort_limit = 7.0
        robot_cfg.actuators["base_legs"].saturation_effort = 12.0
        robot_cfg.actuators["base_legs"].stiffness = 28.0
        robot_cfg.actuators["base_legs"].velocity_limit = 20.0
        robot_cfg.actuators["base_legs"].damping = 2.0
        # 初始位姿z=0.3匹配关节角度(大腿±0.08,小腿-0.16)的站立高度
        # root_height终止门槛已降到0.15m，不会误触发
        self.scene.robot = robot_cfg

        self.episode_length_s = 20.0

        # ---------- Slope terrain ----------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=SLOPE_TERRAIN_SIZE,
            border_width=10.0,
            num_rows=1,
            num_cols=len(SLOPE_ANGLES),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                f"directional_slope_{index:02d}": DirectionalSlopeTerrainCfg(
                    proportion=1.0 / len(SLOPE_ANGLES),
                    slope_angle_range=(slope_angle, slope_angle),
                    approach_length=SLOPE_APPROACH_LENGTH,
                    top_platform_length=SLOPE_TOP_PLATFORM_LENGTH,
                    spawn_x=SLOPE_SPAWN_X,
                    lane_width=2.4,
                    transition_length=SLOPE_TRANSITION_LENGTH,
                    transition_segments=8,
                )
                for index, slope_angle in enumerate(SLOPE_ANGLES)
            },
        )
        self.scene.terrain.max_init_terrain_level = 0

        # ---------- Actions ----------
        self.actions.joint_pos.joint_names = ACTUATED_JOINTS
        self.actions.joint_pos.scale = 0.20
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.clip = JOINT_TARGET_LIMITS

        # ---------- Commands: forward-only for slope climbing ----------
        self.commands.base_velocity.resampling_time_range = (5.0, 8.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.25
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, LOW_SPEED_RANGE[1])
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ---------- Observations: Sim2Real (same as Go2-11) ----------
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

        # ---------- 性能优化 ----------
        # 接触传感器：从监视所有body缩到仅监视需要的5个 (base + 4 feet)
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/(base|FL_foot|FR_foot|RL_foot|RR_foot)"
        # 地形border缩小(从10→3)，地形mesh更小
        self.scene.terrain.terrain_generator.border_width = 3.0
        # 缓存地形mesh，二次运行节省启动时间
        self.scene.terrain.terrain_generator.use_cache = True

        # ---------- Domain randomisation (from Go2-11) ----------
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
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0), "yaw": (-0.10, 0.10)},
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

        # ---------- Rewards ----------
        # --- Base reward tuning (same as Go2-11) ---
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = FOOT_BODIES
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.2
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.10
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        # Fine-grain tracking kernels for low-speed precision (Go2-11 fix)
        self.rewards.track_lin_vel_xy_fine_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=0.8,
            params={"std": 0.04, "command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_fine_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.6,
            params={"std": 0.10, "command_name": "base_velocity"},
        )
        self.rewards.dof_acc_l2.weight = -7.0e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_second_difference_l2 = RewTerm(
            func=mdp.action_second_difference_l2,
            weight=-0.004,
        )
        # flat_orientation replaced by slope_aligned_orientation_l2 below
        self.rewards.flat_orientation_l2.weight = 0.0
        # Base height: use absolute height (same as Go2-11), not relative to feet
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-8.0,
            params={"target_height": TARGET_BASE_HEIGHT},
        )

        # --- Slope-specific rewards ---
        # Slope-aligned orientation (pitch follows slope, roll stays level)
        self.rewards.slope_aligned_orientation_l2 = RewTerm(
            func=mdp.slope_aligned_orientation_l2,
            weight=-2.0,
            params={
                "slope_angles": SLOPE_ANGLES,
                "approach_length": SLOPE_APPROACH_LENGTH,
                "top_platform_length": SLOPE_TOP_PLATFORM_LENGTH,
                "spawn_x": SLOPE_SPAWN_X,
                "transition_length": SLOPE_TRANSITION_LENGTH,
                "slope_follow_ratio": 0.85,
                "roll_weight": 1.0,
            },
        )
        # Straight-line tracking (penalise lateral deviation)
        self.rewards.track_straight_line_y_l2 = RewTerm(
            func=mdp.track_straight_line_y_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "max_lateral_error": 0.35,
            },
        )

        # --- Velocity error penalties (Go2-11 style, NOT unconditional lateral_yaw_vel_l2) ---
        # Lateral/yaw error relative to command (no vy/yaw in forward-only command)
        self.rewards.lateral_yaw_vel_l2 = None  # removed — causes unconditional penalty
        self.rewards.commanded_lateral_yaw_vel_error_l2 = RewTerm(
            func=mdp.commanded_lateral_yaw_vel_error_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "yaw_weight": 2.0,
            },
        )
        self.rewards.base_ang_vel_xy_l2 = RewTerm(
            func=mdp.base_ang_vel_xy_l2,
            weight=-0.4,
        )

        # --- Gait contact matching (Go2-11 smooth-transition version) ---
        self.rewards.crawl_contact_match = RewTerm(
            func=mdp.crawl_contact_match,
            weight=0.3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "threshold": 1.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
            },
        )
        self.rewards.missing_stance_contacts = RewTerm(
            func=mdp.crawl_missing_stance_contacts,
            weight=-0.45,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "threshold": 1.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
            },
        )
        self.rewards.extra_swing_contacts = RewTerm(
            func=mdp.crawl_extra_swing_contacts,
            weight=-0.7,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "threshold": 1.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
            },
        )

        # --- Slope-aware swing foot clearance (height above terrain, not origin) ---
        self.rewards.swing_foot_clearance = RewTerm(
            func=mdp.crawl_slope_swing_foot_clearance,
            weight=0.75,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "target_height": LOW_GAIT_SWING_HEIGHT,
                "slope_angles": SLOPE_ANGLES,
                "approach_length": SLOPE_APPROACH_LENGTH,
                "top_platform_length": SLOPE_TOP_PLATFORM_LENGTH,
                "spawn_x": SLOPE_SPAWN_X,
                "transition_length": SLOPE_TRANSITION_LENGTH,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        # Smooth swing trajectory tracking (Go2-11 additions)
        self.rewards.swing_foot_height_tracking = RewTerm(
            func=mdp.crawl_swing_foot_height_tracking,
            weight=0.20,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "target_height": LOW_GAIT_SWING_HEIGHT,
                "sigma": 0.012,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.swing_foot_vertical_velocity_tracking = RewTerm(
            func=mdp.crawl_swing_foot_vertical_velocity_tracking,
            weight=0.06,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "target_height": LOW_GAIT_SWING_HEIGHT,
                "sigma": 0.18,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )

        # --- Stance foot slide penalty (with smooth transitions) ---
        self.rewards.stance_feet_slide = RewTerm(
            func=mdp.crawl_stance_feet_slide,
            weight=-0.35,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "threshold": 1.0,
                "max_value": 2.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
            },
        )

        # --- Touchdown impact penalty (Sim2Real) ---
        self.rewards.touchdown_impact_l2 = RewTerm(
            func=mdp.crawl_touchdown_impact_l2,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "soft_force_limit": 40.0,
                "max_normalized_excess": 3.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )

        # --- Stand-still penalties ---
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

        # --- Termination penalty ---
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=-100.0,
            params={"term_keys": ["base_contact", "root_height"]},
        )

        # ---------- Terminations ----------
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": MIN_BASE_HEIGHT},
        )


@configclass
class QuadLegGo2Slope1EnvCfg_PLAY(QuadLegGo2Slope1EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = len(SLOPE_ANGLES)
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # Keep one row and one lane for every configured slope angle.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = len(SLOPE_ANGLES)
            self.scene.terrain.terrain_generator.curriculum = True

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
