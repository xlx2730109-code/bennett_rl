"""Independent Bennett Sim2Real task for blind forward stair climbing."""

from __future__ import annotations

from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
)

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V6

from . import mdp


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
JOINT_TARGET_LIMITS = {
    ".*_thigh": (-0.80, 0.80),
    ".*_calf": (-0.90, 0.55),
}

STAIR_HEIGHT_LEVELS = tuple(0.01 * (index + 1) for index in range(10))
STAIR_DEPTHS = tuple(0.25 + 0.01 * index for index in range(11))
STAIR_TERRAIN_SIZE = (5.0, 3.0)
STAIR_NUM_STEPS = 6
STAIR_SPAWN_X = 0.75
STAIR_APPROACH_DISTANCE = 0.60
STAIR_TOP_START_DISTANCES = tuple(
    STAIR_APPROACH_DISTANCE + STAIR_NUM_STEPS * depth for depth in STAIR_DEPTHS
)
STAIR_LANE_HALF_WIDTH = 1.05

TARGET_BASE_HEIGHT_ABOVE_FEET = 0.38
MINIMUM_FAILURE_CLEARANCE = 0.16
MINIMUM_SUCCESS_CLEARANCE = 0.20
COMMAND_DEADBAND = 0.025
UPSTAIR_SPEED_RANGE = (0.16, 0.24)
UPSTAIR_HEADING_RAD = 0.0
MAX_HEADING_CORRECTION_RAD_S = 0.40
SUPPORT_CONTACT_THRESHOLD = 1.0
MINIMUM_HEIGHT_SUPPORT_CONTACTS = 2
PROGRESS_WINDOW_S = 8.0
MINIMUM_PROGRESS_COMMAND_FRACTION = 0.45
MAXIMUM_SWING_CLEARANCE = 0.18
SWING_OVERCLEARANCE_NORMALIZATION = 0.05
ACTION_SOFT_LIMIT = 3.0
ACTION_HARD_LIMIT = 3.5
LEVEL_VALIDATION_EPISODES = 1024
LEVEL_REQUIRED_SUCCESS_RATE = 0.70
LEVEL_REQUIRED_CONSECUTIVE_PASS_BATCHES = 2
PREVIOUS_LEVEL_REPLAY_FRACTION = 0.25


def _stair_generator() -> TerrainGeneratorCfg:
    """Create ten exact height rows and eleven tread-depth columns."""

    return TerrainGeneratorCfg(
        seed=42,
        size=STAIR_TERRAIN_SIZE,
        border_width=3.0,
        num_rows=len(STAIR_HEIGHT_LEVELS),
        num_cols=len(STAIR_DEPTHS),
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        sub_terrains={
            f"ascending_{int(round(depth * 100)):02d}cm": mdp.AscendingStairsTerrainCfg(
                proportion=1.0 / len(STAIR_DEPTHS),
                height_levels=STAIR_HEIGHT_LEVELS,
                step_depth=depth,
                num_steps=STAIR_NUM_STEPS,
                spawn_x=STAIR_SPAWN_X,
                approach_distance=STAIR_APPROACH_DISTANCE,
            )
            for depth in STAIR_DEPTHS
        },
    )


@configclass
class StairPrivilegedObservationsCfg(ObsGroup):
    """Terrain/state data available only to the training critic."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )
    foot_contacts = ObsTerm(
        func=mdp.foot_contact_state,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FOOT_BODIES, preserve_order=True
            ),
            "threshold": 1.0,
        },
    )
    terrain_level = ObsTerm(
        func=mdp.normalized_terrain_level,
        params={"max_level": len(STAIR_HEIGHT_LEVELS) - 1},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StairObservationsCfg(ObservationsCfg):
    """Keep the deployed actor free of simulation-only terrain data."""

    privileged: StairPrivilegedObservationsCfg = StairPrivilegedObservationsCfg()


@configclass
class QuadLegStair1EnvCfg(LocomotionVelocityRoughEnvCfg):
    """Forward-only, gait-free and domain-randomized Bennett stair task."""

    observations: StairObservationsCfg = StairObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2048
        self.episode_length_s = 40.0

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
        # Measured Bennett output-side contract used by the real controller.
        robot_cfg.actuators["base_legs"].effort_limit = 8.0
        robot_cfg.actuators["base_legs"].saturation_effort = 20.0
        robot_cfg.actuators["base_legs"].velocity_limit = 19.896753
        robot_cfg.actuators["base_legs"].stiffness = 28.0
        robot_cfg.actuators["base_legs"].damping = 2.0
        self.scene.robot = robot_cfg

        self.actions.joint_pos.joint_names = ACTUATED_JOINTS
        self.actions.joint_pos.scale = 0.20
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.clip = JOINT_TARGET_LIMITS

        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            # Match the forward contract that Slope4 actually learned.  The
            # generated yaw command also makes world-+X lane recovery visible
            # through the existing 3-D command observation.
            resampling_time_range=(4.0, 7.0),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=UPSTAIR_SPEED_RANGE,
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(
                    -MAX_HEADING_CORRECTION_RAD_S,
                    MAX_HEADING_CORRECTION_RAD_S,
                ),
                heading=(UPSTAIR_HEADING_RAD, UPSTAIR_HEADING_RAD),
            ),
        )

        # Actor contract: IMU, command, eight joint positions/velocities and
        # previous action. It is 33-D and contains no height or contact sensor.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        joint_cfg = SceneEntityCfg(
            "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
        )
        self.observations.policy.joint_pos.params["asset_cfg"] = joint_cfg
        self.observations.policy.joint_vel.params["asset_cfg"] = joint_cfg
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _stair_generator()
        self.scene.terrain.use_terrain_origins = True
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.validated_stair_level,
            params={
                "success_term_name": "stair_success",
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "max_level": len(STAIR_HEIGHT_LEVELS) - 1,
                "validation_episodes": LEVEL_VALIDATION_EPISODES,
                "required_success_rate": LEVEL_REQUIRED_SUCCESS_RATE,
                "required_consecutive_pass_batches": LEVEL_REQUIRED_CONSECUTIVE_PASS_BATCHES,
                "previous_level_replay_fraction": PREVIOUS_LEVEL_REPLAY_FRACTION,
            },
        )

        self.scene.contact_forces.prim_path = (
            "{ENV_REGEX_NS}/Robot/(base|FL_foot|FR_foot|RL_foot|RR_foot)"
        )

        # Sim2Real randomization around known hardware values. Exact joint
        # order, policy rate, action scale and torque limits are not randomized.
        self.events.physics_material.params["static_friction_range"] = (0.6, 1.3)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.1)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.02)
        self.events.physics_material.params["num_buckets"] = 64
        self.events.physics_material.params["make_consistent"] = True
        self.events.add_base_mass.params["mass_distribution_params"] = (0.90, 1.10)
        self.events.add_base_mass.params["operation"] = "scale"
        self.events.add_base_mass.params["recompute_inertia"] = True
        self.events.add_base_mass.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_com.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_com.params["com_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.01, 0.01),
        }
        self.events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
                ),
                "stiffness_distribution_params": (0.80, 1.20),
                "damping_distribution_params": (0.70, 1.30),
                "operation": "scale",
            },
        )
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
            ),
            "position_range": (-0.04, 0.04),
            "velocity_range": (-0.05, 0.05),
        }
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.06, 0.06),
                "y": (-0.08, 0.08),
                "yaw": (-0.06, 0.06),
            },
            "velocity_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.0),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.03, 0.03),
            },
        }
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 14.0),
            params={
                "velocity_range": {
                    "x": (-0.25, 0.25),
                    "y": (-0.20, 0.20),
                    "yaw": (-0.15, 0.15),
                }
            },
        )

        # Gait-free objective: no clock, diagonal pair, fixed swing duration,
        # desired contact sequence or joint-reference trajectory is present.
        # A broad tracking kernel and a very small generic air-time term supply
        # motion-discovery gradients without prescribing which leg moves.
        self.rewards.undesired_contacts = None
        # Stair lanes are fixed in world +X.  Reward actual uphill translation,
        # not body-frame motion that can be exploited by turning or rocking.
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_uphill_world_velocity_exp
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.35
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.track_lin_vel_xy_fine_exp = RewTerm(
            func=mdp.track_uphill_world_velocity_exp,
            weight=0.8,
            params={"std": 0.12, "command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_fine_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.3,
            params={"std": 0.10, "command_name": "base_velocity"},
        )
        self.rewards.uphill_velocity_progress = RewTerm(
            func=mdp.uphill_velocity_progress,
            weight=0.8,
            params={"command_name": "base_velocity"},
        )
        self.rewards.feet_air_time.weight = 0.02
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = FOOT_BODIES
        self.rewards.feet_air_time.params["sensor_cfg"].preserve_order = True
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.lin_vel_z_l2.weight = -0.20
        self.rewards.ang_vel_xy_l2.weight = -0.10
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -7.0e-7
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.action_soft_limit = RewTerm(
            func=mdp.action_soft_limit_l2,
            weight=-0.05,
            params={
                "soft_limit": ACTION_SOFT_LIMIT,
                "hard_limit": ACTION_HARD_LIMIT,
            },
        )
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.dof_pos_limits.weight = -0.20

        contact_cfg = SceneEntityCfg(
            "contact_forces", body_names=FOOT_BODIES, preserve_order=True
        )
        foot_cfg = SceneEntityCfg(
            "robot", body_names=FOOT_BODIES, preserve_order=True
        )
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_above_feet_l2,
            weight=-4.0,
            params={
                "target_height": TARGET_BASE_HEIGHT_ABOVE_FEET,
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "contact_threshold": SUPPORT_CONTACT_THRESHOLD,
                "minimum_support_contacts": MINIMUM_HEIGHT_SUPPORT_CONTACTS,
            },
        )
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.15,
            params={"sensor_cfg": contact_cfg, "asset_cfg": foot_cfg},
        )
        self.rewards.swing_clearance = RewTerm(
            func=mdp.stair_swing_clearance,
            weight=0.15,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "contact_threshold": 1.0,
                "clearance_margin": 0.025,
                "height_levels": STAIR_HEIGHT_LEVELS,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.swing_overclearance = RewTerm(
            func=mdp.stair_swing_overclearance_l2,
            weight=-0.05,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "contact_threshold": 1.0,
                "maximum_clearance": MAXIMUM_SWING_CLEARANCE,
                "normalization": SWING_OVERCLEARANCE_NORMALIZATION,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.minimum_support = RewTerm(
            func=mdp.minimum_support_contacts_l2,
            weight=-0.10,
            params={
                "sensor_cfg": contact_cfg,
                "threshold": 1.0,
                "minimum_contacts": 1,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.lane_deviation = RewTerm(
            func=mdp.lane_deviation_l2,
            weight=-0.25,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.touchdown_impact_l2 = RewTerm(
            func=mdp.moving_touchdown_impact_l2,
            weight=-0.04,
            params={
                "sensor_cfg": contact_cfg,
                "soft_force_limit": 40.0,
                "max_normalized_excess": 3.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.stair_success = RewTerm(
            func=mdp.is_terminated_term,
            weight=250.0,
            params={"term_keys": ["stair_success"]},
        )
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=-100.0,
            params={"term_keys": ["base_contact", "low_base_clearance", "outside_lane"]},
        )

        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY
        self.terminations.low_base_clearance = DoneTerm(
            func=mdp.base_clearance_above_feet_below_minimum,
            params={
                # Give early exploration room to recover while the soft
                # 0.38-m height objective still rejects a crawling solution.
                "minimum_clearance": MINIMUM_FAILURE_CLEARANCE,
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "contact_threshold": SUPPORT_CONTACT_THRESHOLD,
                "minimum_support_contacts": MINIMUM_HEIGHT_SUPPORT_CONTACTS,
            },
        )
        self.terminations.insufficient_progress = DoneTerm(
            func=mdp.insufficient_stair_progress,
            params={
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "window_s": PROGRESS_WINDOW_S,
                "minimum_command_fraction": MINIMUM_PROGRESS_COMMAND_FRACTION,
            },
        )
        self.terminations.outside_lane = DoneTerm(
            func=mdp.outside_stair_lane,
            params={"maximum_lateral_distance": STAIR_LANE_HALF_WIDTH},
        )
        self.terminations.stair_success = DoneTerm(
            func=mdp.stair_top_success,
            params={
                "top_platform_start_distances": STAIR_TOP_START_DISTANCES,
                "foot_margin": 0.05,
                "hold_time_s": 0.25,
                "minimum_clearance": MINIMUM_SUCCESS_CLEARANCE,
                "minimum_upright_cosine": 0.75,
                "maximum_lateral_distance": STAIR_LANE_HALF_WIDTH,
                "asset_cfg": foot_cfg,
            },
        )

        self.rewards.termination_penalty.params["term_keys"].append(
            "insufficient_progress"
        )


@configclass
class QuadLegStair1EnvCfg_PLAY(QuadLegStair1EnvCfg):
    """Nominal evaluation configuration without training randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 22
        self.scene.env_spacing = 3.0
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = True
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params = {
            "pose_range": {},
            "velocity_range": {},
        }
