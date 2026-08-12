"""Standalone Bennett locomotion on a progressive fixed-gravel terrain."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
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
TARGET_BASE_HEIGHT = 0.38
MINIMUM_BASE_CLEARANCE = 0.20
COMMAND_DEADBAND = 0.025
COMMAND_MODE_PROBABILITIES = (
    0.10,  # stand
    0.22,  # forward
    0.18,  # backward
    0.07,  # lateral left
    0.07,  # lateral right
    0.12,  # yaw only
    0.12,  # forward + yaw
    0.08,  # backward + yaw
    0.04,  # lateral + yaw
)

GRAVEL_LEVELS = 10
GRAVEL_TRAIN_COLUMNS = 20
GRAVEL_PLAY_COLUMNS = 5
GRAVEL_TERRAIN_SIZE = (8.0, 8.0)
GRAVEL_CELL_WIDTH = 0.18
GRAVEL_MAX_HALF_HEIGHT = 0.04
GRAVEL_PLATFORM_WIDTH = 2.0


def _gravel_generator(*, num_cols: int = GRAVEL_TRAIN_COLUMNS) -> TerrainGeneratorCfg:
    """Create a row-wise curriculum of increasingly uneven fixed stones.

    ``MeshRandomGridTerrainCfg`` is used because its height amplitude really is
    interpolated by Isaac Lab's difficulty parameter. In contrast, the stock
    ``HfRandomUniformTerrainCfg`` currently ignores difficulty entirely.
    """

    return TerrainGeneratorCfg(
        seed=42,
        size=GRAVEL_TERRAIN_SIZE,
        # Match the official Go2 rough-terrain layout: difficulty changes by
        # row, while columns provide independent samples at the same level.
        border_width=20.0,
        num_rows=GRAVEL_LEVELS,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        # Match the official Go2 rough config: generate in memory so training
        # does not depend on write access to Isaac Lab's global terrain cache.
        use_cache=False,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        sub_terrains={
            "gravel": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=1.0,
                grid_width=GRAVEL_CELL_WIDTH,
                grid_height_range=(0.0, GRAVEL_MAX_HALF_HEIGHT),
                platform_width=GRAVEL_PLATFORM_WIDTH,
            )
        },
    )


@configclass
class GravelPrivilegedObservationsCfg(ObsGroup):
    """Simulation-only terrain/state observations used by the critic."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GravelObservationsCfg(ObservationsCfg):
    """Keep the actor deployable while giving the critic terrain context."""

    privileged: GravelPrivilegedObservationsCfg = GravelPrivilegedObservationsCfg()


@configclass
class QuadLegGravel1EnvCfg(LocomotionVelocityRoughEnvCfg):
    """Domain-randomized Bennett gravel locomotion for Sim2Real transfer."""

    observations: GravelObservationsCfg = GravelObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

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
        robot_cfg.actuators["base_legs"].effort_limit = 8.0
        robot_cfg.actuators["base_legs"].saturation_effort = 20.0
        robot_cfg.actuators["base_legs"].velocity_limit = 20.0
        robot_cfg.actuators["base_legs"].stiffness = 28.0
        robot_cfg.actuators["base_legs"].damping = 2.0
        self.scene.robot = robot_cfg

        self.actions.joint_pos.joint_names = ACTUATED_JOINTS
        self.actions.joint_pos.scale = 0.20
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.clip = JOINT_TARGET_LIMITS

        # One explicit command mode per complete episode. This removes the
        # distance-cancellation failure in the previous 10 s command schedule
        # and guarantees coverage of reverse/lateral/yaw behaviours.
        self.commands.base_velocity = mdp.BalancedOmnidirectionalVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            mode_probabilities=COMMAND_MODE_PROBABILITIES,
            min_abs_lin_vel_x=0.08,
            min_abs_lin_vel_y=0.05,
            min_abs_ang_vel_z=0.15,
            ranges=mdp.BalancedOmnidirectionalVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.35),
                lin_vel_y=(-0.12, 0.12),
                ang_vel_z=(-0.50, 0.50),
                heading=None,
            ),
        )

        # The actor stays on the existing 33-D hardware observation contract.
        # Height scan and base linear velocity exist only in ``privileged`` and
        # therefore are never required by an exported hardware policy.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None
        joint_cfg = SceneEntityCfg(
            "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
        )
        self.observations.policy.joint_pos.params["asset_cfg"] = joint_cfg
        self.observations.policy.joint_vel.params["asset_cfg"] = joint_cfg

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = _gravel_generator()
        self.scene.terrain.use_terrain_origins = True
        self.scene.terrain.max_init_terrain_level = 1
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.gravel_terrain_levels,
            params={
                "command_name": "base_velocity",
                "linear_reward_name": "track_lin_vel_xy_exp",
                "yaw_reward_name": "track_ang_vel_z_exp",
                "command_deadband": COMMAND_DEADBAND,
                "promote_linear_score": 0.72,
                "promote_yaw_score": 0.65,
                "demote_linear_score": 0.45,
                "demote_yaw_score": 0.40,
            },
        )

        self.scene.contact_forces.prim_path = (
            "{ENV_REGEX_NS}/Robot/(base|FL_foot|FR_foot|RL_foot|RR_foot)"
        )

        # Sim2Real randomization around the measured Bennett control contract.
        # The actuator limits and action scale above remain exact; uncertainty
        # is injected into contact, rigid-body and closed-loop motor dynamics.
        self.events.push_robot = None
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
        # Persistent external wrench training masks poor terrain mechanics and
        # is not part of the identified hardware uncertainty model.
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
            ),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.05, 0.05),
        }
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.50, 0.50),
                "y": (-0.50, 0.50),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        }

        # Gait-free rough-terrain mechanics. These terms do not impose a clock
        # or a leg order, but they prevent the low crouched/shuffling solution.
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.15
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.dof_acc_l2.weight = -7.0e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-8.0,
            params={
                "target_height": TARGET_BASE_HEIGHT,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        contact_cfg = SceneEntityCfg(
            "contact_forces", body_names=FOOT_BODIES, preserve_order=True
        )
        foot_cfg = SceneEntityCfg(
            "robot", body_names=FOOT_BODIES, preserve_order=True
        )
        self.rewards.gait_free_stance_feet_slide = RewTerm(
            func=mdp.gait_free_stance_feet_slide,
            weight=-0.30,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "threshold": 1.0,
                "max_value": 2.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.minimum_support_contacts = RewTerm(
            func=mdp.minimum_support_contacts_l2,
            weight=-0.50,
            params={
                "sensor_cfg": contact_cfg,
                "threshold": 1.0,
                "minimum_contacts": 2,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.gait_free_swing_clearance = RewTerm(
            func=mdp.gait_free_swing_clearance,
            weight=0.35,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "threshold": 1.0,
                "target_height": 0.05,
                "sigma": 0.018,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.leg_lift_starvation = RewTerm(
            func=mdp.gait_free_leg_lift_starvation_l2,
            weight=-0.40,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "contact_threshold": 1.0,
                "valid_lift_height": 0.03,
                "slow_allowed_time": 1.20,
                "fast_allowed_time": 0.65,
                "min_equivalent_speed": 0.06,
                "max_equivalent_speed": 0.35,
                "yaw_equivalent_radius": 0.20,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "max_normalized_excess": 2.0,
            },
        )

        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY
        self.terminations.low_base_clearance = DoneTerm(
            func=mdp.root_height_above_terrain_below_minimum,
            params={
                "minimum_clearance": MINIMUM_BASE_CLEARANCE,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )


@configclass
class QuadLegGravel1EnvCfg_PLAY(QuadLegGravel1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = GRAVEL_LEVELS * GRAVEL_PLAY_COLUMNS
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_generator = _gravel_generator(
            num_cols=GRAVEL_PLAY_COLUMNS
        )
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        # Preserve one terrain row per level for visual comparison.
        self.scene.terrain.terrain_generator.curriculum = True
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
