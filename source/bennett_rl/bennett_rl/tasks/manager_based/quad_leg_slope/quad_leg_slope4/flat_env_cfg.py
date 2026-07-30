"""Self-contained Bennett Slope4 locomotion objective and flat smoke terrain."""

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
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

TARGET_BASE_HEIGHT = 0.38
MIN_BASE_HEIGHT = 0.27
COMMAND_DEADBAND = 0.025
UPHILL_SPEED_RANGE = (0.16, 0.24)
UPHILL_HEADING_RAD = 0.0
MAX_HEADING_CORRECTION_RAD_S = 0.40
JOINT_TARGET_LIMITS = {
    ".*_thigh": (-0.80, 0.80),
    ".*_calf": (-0.90, 0.55),
}

# Speed-conditioned diagonal trot.  The schedule is a soft contact target:
# the policy still controls every joint and can prolong support on the slope.
GAIT_PARAMS = {
    "command_name": "base_velocity",
    "command_deadband": COMMAND_DEADBAND,
    # At the 0.20 m/s command midpoint this gives 0.85 Hz instead of 1.00 Hz.
    # Keeping speed unchanged therefore asks for an approximately 18% longer
    # cycle stride without hard-coding footholds.
    "min_frequency_hz": 0.65,
    "max_frequency_hz": 1.05,
    "min_equivalent_speed": 0.12,
    "max_equivalent_speed": 0.28,
    "low_speed_duty_factor": 0.68,
    "high_speed_duty_factor": 0.60,
    # The identical Bennett linkage already tracked 0.045 m in Trot1.
    # Slope3 used only 0.035 m and substantially under-tracked that target.
    "swing_height": 0.045,
    "yaw_equivalent_radius": 0.10,
}
CONTACT_TRANSITION_FRACTION = 0.040


@configclass
class QuadLegSlope4FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Complete Slope4 contract on flat ground, with no Bennett task inheritance."""

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
        # DM-J8006-2EC V1.1 output-side contract used by training and deployment.
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

        # Every lane rises along world +X.  Commands contain forward motion and
        # only the yaw correction required to recover the zero world heading.
        self.commands.base_velocity = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 7.0),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=UPHILL_SPEED_RANGE,
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(
                    -MAX_HEADING_CORRECTION_RAD_S,
                    MAX_HEADING_CORRECTION_RAD_S,
                ),
                heading=(UPHILL_HEADING_RAD, UPHILL_HEADING_RAD),
            ),
        )

        # Exact 50-value hardware-observable policy contract:
        # angular velocity, projected gravity, command, 8 positions,
        # 8 velocities, previous 8 actions, global/leg phases,
        # desired contacts and speed-conditioned gait parameters.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
        )
        self.observations.policy.trot_phase = ObsTerm(
            func=mdp.commanded_trot_global_phase_sin_cos,
            params=dict(GAIT_PARAMS),
        )
        self.observations.policy.trot_leg_phase = ObsTerm(
            func=mdp.commanded_trot_leg_phase_sin_cos,
            params=dict(GAIT_PARAMS),
        )
        self.observations.policy.desired_contacts = ObsTerm(
            func=mdp.commanded_trot_desired_contacts,
            params=dict(GAIT_PARAMS),
        )
        self.observations.policy.gait_params = ObsTerm(
            func=mdp.commanded_trot_gait_params,
            params=dict(GAIT_PARAMS),
        )

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42,
            curriculum=False,
            size=(256.0, 256.0),
            num_rows=1,
            num_cols=1,
            color_scheme="none",
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        )
        self.scene.terrain.use_terrain_origins = False
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        self.scene.contact_forces.prim_path = (
            "{ENV_REGEX_NS}/Robot/(base|FL_foot|FR_foot|RL_foot|RR_foot)"
        )

        # Task-local Sim2Real randomization.
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
        self.events.base_external_force_torque.params["asset_cfg"].body_names = BASE_BODY
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-0.5, 0.5)
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=ACTUATED_JOINTS, preserve_order=True
            ),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        }
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "yaw": (-0.10, 0.10),
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
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            params={
                "velocity_range": {
                    "x": (-0.45, 0.45),
                    "y": (-0.35, 0.35),
                    "yaw": (-0.25, 0.25),
                }
            },
            interval_range_s=(6.0, 12.0),
        )

        # General locomotion and straight-uphill objective.
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.action_rate_l2.weight = -0.03
        self.rewards.action_second_difference_l2 = RewTerm(
            func=mdp.action_second_difference_l2,
            weight=-0.006,
        )
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.12
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.track_lin_vel_xy_fine_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=0.8,
            params={"std": 0.05, "command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_fine_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.5,
            params={"std": 0.10, "command_name": "base_velocity"},
        )
        self.rewards.uphill_velocity_progress = RewTerm(
            func=mdp.uphill_velocity_progress,
            weight=0.8,
            params={"command_name": "base_velocity"},
        )
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-8.0,
            params={"target_height": TARGET_BASE_HEIGHT},
        )
        self.rewards.commanded_yaw_error_l2 = RewTerm(
            func=mdp.commanded_yaw_error_l2,
            weight=-0.8,
            params={
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.commanded_straight_lateral_yaw_vel_l2 = RewTerm(
            func=mdp.commanded_straight_lateral_yaw_vel_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.base_ang_vel_xy_l2 = RewTerm(
            func=mdp.base_ang_vel_xy_l2,
            weight=-0.4,
        )

        contact_cfg = SceneEntityCfg(
            "contact_forces", body_names=FOOT_BODIES, preserve_order=True
        )
        foot_cfg = SceneEntityCfg(
            "robot", body_names=FOOT_BODIES, preserve_order=True
        )
        self.rewards.touchdown_impact_l2 = RewTerm(
            func=mdp.moving_touchdown_impact_l2,
            weight=-0.06,
            params={
                "sensor_cfg": contact_cfg,
                "soft_force_limit": 40.0,
                "max_normalized_excess": 3.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        self.rewards.trot_contact_match = RewTerm(
            func=mdp.trot_contact_match,
            weight=0.30,
            params={
                "sensor_cfg": contact_cfg,
                "threshold": 1.0,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
                **GAIT_PARAMS,
            },
        )
        self.rewards.trot_missing_stance_contacts = RewTerm(
            func=mdp.trot_missing_stance_contacts,
            weight=-0.30,
            params={
                "sensor_cfg": contact_cfg,
                "threshold": 1.0,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
                **GAIT_PARAMS,
            },
        )
        self.rewards.trot_extra_swing_contacts = RewTerm(
            func=mdp.trot_extra_swing_contacts,
            weight=-0.45,
            params={
                "sensor_cfg": contact_cfg,
                "threshold": 1.0,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
                **GAIT_PARAMS,
            },
        )
        self.rewards.trot_swing_foot_height_tracking = RewTerm(
            func=mdp.trot_swing_foot_height_tracking,
            weight=0.70,
            params={
                "asset_cfg": foot_cfg,
                "sigma": 0.012,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
                **GAIT_PARAMS,
            },
        )
        self.rewards.trot_worst_swing_foot_height_shortfall_l2 = RewTerm(
            func=mdp.trot_worst_swing_foot_height_shortfall_l2,
            weight=-0.60,
            params={
                "asset_cfg": foot_cfg,
                # The already-good rear feet reach roughly 28--36 mm. Require
                # every scheduled swing foot to reach 75% of the existing
                # 45-mm smooth profile instead of increasing the lift target.
                "minimum_height_fraction": 0.75,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
                **GAIT_PARAMS,
            },
        )
        self.rewards.trot_stance_feet_slide = RewTerm(
            func=mdp.trot_stance_feet_slide,
            weight=-0.30,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "threshold": 1.0,
                "transition_fraction": CONTACT_TRANSITION_FRACTION,
                "max_value": 2.0,
                **GAIT_PARAMS,
            },
        )
        self.rewards.feet_lateral_stance_width_excess_l2 = RewTerm(
            func=mdp.feet_lateral_stance_width_excess_l2,
            weight=-6.0,
            params={
                "asset_cfg": foot_cfg,
                # Body-frame front/rear pair widths. Neutral measurements are
                # about 0.36 m; the margin preserves uphill balance correction.
                "max_pair_width": (0.40, 0.39),
                # Keep the gradient active across the observed ~0.55 m front
                # stance instead of saturating after only 0.10 m of excess.
                "max_excess": 0.25,
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

        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=-100.0,
            params={"term_keys": ["base_contact", "root_height"]},
        )
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": MIN_BASE_HEIGHT},
        )


@configclass
class QuadLegSlope4FlatEnvCfg_PLAY(QuadLegSlope4FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
