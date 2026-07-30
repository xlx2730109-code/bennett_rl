"""First Bennett diagonal-trot task, isolated from the Go2 crawl family."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

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
COMMAND_MODE_PROBABILITIES = (0.12, 0.22, 0.18, 0.16, 0.18, 0.14)
JOINT_TARGET_LIMITS = {
    ".*_thigh": (-0.80, 0.80),
    ".*_calf": (-0.90, 0.55),
}

# The clock is speed-conditioned instead of fixed.  The policy sees these
# values and the schedule is only a soft contact target; joint trajectories
# remain fully policy-controlled.
GAIT_PARAMS = {
    "command_name": "base_velocity",
    "command_deadband": COMMAND_DEADBAND,
    "min_frequency_hz": 0.75,
    "max_frequency_hz": 1.35,
    "min_equivalent_speed": 0.08,
    "max_equivalent_speed": 0.35,
    "low_speed_duty_factor": 0.62,
    "high_speed_duty_factor": 0.54,
    "swing_height": 0.045,
    "yaw_equivalent_radius": 0.20,
}
CONTACT_TRANSITION_FRACTION = 0.035


@configclass
class QuadLegTrot1FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
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
        # User-confirmed output-side motor ratings: 8 Nm continuous, 20 Nm peak.
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

        self.commands.base_velocity = mdp.BalancedTrotVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 7.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            mode_probabilities=COMMAND_MODE_PROBABILITIES,
            min_abs_lin_vel_x=0.08,
            min_abs_ang_vel_z=0.18,
            ranges=mdp.BalancedTrotVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.35),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.60, 0.60),
                heading=None,
            ),
        )

        # Hardware-observable policy contract.
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

        # Local flat mesh avoids remote-Nucleus dependencies.
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

        # Only the base and four real V6 foot bodies are needed.
        self.scene.contact_forces.prim_path = (
            "{ENV_REGEX_NS}/Robot/(base|FL_foot|FR_foot|RL_foot|RR_foot)"
        )

        # Keep Go2-13's measured Sim2Real uncertainty ranges.
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
                "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
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

        # General locomotion objective.
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -7.0e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_second_difference_l2 = RewTerm(
            func=mdp.action_second_difference_l2,
            weight=-0.004,
        )
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.15
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.track_lin_vel_xy_fine_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=0.7,
            params={"std": 0.06, "command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_fine_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.5,
            params={"std": 0.10, "command_name": "base_velocity"},
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
            params={"command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.commanded_straight_lateral_yaw_vel_l2 = RewTerm(
            func=mdp.commanded_straight_lateral_yaw_vel_l2,
            weight=-1.0,
            params={"command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
        )
        self.rewards.base_ang_vel_xy_l2 = RewTerm(func=mdp.base_ang_vel_xy_l2, weight=-0.4)

        contact_cfg = SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True)
        foot_cfg = SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True)
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
class QuadLegTrot1FlatEnvCfg_PLAY(QuadLegTrot1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
