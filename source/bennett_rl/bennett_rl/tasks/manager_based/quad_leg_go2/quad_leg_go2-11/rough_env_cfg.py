# 相较于10，修改了很多。



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

# old: Go2-7 uses ["FL_1", "FR_1", "RL_1", "RR_1"] on the old USD.
FOOT_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
BASE_BODY = ["base"]
TARGET_BASE_HEIGHT = 0.38
MIN_BASE_HEIGHT = 0.27
LOW_SPEED_RANGE = (0.04, 0.18)
COMMAND_DEADBAND = 0.025
LOW_GAIT_FREQUENCY_HZ = 0.55
LOW_GAIT_DUTY_FACTOR = 0.78
LOW_GAIT_SWING_HEIGHT = 0.065
CONTACT_TRANSITION_FRACTION = 0.04
# stand, forward, backward, yaw, forward+yaw, backward+yaw, lateral, lateral+yaw
COMMAND_MODE_PROBABILITIES = (0.15, 0.14, 0.14, 0.20, 0.15, 0.15, 0.035, 0.035)
JOINT_TARGET_LIMITS = {
    ".*_thigh": (-0.80, 0.80),
    ".*_calf": (-0.90, 0.55),
}
FRICTION_STATIC_RANGE = (0.6, 1.3)
FRICTION_DYNAMIC_RANGE = (0.5, 1.1)
BASE_MASS_SCALE_RANGE = (0.90, 1.10)
BASE_COM_RANGE = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)}
ACTUATOR_STIFFNESS_SCALE_RANGE = (0.80, 1.20)
ACTUATOR_DAMPING_SCALE_RANGE = (0.70, 1.30)


@configclass
class QuadLegGo211RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # old: Go2-7 uses BENNETT_CFG_V4 on Urdf_Bennett_1.
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
        # 24 V manual: approximately 190 rpm no-load output speed (19.9 rad/s).
        robot_cfg.actuators["base_legs"].velocity_limit = 20.0

        robot_cfg.actuators["base_legs"].damping = 2.0
        self.scene.robot = robot_cfg

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Bennett has 8 actuated joints. Keep the order identical to deployment.
        self.actions.joint_pos.joint_names = ACTUATED_JOINTS
        self.actions.joint_pos.scale = 0.20
        self.actions.joint_pos.preserve_order = True
        # The current USD has infinite position limits.  Clip absolute processed
        # targets here so safety does not depend only on the PPO runner clip.
        self.actions.joint_pos.clip = JOINT_TARGET_LIMITS

        # old: Go2-10 sampled vx/vy/yaw independently.  That gave little
        # coverage to the exact keyboard modes (pure reverse, pure yaw, and
        # simultaneous forward/yaw) used on hardware.
        self.commands.base_velocity = mdp.BalancedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 7.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            # Avoid a nonessential remote USD dependency during headless training.
            debug_vis=False,
            mode_probabilities=COMMAND_MODE_PROBABILITIES,
            min_abs_lin_vel_x=0.06,
            min_abs_lin_vel_y=0.04,
            min_abs_ang_vel_z=0.15,
            ranges=mdp.BalancedVelocityCommandCfg.Ranges(
                lin_vel_x=(-LOW_SPEED_RANGE[1], LOW_SPEED_RANGE[1]),
                lin_vel_y=(-0.10, 0.10),
                ang_vel_z=(-0.50, 0.50),
                heading=None,
            ),
        )

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

        # Domain randomization follows Go2-7, but body names are scoped to the new USD.
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
        # Preserve Go2-10's early locomotion signal.  Removing both air-time
        # and clearance made four-step collapse a stable local optimum in Go2-12.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = FOOT_BODIES
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.2
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.10
        # Go2-10's final yaw error remained large relative to its command range.
        # A modest +33% task-weight increase keeps yaw below linear tracking but
        # makes pure and combined turn modes worth learning.
        self.rewards.track_ang_vel_z_exp.weight = 0.8
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        # A narrow companion kernel makes exact low-speed tracking materially
        # better than standing still, while the broad kernels preserve a dense
        # early-learning signal.
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
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-8.0,
            params={"target_height": TARGET_BASE_HEIGHT},
        )
        # This term penalizes lateral and yaw velocity unconditionally, which
        # conflicts with omnidirectional velocity tracking.
        self.rewards.lateral_yaw_vel_l2 = None
        self.rewards.commanded_lateral_yaw_vel_error_l2 = RewTerm(
            func=mdp.commanded_lateral_yaw_vel_error_l2,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "yaw_weight": 2.0,
            },
        )
        self.rewards.commanded_straight_lateral_yaw_vel_l2 = RewTerm(
            func=mdp.commanded_straight_lateral_yaw_vel_l2,
            weight=-1.2,
            params={
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
                "straight_command_deadband": COMMAND_DEADBAND,
                # old: 0.08 treated small but intentional keyboard yaw as a
                # straight command, directly opposing the yaw-tracking reward.
                "yaw_command_deadband": COMMAND_DEADBAND,
                "yaw_weight": 1.0,
            },
        )
        self.rewards.base_ang_vel_xy_l2 = RewTerm(
            func=mdp.base_ang_vel_xy_l2,
            weight=-0.4,
        )
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
        # old: Go2-8/10 penalized only FR touchdown events, which can bias the crawl cycle.
        self.rewards.fr_swing_touchdown_events = None
        # Keep the proven clearance reward as the primary lift-off signal, then
        # use the smooth terms only as low-weight trajectory shaping.
        self.rewards.swing_foot_clearance = RewTerm(
            func=mdp.crawl_swing_foot_clearance,
            weight=0.80,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
                "frequency_hz": LOW_GAIT_FREQUENCY_HZ,
                "duty_factor": LOW_GAIT_DUTY_FACTOR,
                "target_height": LOW_GAIT_SWING_HEIGHT,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
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

        # A non-timeout failure must be more expensive than the apparent
        # benefit of ending an episode after only a few low-motion steps.
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=-100.0,
            params={"term_keys": ["base_contact", "root_height"]},
        )


        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": MIN_BASE_HEIGHT},
        )


@configclass
class QuadLegGo211RoughEnvCfg_PLAY(QuadLegGo211RoughEnvCfg):
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
