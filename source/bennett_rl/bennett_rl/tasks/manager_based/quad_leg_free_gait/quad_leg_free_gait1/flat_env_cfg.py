"""Gait-free omnidirectional Bennett locomotion on flat ground."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...quad_leg_trot.quad_leg_trot1.flat_env_cfg import (
    ACTUATED_JOINTS,
    COMMAND_DEADBAND,
    FOOT_BODIES,
    QuadLegTrot1FlatEnvCfg,
)

from . import mdp


COMMAND_MODE_PROBABILITIES = (0.10, 0.16, 0.16, 0.11, 0.11, 0.12, 0.10, 0.08, 0.06)


@configclass
class QuadLegFreeGait1FlatEnvCfg(QuadLegTrot1FlatEnvCfg):
    """Remove prescribed trot state while retaining the validated motor-B plant."""

    def __post_init__(self):
        super().__post_init__()

        # Motor B: 8 Nm continuous, 20 Nm transient saturation and the
        # datasheet-derived 120 rpm no-load-side speed limit.
        self.scene.robot.actuators["base_legs"].effort_limit = 8.0
        self.scene.robot.actuators["base_legs"].saturation_effort = 20.0
        self.scene.robot.actuators["base_legs"].velocity_limit = 19.896753

        self.commands.base_velocity = mdp.BalancedOmnidirectionalVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 7.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            mode_probabilities=COMMAND_MODE_PROBABILITIES,
            min_abs_lin_vel_x=0.08,
            min_abs_lin_vel_y=0.06,
            min_abs_ang_vel_z=0.18,
            ranges=mdp.BalancedOmnidirectionalVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.35),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.60, 0.60),
                heading=None,
            ),
        )

        # 33 hardware-observable values only. No clock, desired contact,
        # fixed step frequency, or leg-order observation remains.
        self.observations.policy.trot_phase = None
        self.observations.policy.trot_leg_phase = None
        self.observations.policy.desired_contacts = None
        self.observations.policy.gait_params = None

        # Remove every reward that references the diagonal-trot schedule.
        self.rewards.trot_contact_match = None
        self.rewards.trot_missing_stance_contacts = None
        self.rewards.trot_extra_swing_contacts = None
        self.rewards.trot_swing_foot_height_tracking = None
        self.rewards.trot_stance_feet_slide = None

        contact_cfg = SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True)
        foot_cfg = SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True)
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
                "target_height": 0.035,
                "sigma": 0.012,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )


@configclass
class QuadLegFreeGait1FlatEnvCfg_PLAY(QuadLegFreeGait1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
