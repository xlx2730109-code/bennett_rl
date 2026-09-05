"""Canonical emergent-gait Bennett locomotion on flat ground.

This is a direct port of the open-source Isaac Lab ``Go2FlatEnvCfg`` reward /
command recipe (``isaaclab_tasks/.../locomotion/velocity/config/go2/flat_env_cfg.py``)
onto the Bennett body, with every Bennett-specific reward removed so the gait can
emerge freely. Differences from go2 are deliberate and minimal:

  * ``base_lin_vel`` stays out of the observation. Bennett has no odometry /
    motion capture on hardware, so the body x/y velocity is not observable on
    the real robot (this is already the frozen 33-dim contract). The reward
    terms still read ``base_lin_vel`` internally from the sim, so training
    quality is unaffected -- only the policy's *input* omits it.
  * command ranges are scaled down to Bennett's size (it is much slower than the
    1 m/s go2), and the resampling window is shorter for more coverage.

Everything else -- ``heading_command`` omnidirectional control, the reward
weights, ``feet_air_time`` as the emergent driver (weight 0.25, threshold 0.25),
``flat_orientation_l2``, the torques / dof-acc / action-rate penalties -- follows
go2-flat.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import feet_air_time

from ...quad_leg_trot.quad_leg_trot1.flat_env_cfg import FOOT_BODIES
from ...quad_leg_free_gait.quad_leg_free_gait1.flat_env_cfg import QuadLegFreeGait1FlatEnvCfg

from . import mdp


@configclass
class QuadLegFreeGait3FlatEnvCfg(QuadLegFreeGait1FlatEnvCfg):
    """go2-flat reward/command recipe mapped onto Bennett, obs free of gait terms."""

    def __post_init__(self):
        super().__post_init__()

        # --- command: omnidirectional, raw yaw-rate (transferable to the bridge) ---
        # heading_command stays False on purpose: the bennett_deploy bridge
        # commands a raw (vx, vy, wz) velocity vector straight from the keyboard,
        # so the policy must be trained to read the 3rd command component as a
        # yaw-RATE, not as a heading-error-derived value. Plain uniform sampling
        # over the ranges scaled to Bennett (go2 uses +/-1.0; Bennett is slower).
        self.commands.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 7.0),
            rel_standing_envs=0.05,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.35, 0.35),
                lin_vel_y=(-0.25, 0.25),
                ang_vel_z=(-0.60, 0.60),
                heading=None,
            ),
        )

        # --- rewards: null every Bennett custom term trot1/free_gait1 added ---
        self.rewards.base_height_l2 = None
        self.rewards.undesired_contacts = None  # base body_names=".*THIGH" won't match Bennett's lowercase *_thigh
        self.rewards.base_ang_vel_xy_l2 = None
        self.rewards.commanded_yaw_error_l2 = None
        self.rewards.commanded_straight_lateral_yaw_vel_l2 = None
        self.rewards.track_lin_vel_xy_fine_exp = None
        self.rewards.track_ang_vel_z_fine_exp = None
        self.rewards.stand_base_still = None
        self.rewards.stand_joint_still = None
        self.rewards.stand_default_pose = None
        self.rewards.touchdown_impact_l2 = None
        self.rewards.action_second_difference_l2 = None
        self.rewards.termination_penalty = None
        self.rewards.gait_free_stance_feet_slide = None
        self.rewards.gait_free_swing_clearance = None
        self.rewards.minimum_support_contacts = None

        # --- rewards: go2-flat canonical values (override trot1's tuned numbers) ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.25)
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -2.5

        # The emergent-gait driver. Bennett's short steps stay airborne well
        # under the go2 0.5 s threshold, so the threshold is scaled down.
        self.rewards.feet_air_time = RewTerm(
            func=feet_air_time,
            weight=0.25,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
                "threshold": 0.25,
            },
        )


@configclass
class QuadLegFreeGait3FlatEnvCfg_PLAY(QuadLegFreeGait3FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
