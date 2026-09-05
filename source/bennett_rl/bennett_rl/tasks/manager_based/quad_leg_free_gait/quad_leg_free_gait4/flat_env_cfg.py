"""Higher-clearance emergent-gait Bennett locomotion on flat ground.

Identical to ``QuadLegFreeGait3FlatEnvCfg`` (the canonical go2-flat reward /
command port) except for ONE addition: a one-sided saturating swing-foot
clearance penalty that nudges the policy to lift each swinging foot to a
deliberately generous height. The reason is sim2real: the previously-trained
sim lift (free_gait3's target) transferred to the real robot as a foot that
dragged / shuffled along the ground. Doubling the sim clearance gives the real
foot margin so it still clears the floor after the lift drop.

  * ``min_clearance = 0.07`` ~= 2x the old sim-target 0.035. This is baseline A
    the user asked for (0.035 doubled), NOT trot1's value.
  * Shape = one-sided saturating ``(1 - exp(-k * shortfall))`` (arXiv:2403.10723),
    not free_gait1's narrow Gaussian -- a small change in *height* here changes
    the reward smoothly and never saturates to zero gradient, which is exactly
    what the old Gaussian got wrong.

Note: the swing-clearance reward is a PURE sim-terrain construct (it reads foot
world-Z and contact); it does NOT add anything to the frozen 33-dim observation
(no gait clock, no phase, no swing flags), so the obs contract is unchanged and
deployable via the same free-gait bridge.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...quad_leg_trot.quad_leg_trot1.flat_env_cfg import COMMAND_DEADBAND, FOOT_BODIES
from ...quad_leg_free_gait.quad_leg_free_gait3.flat_env_cfg import QuadLegFreeGait3FlatEnvCfg

from . import mdp


@configclass
class QuadLegFreeGait4FlatEnvCfg(QuadLegFreeGait3FlatEnvCfg):
    """free_gait3 + higher swing clearance for sim2real transfer margin."""

    def __post_init__(self):
        super().__post_init__()

        contact_cfg = SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True)
        foot_cfg = SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True)

        # Anti-drag clearance (arXiv:2403.10723 form). The policy is penalised for
        # any swinging foot below min_clearance, and the penalty saturates at the
        # weight via `1 - exp(-k*shortfall)` so it never pushes an over-lift kick.
        self.rewards.swing_foot_clearance = RewTerm(
            func=mdp.swing_foot_clearance,
            weight=-0.15,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "threshold": 1.0,
                "min_clearance": 0.07,
                "saturating_k": 40.0,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )


@configclass
class QuadLegFreeGait4FlatEnvCfg_PLAY(QuadLegFreeGait4FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
