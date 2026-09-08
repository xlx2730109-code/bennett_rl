"""Higher-clearance emergent-gait Bennett locomotion on flat ground.

Identical to ``QuadLegFreeGait3FlatEnvCfg`` (the canonical go2-flat reward /
command port) except for THREE gait-shaping additions, all motivated by the
same sim2real symptom: the previously-trained sim lift (free_gait3) transferred
to the real robot as a foot that dragged / shuffled along the ground.

  * ``swing_foot_clearance`` -- speed-scaled, mid-swing-weighted swing-height
    penalty. v1 charged the whole airborne arc and trained into a
    smoothness tug-of-war; v2 fixed that but its MEAN-over-swinging-feet
    normalization plus the free never-swinging foot produced a
    division-of-labor optimum: one leg high-stepped at 137 mm while the other
    three glued to the floor (play data: RL 62% airborne / FR 92.7% contact).
    v3 normalizes by the FIXED foot count and taxes overshoot above a wide
    free band (1.5x target), so every foot must swing well and flailing is
    no longer free.
  * ``feet_stance_time`` -- anti-glue mirror of the official
    ``feet_air_time``: at each foot's lift-off, tax ``(contact_time -
    threshold)`` with the threshold relaxing from 0.25 s at nominal speed to
    0.6 s at the crawl gate. This is what makes gluing EXPENSIVE -- without
    it a never-swinging foot costs zero clearance.
  * ``feet_slide`` -- the official Isaac Lab stance-foot slide penalty
    (weight -0.10): the hardware failure was a foot sliding along the ground,
    a STANCE fault no swing-height term can reach.

None of the terms add anything to the frozen 33-dim observation (no gait
clock, no phase, no swing flags), so the obs contract is unchanged and
deployable via the same free-gait bridge.
"""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards as velocity_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...quad_leg_trot.quad_leg_trot1.flat_env_cfg import FOOT_BODIES
from ...quad_leg_free_gait.quad_leg_free_gait3.flat_env_cfg import QuadLegFreeGait3FlatEnvCfg

from . import mdp


@configclass
class QuadLegFreeGait4FlatEnvCfg(QuadLegFreeGait3FlatEnvCfg):
    """free_gait3 + speed-scaled swing clearance + anti-glue + stance anti-drag."""

    def __post_init__(self):
        super().__post_init__()

        contact_cfg = SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True)
        foot_cfg = SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True)

        # Speed-scaled, mid-swing-weighted clearance with overshoot tax and
        # fixed-foot-count normalization (see mdp/rewards.py for the history).
        # Weight -0.40 (v3.1): at -0.15 the trained policy equilibrium sat at
        # ~32 mm apexes -- it PAID the clearance tax (flat -0.036 over iters
        # 300-600) because doubling the lift costs more in the quadratic
        # dof_acc / action_rate than the tax saved. -0.40 tips that balance.
        self.rewards.swing_foot_clearance = RewTerm(
            func=mdp.swing_foot_clearance,
            weight=-0.40,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
                "threshold": 1.0,
                "clearance_low": 0.03,
                "clearance_high": 0.07,
                "speed_ref": 0.30,
                "v_foot_ref": 0.5,
                "saturating_k": 40.0,
                "over_free_mult": 1.5,
                "over_norm": 0.05,
                "command_name": "base_velocity",
            },
        )

        # Anti-glue: every foot must release within ~0.25 s at nominal speed
        # (relaxed to 0.6 s at the crawl gate) -- gluing is no longer free.
        self.rewards.feet_stance_time = RewTerm(
            func=mdp.feet_stance_time,
            weight=-0.25,
            params={
                "sensor_cfg": contact_cfg,
                "command_name": "base_velocity",
                "stance_thr_fast": 0.25,
                "stance_thr_slow": 0.60,
                "speed_ref": 0.30,
            },
        )

        # Official stance-foot slide penalty: punishes the dragging/shuffling
        # foot exactly where it happens (in contact).
        self.rewards.feet_slide = RewTerm(
            func=velocity_mdp.feet_slide,
            weight=-0.10,
            params={
                "sensor_cfg": contact_cfg,
                "asset_cfg": foot_cfg,
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
