"""Motor-model A/B variants of the unchanged Bennett Trot1 task.

A reproduces the older conservative actuator settings.
B uses the DM-J8006-2EC V1.1 24 V datasheet candidate from:
    assets/motor/dm_j8006_2ec_v1_1_24v.yaml

No reward, observation, gait, command, action, PD, event, terrain, or
termination setting is changed by this package.
"""

from isaaclab.utils import configclass

from ..quad_leg_trot1.flat_env_cfg import QuadLegTrot1FlatEnvCfg


MOTOR_A = {
    "effort_limit": 7.0,
    "saturation_effort": 12.0,
    "velocity_limit": 20.0,
}
MOTOR_B = {
    "effort_limit": 8.0,
    "saturation_effort": 20.0,
    "velocity_limit": 19.8967534727,
}


def _apply_motor_variant(cfg: QuadLegTrot1FlatEnvCfg, values: dict[str, float]) -> None:
    actuator = cfg.scene.robot.actuators["base_legs"]
    actuator.effort_limit = values["effort_limit"]
    actuator.saturation_effort = values["saturation_effort"]
    actuator.velocity_limit = values["velocity_limit"]


def _configure_play(cfg: QuadLegTrot1FlatEnvCfg) -> None:
    cfg.scene.num_envs = 5
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


@configclass
class QuadLegTrotMotorABACfg(QuadLegTrot1FlatEnvCfg):
    """Variant A: older 7 Nm / 12 Nm / 20 rad/s settings."""

    def __post_init__(self):
        super().__post_init__()
        _apply_motor_variant(self, MOTOR_A)


@configclass
class QuadLegTrotMotorABACfg_PLAY(QuadLegTrotMotorABACfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class QuadLegTrotMotorABBCfg(QuadLegTrot1FlatEnvCfg):
    """Variant B: datasheet candidate 8 Nm / 20 Nm / 19.8968 rad/s."""

    def __post_init__(self):
        super().__post_init__()
        _apply_motor_variant(self, MOTOR_B)


@configclass
class QuadLegTrotMotorABBCfg_PLAY(QuadLegTrotMotorABBCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)
