"""Identical PPO settings with separate log namespaces for motor A and B."""

from isaaclab.utils import configclass

from ...quad_leg_trot1.agents.rsl_rl_ppo_cfg import QuadLegTrot1FlatPPORunnerCfg


@configclass
class QuadLegTrotMotorABARunnerCfg(QuadLegTrot1FlatPPORunnerCfg):
    experiment_name = "quad_leg_trot/quad_leg_trot-motor-ab/a_current"


@configclass
class QuadLegTrotMotorABBRunnerCfg(QuadLegTrot1FlatPPORunnerCfg):
    experiment_name = "quad_leg_trot/quad_leg_trot-motor-ab/b_datasheet"
