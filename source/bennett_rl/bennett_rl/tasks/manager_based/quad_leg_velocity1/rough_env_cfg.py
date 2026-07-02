from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V1


ACTIVE_JOINTS = [
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
]


@configclass
class BennettQuadVelocityRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Bennett quadruped velocity-tracking locomotion task on rough terrain.

    This is a velocity-command policy task. The policy outputs normalized joint-position actions for the eight
    actuated thigh/calf joints; Isaac Lab applies them through the Bennett DCMotorCfg PD actuator model. It is not a
    hardware MIT-command runner.
    """

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 4
        self.sim.dt = 1.0 / 200.0

        self.scene.robot = BENNETT_CFG_V1.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.actions.joint_pos.joint_names = ACTIVE_JOINTS
        self.actions.joint_pos.scale = 0.25

        # Match the flat run at logs/rsl_rl/Bennett_quad_velocity_flat/2026-06-30_20-38-20.
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.08, 0.16)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Sim2Real-oriented observation shape: no direct base linear velocity observation.
        # old: inherited LocomotionVelocityRoughEnvCfg policy.base_lin_vel.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS)
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS)

        # Scale down rough terrains for Bennett's small body/leg dimensions.
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # old: self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_1"
        self.rewards.feet_air_time.weight = 0.05
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -2.0e-5
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.10
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.3
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.02

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class BennettQuadVelocityRoughEnvCfg_PLAY(BennettQuadVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


BennettGo2RoughEnvCfg = BennettQuadVelocityRoughEnvCfg
BennettGo2RoughEnvCfg_PLAY = BennettQuadVelocityRoughEnvCfg_PLAY
