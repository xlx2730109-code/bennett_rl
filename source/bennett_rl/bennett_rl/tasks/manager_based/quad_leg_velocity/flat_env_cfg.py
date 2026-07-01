from isaaclab.utils import configclass

from .rough_env_cfg import BennettQuadVelocityRoughEnvCfg


@configclass
class BennettQuadVelocityFlatEnvCfg(BennettQuadVelocityRoughEnvCfg):
    """Bennett quadruped velocity-tracking locomotion task on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class BennettQuadVelocityFlatEnvCfg_PLAY(BennettQuadVelocityFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


BennettGo2FlatEnvCfg = BennettQuadVelocityFlatEnvCfg
BennettGo2FlatEnvCfg_PLAY = BennettQuadVelocityFlatEnvCfg_PLAY
