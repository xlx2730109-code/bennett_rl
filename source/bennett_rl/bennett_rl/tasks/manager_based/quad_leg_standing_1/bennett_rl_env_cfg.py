"""Bennett pure standing task.

This first-stage task intentionally avoids velocity commands, gait rewards, foot-air-time rewards,
height scans, push disturbances, and domain randomization. The actor observations are limited to
signals that can be provided on the real robot: IMU angular velocity, projected gravity, actuated
joint position/velocity feedback, and previous action.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V3

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

FOOT_BODIES = [".*_1"]
BASE_BODY = ["base"]

ACTION_SCALE = 0.08
TARGET_BASE_HEIGHT = 0.32
SOFT_TORQUE_LIMIT = 8.0
HARD_EFFORT_LIMIT = 12.0


def _make_standing_bennett_cfg() -> ArticulationCfg:
    robot_cfg = BENNETT_CFG_V3.copy()
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.articulation_props.fix_root_link = False
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
    robot_cfg.actuators["base_legs"].effort_limit = HARD_EFFORT_LIMIT
    robot_cfg.actuators["base_legs"].saturation_effort = HARD_EFFORT_LIMIT
    robot_cfg.actuators["base_legs"].stiffness = 40.0
    robot_cfg.actuators["base_legs"].damping = 1.5
    return robot_cfg


@configclass
class QuadLegStandingSceneCfg(InteractiveSceneCfg):
    """Flat-ground Bennett scene for pure standing."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = _make_standing_bennett_cfg()

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


@configclass
class CommandsCfg:
    """No velocity command for pure standing."""

    pass


@configclass
class ActionsCfg:
    """Eight-dimensional actuated joint-position residual action."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTUATED_JOINTS,
        scale=ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Sim-to-real oriented standing observations: 3 + 3 + 8 + 8 + 8 = 30 dims."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-10.0, 10.0), scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-1.0, 1.0))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
            clip=(-1.0, 1.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
            clip=(-20.0, 20.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-1.0, 1.0))

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """First-stage reset only; no push or domain randomization yet."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Standing rewards. Positive terms define the target; negative terms suppress unsafe exploits."""

    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    upright = RewTerm(func=mdp.upright_exp, weight=4.0, params={"sigma": 0.12})
    base_height = RewTerm(
        func=mdp.base_height_exp,
        weight=2.0,
        params={"target_height": TARGET_BASE_HEIGHT, "sigma": 0.04},
    )
    zero_xy_velocity = RewTerm(func=mdp.zero_xy_lin_vel_exp, weight=1.5, params={"sigma": 0.10})
    zero_angular_velocity = RewTerm(func=mdp.zero_ang_vel_exp, weight=1.0, params={"sigma": 0.25})
    default_joint_pose = RewTerm(
        func=mdp.default_joint_pose_exp,
        weight=1.0,
        params={
            "sigma": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
        },
    )
    all_feet_contact = RewTerm(
        func=mdp.all_feet_contact,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES), "threshold": 1.0},
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
            "threshold": 1.0,
            "max_value": 2.0,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_vel = RewTerm(
        func=mdp.bounded_joint_vel,
        weight=-0.03,
        params={"scale": 20.0, "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
    )
    joint_acc = RewTerm(
        func=mdp.bounded_joint_acc,
        weight=-0.01,
        params={"scale": 500.0, "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
    )
    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
    )
    soft_torque_limit = RewTerm(
        func=mdp.soft_torque_limit_l2,
        weight=-5.0e-4,
        params={
            "soft_limit": SOFT_TORQUE_LIMIT,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
        },
    )
    missing_feet_contact = RewTerm(
        func=mdp.missing_feet_contact,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES), "threshold": 1.0},
    )
    undesired_base_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY), "threshold": 1.0},
    )
    fallen = RewTerm(
        func=mdp.is_terminated_term,
        weight=-8.0,
        params={"term_keys": ["base_contact", "low_base_height", "excessive_tilt"]},
    )


@configclass
class TerminationsCfg:
    """Standing failure conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY), "threshold": 1.0},
    )
    low_base_height = DoneTerm(
        func=mdp.base_height_below,
        params={"minimum_height": 0.60 * TARGET_BASE_HEIGHT},
    )
    excessive_tilt = DoneTerm(
        func=mdp.base_tilt_over,
        params={"max_projected_gravity_xy": math.sin(0.75)},
    )


@configclass
class BennettQuadLegStandingEnvCfg(ManagerBasedRLEnvCfg):
    scene: QuadLegStandingSceneCfg = QuadLegStandingSceneCfg(num_envs=4096, env_spacing=1.5, clone_in_fabric=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.sim.dt = 0.005
        self.episode_length_s = 20.0
        self.viewer.eye = (1.8, -2.4, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.min_velocity_iteration_count = 1
        self.sim.physx.max_velocity_iteration_count = 4
        self.sim.physx.enable_external_forces_every_iteration = True
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BennettQuadLegStandingEnvCfg_PLAY(BennettQuadLegStandingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False

# python .\scripts\zero_agent.py --task=Isaac-Bennett-QuadLeg-Standing-v0 --num_envs=20
# python .\scripts\rsl_rl\train.py --task=Isaac-Bennett-QuadLeg-Standing-v0 --headless
# python .\scripts\rsl_rl\play.py --task=Isaac-Bennett-QuadLeg-Standing-Play-v0 --video --checkpoint


# 训第一阶段静态站立保持，不是训走路、摆腿、起立，也不是抗推扰。所以它“一动不动”是当前配置的最优行为。

# 当前任务的目标基本是：
# 保持机身水平：upright
# 保持固定高度：base_height
# 水平速度接近 0：zero_xy_velocity
# 角速度接近 0：zero_angular_velocity
# 关节靠近默认姿态：default_joint_pose
# 四脚接触地面：all_feet_contact
# 动作小、关节速度小、力矩别乱冲

# 这些都关掉了：
# 没有速度 command
# 没有轨迹 command
# 没有 push
# 没有 reset 姿态扰动
# 没有关节扰动
# 动作幅度很小 ACTION_SCALE=0.08
# 所以策略学到的就是：别乱动，输出接近 0 的残差动作，让 PD 按默认姿态站住。


