# 右后单腿，轨迹跟踪


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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V5, BENNETT_CFG_V6

from . import mdp


RR_TRACE_JOINTS = ["RR_thigh", "RR_calf"]  
ACTIVE_LEG_JOINTS = [
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
]


def _make_fixed_bennett_cfg() -> ArticulationCfg:   #作用：生成固定基座Bennett机器人的配置
    robot_cfg = BENNETT_CFG_V5.copy()
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.articulation_props.fix_root_link = True
    return robot_cfg


def _make_fixed_bennett_v4_cfg() -> ArticulationCfg:
    """Create the fixed-base latest Bennett model from Urdf_Bennett_4."""

    # BENNETT_CFG_V6 is the robot-config entry backed by assets/robots/Urdf_Bennett_4.
    robot_cfg = BENNETT_CFG_V6.copy()
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.articulation_props.fix_root_link = True
    return robot_cfg


@configclass
class SingleLegTraceSceneCfg(InteractiveSceneCfg):  # 场景：地面、固定基座 Bennett、灯光
    """Fixed-base Bennett scene for suspended single-leg trace tracking."""

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

    robot: ArticulationCfg = _make_fixed_bennett_cfg()

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class SingleLegTraceV4SceneCfg(SingleLegTraceSceneCfg):
    """Fixed-base single-leg scene using Urdf_Bennett_4."""

    robot: ArticulationCfg = _make_fixed_bennett_v4_cfg()


@configclass
class CommandsCfg: # 无外部 command，轨迹是内部确定的
    """No external command generator; the reference trace is deterministic."""

    pass


@configclass
class ActionsCfg: # RR_thigh/RR_calf 两个动作，带限速
    """Two-dimensional RR thigh/calf action with hardware-like rate limiting."""

    joint_pos = mdp.SingleLegPositionActionCfg(
        class_type=mdp.SingleLegPositionAction,
        asset_name="robot",
        controlled_joint_names=RR_TRACE_JOINTS,
        hold_joint_names=ACTIVE_LEG_JOINTS,
        scale=math.radians(20.0),
        max_joint_speed=math.radians(60.0),
    )


@configclass
class ObservationsCfg:  #作用：生成Bennett机器人的观测配置
    """Observations for suspended single-leg trace tracking."""

    @configclass
    class PolicyCfg(ObsGroup):
        phase = ObsTerm(func=mdp.single_leg_phase)
        reference_offsets = ObsTerm(func=mdp.single_leg_reference_offsets)
        tracking_error = ObsTerm(
            func=mdp.single_leg_tracking_error,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RR_TRACE_JOINTS)},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RR_TRACE_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RR_TRACE_JOINTS)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg: # reset 腿关节
    """Reset all active leg joints to the Bennett default pose."""

    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_LEG_JOINTS),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:   #作用：生成Bennett机器人的奖励配置：tracking、速度、action_rate、torque
    """Rewards for smooth RR single-leg reference tracking."""

    alive = RewTerm(func=mdp.is_alive, weight=0.1)  #
    track_reference = RewTerm(
        func=mdp.single_leg_track_reference_exp,
        weight=5.0,
        params={"sigma": 0.08, "asset_cfg": SceneEntityCfg("robot", joint_names=RR_TRACE_JOINTS)},
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RR_TRACE_JOINTS)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RR_TRACE_JOINTS)},
    )


@configclass
class V4RewardsCfg(RewardsCfg):
    """V4 rewards with a wider physical basin and direct sinusoidal action guidance."""

    track_reference = RewTerm(
        func=mdp.single_leg_track_reference_exp,
        weight=5.0,
        params={"sigma": 0.15, "asset_cfg": SceneEntityCfg("robot", joint_names=RR_TRACE_JOINTS)},
    )
    track_action_reference = RewTerm(
        func=mdp.single_leg_action_track_reference_exp,
        weight=0.5,
        params={"sigma": 0.06, "action_name": "joint_pos"},
    )


@configclass
class TerminationsCfg: # 只按时间结束
    """Only terminate on time-out for this suspended trace task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class BennettSingleLegTraceEnvCfg(ManagerBasedRLEnvCfg): # 单一 50Hz 入口，作用是生成Bennett机器人的环境配置
    scene: SingleLegTraceSceneCfg = SingleLegTraceSceneCfg(num_envs=4096, env_spacing=1.5, clone_in_fabric=False)  
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 6 #50Hz
        self.sim.dt = 1.0 / 300 #50Hz

        # self.decimation = 6 #50Hz
        # self.sim.dt = 1.0 / 300 #50Hz

        # self.decimation = 6 #50Hz
        # self.sim.dt = 1.0 / 300 #50Hz

        # self.decimation = 6 #50Hz
        # self.sim.dt = 1.0 / 300 #50Hz

        # self.decimation = 6 #50Hz
        # self.sim.dt = 1.0 / 300 #50Hz

        self.episode_length_s = 8.0
        self.viewer.eye = (1.8, -2.4, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.sim.render_interval = self.decimation


@configclass
class BennettSingleLegTraceEnvCfg_PLAY(BennettSingleLegTraceEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1 #Play时只有一个环境
        self.scene.env_spacing = 1.5


@configclass
class BennettSingleLegTraceV4EnvCfg(BennettSingleLegTraceEnvCfg):
    """Independent 50 Hz training task using the latest Urdf_Bennett_4 model."""

    scene: SingleLegTraceV4SceneCfg = SingleLegTraceV4SceneCfg(
        num_envs=4096, env_spacing=1.5, clone_in_fabric=False
    )
    rewards: V4RewardsCfg = V4RewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 1.0 / 300.0
        self.decimation = 6
        self.sim.render_interval = self.decimation


@configclass
class BennettSingleLegTraceV4EnvCfg_PLAY(BennettSingleLegTraceV4EnvCfg):
    """Single-environment close-view playback for the Urdf_Bennett_4 task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.5
        self.viewer.eye = (0.8, -1.2, 0.65)
        self.viewer.lookat = (0.0, -0.08, 0.12)



# python .\scripts\zero_agent.py --task=Isaac-Bennett-SingleLeg-RR-Trace-v0 --num_envs=8
# python .\scripts\rsl_rl\train.py --task=Isaac-Bennett-SingleLeg-RR-Trace-v0 --headless
# python .\scripts\rsl_rl\play.py --task Isaac-Bennett-SingleLeg-RR-Trace-Play-v0 --video --checkpoint
# tensorboard --logdir




# 架构
# 通用配置
#   SingleLegTraceSceneCfg       # 场景：地面、固定基座 Bennett、灯光
#   CommandsCfg                  # 无外部 command，轨迹是内部确定的
#   ActionsCfg                   # RR_thigh/RR_calf 两个动作，带限速
#   ObservationsCfg              # phase/reference/error/joint/action
#   EventCfg                     # reset 腿关节
#   RewardsCfg                   # tracking、速度、action_rate、torque
#   TerminationsCfg              # 只按时间结束
