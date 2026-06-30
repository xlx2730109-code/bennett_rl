# 固定基座四腿轨迹跟踪任务

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

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V3

from . import mdp


QUAD_TRACE_JOINTS = [
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
]


def _make_fixed_bennett_cfg() -> ArticulationCfg:
    robot_cfg = BENNETT_CFG_V3.copy()
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.articulation_props.fix_root_link = True
    return robot_cfg


@configclass
class QuadLegTrackSceneCfg(InteractiveSceneCfg):
    """Fixed-base Bennett scene for suspended four-leg trace tracking."""

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
class CommandsCfg:
    """No external command generator; the reference trace is deterministic."""

    pass


@configclass
class ActionsCfg:
    """Two-dimensional coordinated thigh/calf action with hardware-like rate limiting."""

    joint_pos = mdp.QuadLegPositionActionCfg(
        class_type=mdp.QuadLegPositionAction,
        asset_name="robot",
        controlled_joint_names=QUAD_TRACE_JOINTS,
        hold_joint_names=QUAD_TRACE_JOINTS,
        scale=math.radians(20.0),
        max_joint_speed=math.radians(60.0),
    )


@configclass
class ObservationsCfg:
    """Observations for suspended four-leg trace tracking."""

    @configclass
    class PolicyCfg(ObsGroup):
        phase = ObsTerm(func=mdp.quad_leg_phase)
        reference_offsets = ObsTerm(func=mdp.quad_leg_reference_offsets)
        tracking_error = ObsTerm(
            func=mdp.quad_leg_tracking_error,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=QUAD_TRACE_JOINTS, preserve_order=True)},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=QUAD_TRACE_JOINTS, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=QUAD_TRACE_JOINTS, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset all active leg joints to the Bennett default pose."""

    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=QUAD_TRACE_JOINTS, preserve_order=True),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Rewards for smooth four-leg reference tracking."""

    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    track_fl = RewTerm(
        func=mdp.quad_leg_track_reference_exp,
        weight=1.25,
        params={
            "sigma": 0.08,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_thigh", "FL_calf"], preserve_order=True),
        },
    )
    track_fr = RewTerm(
        func=mdp.quad_leg_track_reference_exp,
        weight=1.25,
        params={
            "sigma": 0.08,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["FR_thigh", "FR_calf"], preserve_order=True),
        },
    )
    track_rl = RewTerm(
        func=mdp.quad_leg_track_reference_exp,
        weight=1.25,
        params={
            "sigma": 0.08,  #作用是：设置跟踪误差的权重
            "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_thigh", "RL_calf"], preserve_order=True),
        },
    )
    track_rr = RewTerm(
        func=mdp.quad_leg_track_reference_exp,
        weight=1.25,
        params={
            "sigma": 0.08,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["RR_thigh", "RR_calf"], preserve_order=True),
        },
    )
    track_action_reference = RewTerm(   #看 policy 输出的 2 维基础动作有没有学到参考轨迹
        func=mdp.quad_leg_action_track_reference_exp,
        weight=0.5,
        params={"sigma": 0.06, "action_name": "joint_pos"},
    )
    joint_vel = RewTerm(    #太负说明关节速度大，动作可能急。
        func=mdp.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=QUAD_TRACE_JOINTS, preserve_order=True)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)    #越接近 0 越平滑
    torques = RewTerm(  #太负说明关节力矩大，动作可能急。真机更容易发热、抖、撞限位。
        func=mdp.joint_torques_l2,
        weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=QUAD_TRACE_JOINTS, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """Only terminate on time-out for this suspended trace task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class BennettQuadLegTrackEnvCfg(ManagerBasedRLEnvCfg):
    scene: QuadLegTrackSceneCfg = QuadLegTrackSceneCfg(num_envs=4096, env_spacing=1.5, clone_in_fabric=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 6
        self.sim.dt = 1.0 / 300.0

        self.episode_length_s = 8.0
        self.viewer.eye = (1.8, -2.4, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.sim.render_interval = self.decimation


@configclass
class BennettQuadLegTrackEnvCfg_PLAY(BennettQuadLegTrackEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.5




# python .\scripts\zero_agent.py --task=Isaac-Bennett-QuadLeg-Track-v0 --num_envs=8
# python .\scripts\rsl_rl\train.py --task=Isaac-Bennett-QuadLeg-Track-v0 --headless
# python .\scripts\rsl_rl\play.py --task=Isaac-Bennett-QuadLeg-Track-Play-v0 --video --checkpoint
# tensorboard --logdir


# obs =
# [
#   phase_sin, phase_cos,                 # 2

#   reference_offsets[8],                 # 8
#   tracking_error[8],                    # 8
#   real_offsets[8],                      # 8
#   real_vels[8],                         # 8

#   last_action[2],                       # 2
# ]
# 总计 2 + 8 + 8 + 8 + 8 + 2 = 36