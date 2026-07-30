# 四足站立策略，给一些扰动，然后机器人在如外力干扰下保持站立


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

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V1

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

ACTION_SCALE = 0.20   #作用是比例因子，用于缩放动作值，每个主动关节最多在默认角附近偏移约 ±0.20 rad。
TARGET_BASE_HEIGHT = 0.32
SOFT_TORQUE_LIMIT = 8.0
HARD_EFFORT_LIMIT = 20.0


def _make_standing_bennett_cfg() -> ArticulationCfg:
    robot_cfg = BENNETT_CFG_V1.copy()
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
    """Perturbation events for testing the V2 default standing pose."""

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY),
            "force_range": (-8.0, 8.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                # "roll": (math.radians(-8.0), math.radians(8.0)),
                # "pitch": (math.radians(-8.0), math.radians(8.0)),
                # 改 reset 姿态扰动，想更难就改成：
                # "roll": (math.radians(-12.0), math.radians(12.0)),
                # "pitch": (math.radians(-12.0), math.radians(12.0)),
                "roll": (math.radians(-20.0), math.radians(20.0)),
                "pitch": (math.radians(-20.0), math.radians(20.0)),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "z": (0.0, 0.0),
                # "x": (-0.08, 0.08),
                # "y": (-0.08, 0.08),
                # "roll": (-0.15, 0.15),
                # "pitch": (-0.15, 0.15),
                # "yaw": (-0.15, 0.15),
                # 改 reset 初速度扰动，想明显一点，可以试：
                # "x": (-0.15, 0.15),
                # "y": (-0.15, 0.15),
                # "roll": (-0.25, 0.25),
                # "pitch": (-0.25, 0.25),
                # "yaw": (-0.25, 0.25),

                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.35, 0.35),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
            # "position_range": (-0.05, 0.05),
            # "velocity_range": (-0.05, 0.05),
            # 改关节扰动，表示 reset 时关节在默认角附近随机偏移。可以逐步加到：
            # "position_range": (-0.30, 0.30),
            # "velocity_range": (-0.30, 0.30),
            "position_range": (-0.60, 0.60),
            "velocity_range": (-0.60, 0.60),

        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        # interval_range_s=(1.5, 2.5),
        # params={"velocity_range": {"x": (-0.35, 0.35), "y": (-0.35, 0.35), "yaw": (-0.45, 0.45)}},
        # 改训练过程中的推，每隔 1.5~2.5s，直接给 base 一个随机速度冲击。不明显，可以试：
        # interval_range_s=(1.0, 2.0),
        # params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.7, 0.7)}},
        interval_range_s=(0.5, 1.5),
        params={"velocity_range": {"x": (-0.65, 0.65), "y": (-0.65, 0.65), "yaw": (-0.9, 0.9)}},
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
    zero_xy_velocity = RewTerm(func=mdp.zero_xy_lin_vel_exp, weight=0.8, params={"sigma": 0.18})
    zero_angular_velocity = RewTerm(func=mdp.zero_ang_vel_exp, weight=0.6, params={"sigma": 0.35})
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
    torque_fl_thigh = RewTerm(
        func=mdp.joint_abs_torque,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FL_thigh"], preserve_order=True)},
    )
    torque_fl_calf = RewTerm(
        func=mdp.joint_abs_torque,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FL_calf"], preserve_order=True)},
    )
    torque_fr_thigh = RewTerm(
        func=mdp.joint_abs_torque,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FR_thigh"], preserve_order=True)},
    )
    torque_fr_calf = RewTerm(
        func=mdp.joint_abs_torque,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FR_calf"], preserve_order=True)},
    )
    torque_rl_thigh = RewTerm(
        func=mdp.joint_abs_torque,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["RL_thigh"], preserve_order=True)},
    )
    torque_rl_calf = RewTerm(
        func=mdp.joint_abs_torque,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["RL_calf"], preserve_order=True)},
    )
    torque_rr_thigh = RewTerm(
        func=mdp.joint_abs_torque,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["RR_thigh"], preserve_order=True)},
    )
    torque_rr_calf = RewTerm(
        func=mdp.joint_abs_torque,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["RR_calf"], preserve_order=True)},
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

# python .\scripts\zero_agent.py --task=Isaac-Bennett-QuadLeg-Standing_2-v0 --num_envs=20
# python .\scripts\rsl_rl\train.py --task=Isaac-Bennett-QuadLeg-Standing_2-v0 --headless
# python .\scripts\rsl_rl\play.py --task=Isaac-Bennett-QuadLeg-Standing_2-Play-v0 --video --checkpoint


# 1. base_ang_vel        3维：机身角速度 xyz，clip [-10,10]，scale 0.2
# 2. projected_gravity  3维：重力方向投影到机身坐标系，clip [-1,1]
# 3. joint_pos_rel      8维：8个主动关节相对默认角度的位置
# 4. joint_vel_rel      8维：8个主动关节速度，clip [-20,20]，scale 0.05
# 5. last_action        8维：上一帧策略输出动作

# 干扰不是越大越好。更准确地说：干扰要覆盖真机会遇到的扰动，但不能远超机器人可恢复范围。
# 干扰加大有好处：能让策略更抗推、抗初始姿态误差、抗速度扰动。但太大会带来反效果：
# 策略学得很僵，靠大力矩硬顶；
# 名义站立变差，没扰动时也不自然；
# 电机力矩更接近限制，真机容易发热、抖动或保护；
# 仿真里能扛住的冲击，真机因为延迟、摩擦、间隙、速度噪声反而扛不住；
# 训练会优化“极端恢复”，牺牲正常站稳质量。

# Episode_Termination/time_out 越接近 1 越好
# Episode_Termination/low_base_height 越接近 0 越好
# Episode_Termination/excessive_tilt 越接近 0 越好
# Mean episode length 越接近 1000 越好



# base_ang_vel 3
# projected_gravity 3
# joint_pos 8
# joint_vel 8
# last_action 8