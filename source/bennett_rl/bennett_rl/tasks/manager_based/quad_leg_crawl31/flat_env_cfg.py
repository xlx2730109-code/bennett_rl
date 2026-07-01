"""Flat-ground Bennett crawl environment with gait-scheduler observations."""
# 加键盘控制，真机部署可以实现静止速度控制,加大了扰动

# 开头导入 Isaac Lab 的配置类、传感器、terrain、robot asset，以及本任务自己的 mdp。
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V1

from . import mdp


# 重要常量：
# ACTUATED_JOINTS
# FOOT_BODIES
# ACTION_SCALE
# CRAWL_FREQUENCY_HZ
# CRAWL_DUTY_FACTOR
# CRAWL_SWING_HEIGHT
# CRAWL_VX
# 它们定义：
# 哪 8 个关节由 policy 控制
# 哪些 body 是脚
# 动作幅度是多少
# crawl 步态频率
# 支撑占空比
# 摆腿高度
# 前进速度目标

ACTUATED_JOINTS = [ #定义哪 8 个关节由 policy 控制
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",


    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
]

FOOT_BODIES = ["FL_1", "FR_1", "RL_1", "RR_1"]  #定义哪些 body 是脚
BASE_BODY = ["base"]    #定义 base body

#影响 policy 能让关节偏离默认姿态的幅度，间接影响腿的摆动幅度。
# policy 输出一般在 [-1, 1]，所以当前：ACTION_SCALE = 0.20 rad ≈ 11.46 deg。 1 rad ≈ 57°17'45''
ACTION_SCALE = 0.30 #定义动作幅度（增大，让腿有更大活动范围）
CRAWL_FREQUENCY_HZ = 0.50   #crawl 步态频率
CRAWL_DUTY_FACTOR = 0.80    #支撑占空比（降低，给摆腿留更多时间）
CRAWL_SWING_HEIGHT = 0.08  #定义摆腿高度
CRAWL_VX = 0.10 #定义前进速度目标
TARGET_BASE_HEIGHT = 0.32 #约束 base 高度，防止趴低后用撑杆式步态取巧
COMMAND_DEADBAND = 0.025  #速度绝对值小于该阈值时，训练成四脚站立不摆腿
STANDING_COMMAND_PROB = 0.35  #显式零速度采样比例：让 policy 学会“不按键=不动”
FRICTION_STATIC_RANGE = (0.6, 1.3)  #随机摩擦
FRICTION_DYNAMIC_RANGE = (0.5, 1.1)
BASE_MASS_SCALE_RANGE = (0.90, 1.10)    #随机 base 质量
# BASE_MASS_SCALE_RANGE = (1.60, 1.60)
BASE_COM_RANGE = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)}
# BASE_COM_RANGE = {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (-0.02, 0.02)}
ACTUATOR_STIFFNESS_SCALE_RANGE = (0.80, 1.20)
ACTUATOR_DAMPING_SCALE_RANGE = (0.70, 1.30)


# Robot 配置
# 这块从 BENNETT_CFG_V~ 拷贝机器人配置，然后针对 crawl 任务改几个参数：
# 设置 robot 的 prim path
# 允许 root 自由运动：fix_root_link = False
# 设置电机力矩上限、刚度、阻尼
# 使用 Bennett 的默认站姿
# 这里决定了机器人本体、初始姿态和 actuator 参数。
def _make_crawl_bennett_cfg() -> ArticulationCfg:
    robot_cfg = BENNETT_CFG_V1.copy()
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.articulation_props.fix_root_link = False
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
    robot_cfg.actuators["base_legs"].effort_limit = 8.0
    robot_cfg.actuators["base_legs"].saturation_effort = 20.0
    robot_cfg.actuators["base_legs"].stiffness = 40.0   #提高刚度，让 stance 腿更能抵抗倾斜力矩
    robot_cfg.actuators["base_legs"].damping = 2        #增大阻尼，减少抖动
    return robot_cfg

# Scene 场景配置，这块定义仿真世界里有什么，即这块负责“环境里摆什么东西”。
@configclass
class BennettQuadCrawlFlatSceneCfg(InteractiveSceneCfg):
    """Flat-ground Bennett scene for crawl observation bring-up."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",   #平地 plane
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

    robot: ArticulationCfg = _make_crawl_bennett_cfg()  #Bennett 机器人

    contact_forces = ContactSensorCfg(  #接触传感器，用来判断脚有没有接触地面
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    dome_light = AssetBaseCfg(  #光照
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


@configclass
class CommandsCfg:  #命令随机化
    """Low-speed forward velocity command for training keyboard-controllable crawl."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=STANDING_COMMAND_PROB,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.16, 0.16),    #随机采样速度命令，范围 ±0.16 m/s
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:   #动作配置
    """Eight-dimensional actuated joint-position residual action."""

    # 意思是 policy 输出 8 维动作，对应 8 个关节的位置残差：
    # FL_thigh, FL_calf,
    # FR_thigh, FR_calf,
    # RL_thigh, RL_calf,
    # RR_thigh, RR_calf
    # scale=ACTION_SCALE 表示动作 [-1, 1] 会被放大成大约 ±0.20 rad 的关节目标偏移。
    # use_default_offset=True 表示动作是相对默认站姿的偏移，不是绝对关节角。
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTUATED_JOINTS,
        scale=ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:  #观测配置，当前总维度是 50
    """Sim-to-real oriented crawl observations.

    Term dimensions:
        base_ang_vel 3, projected_gravity 3, fixed_velocity_command 3,
        joint_pos 8, joint_vel 8, last_action 8, global_phase 2,
        leg_phase 8, desired_contacts 4, gait_params 3. Total: 50.
    """

    @configclass
    class PolicyCfg(ObsGroup):  #观测噪声
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.10, n_max=0.10),
            clip=(-10.0, 10.0),
            scale=0.2,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-1.0, 1.0),   
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.50, n_max=0.50),
            clip=(-20.0, 20.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-1.0, 1.0))
        crawl_phase = ObsTerm(  #当前 crawl 周期走到哪了
            func=mdp.commanded_crawl_global_phase_sin_cos,
            params={
                "frequency_hz": CRAWL_FREQUENCY_HZ,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        crawl_leg_phase = ObsTerm(  #每条腿处于什么相位
            func=mdp.commanded_crawl_leg_phase_sin_cos,
            params={
                "frequency_hz": CRAWL_FREQUENCY_HZ,
                "duty_factor": CRAWL_DUTY_FACTOR,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        desired_contacts = ObsTerm( #哪条腿当前应该支撑，哪条腿当前应该摆动
            func=mdp.commanded_crawl_desired_contacts,
            params={
                "frequency_hz": CRAWL_FREQUENCY_HZ,
                "duty_factor": CRAWL_DUTY_FACTOR,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )
        gait_params = ObsTerm(  #当前步态频率、占空比、摆腿高度是多少
            func=mdp.commanded_crawl_gait_params,
            params={
                "frequency_hz": CRAWL_FREQUENCY_HZ,
                "duty_factor": CRAWL_DUTY_FACTOR,
                "swing_height": CRAWL_SWING_HEIGHT,
                "command_name": "base_velocity",
                "command_deadband": COMMAND_DEADBAND,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg: #重置配置
    """Minimal deterministic reset for flat crawl observation bring-up."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": FRICTION_STATIC_RANGE,
            "dynamic_friction_range": FRICTION_DYNAMIC_RANGE,
            "restitution_range": (0.0, 0.02),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY),
            "mass_distribution_params": BASE_MASS_SCALE_RANGE,
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY),
            "com_range": BASE_COM_RANGE,
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
            "stiffness_distribution_params": ACTUATOR_STIFFNESS_SCALE_RANGE,
            "damping_distribution_params": ACTUATOR_DAMPING_SCALE_RANGE,
            "operation": "scale",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY),
            "force_range": (-3.0, 3.0),
            "torque_range": (-0.5, 0.5),
        },
    )

    reset_base = EventTerm( #重置机身位置、姿态、速度
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.10, 0.10)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    reset_joints = EventTerm(   #重置 8 个关节到默认姿态附近
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
            "position_range": (-0.05, 0.05),  #小范围随机初始关节偏移，防止完全对称过拟合
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm( #训练中随机轻推，增加鲁棒性
        func=mdp.push_by_setting_velocity,
        mode="interval",
        params={"velocity_range": {"x": (-0.45, 0.45), "y": (-0.35, 0.35), "yaw": (-0.25, 0.25)}},
        interval_range_s=(6.0, 12.0),
    )


@configclass
class RewardsCfg:   #奖励配置
    """Flat crawl rewards conditioned on the scheduler."""
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    track_forward = RewTerm(    #向前速度跟踪 command manager 采样的低速命令
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.10},
    )
    base_height = RewTerm(  #保持 base 高度，防止趴低拖拽
        func=mdp.base_height_exp,
        weight=1.5,
        params={"target_height": TARGET_BASE_HEIGHT, "sigma": 0.04},
    )
    default_joint_pose = RewTerm(   #防止腿大幅偏离默认姿态，抑制撑杆式伸腿
        func=mdp.default_joint_pose_exp,
        weight=0.8,
        params={
            "sigma": 0.18,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
        },
    )
    crawl_contact_match = RewTerm(  #脚接触状态匹配 scheduler (降低权重，与下面两项有重叠)
        func=mdp.crawl_contact_match,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
            "frequency_hz": CRAWL_FREQUENCY_HZ,
            "duty_factor": CRAWL_DUTY_FACTOR,
            "threshold": 1.0,
            "command_name": "base_velocity",
            "command_deadband": COMMAND_DEADBAND,
        },
    )
    missing_stance_contacts = RewTerm(  #惩罚错误步态，该支撑的脚没落地
        func=mdp.crawl_missing_stance_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
            "frequency_hz": CRAWL_FREQUENCY_HZ,
            "duty_factor": CRAWL_DUTY_FACTOR,
            "threshold": 1.0,
            "command_name": "base_velocity",
            "command_deadband": COMMAND_DEADBAND,
        },
    )
    extra_swing_contacts = RewTerm( #严重惩罚错误步态——该摆动的脚还拖地（站立时此惩罚使不动≈不划算）
        func=mdp.crawl_extra_swing_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
            "frequency_hz": CRAWL_FREQUENCY_HZ,
            "duty_factor": CRAWL_DUTY_FACTOR,
            "threshold": 1.0,
            "command_name": "base_velocity",
            "command_deadband": COMMAND_DEADBAND,
        },
    )
    swing_foot_clearance = RewTerm( #摆动脚抬起来
        func=mdp.crawl_swing_foot_clearance,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
            "frequency_hz": CRAWL_FREQUENCY_HZ,
            "duty_factor": CRAWL_DUTY_FACTOR,
            "target_height": CRAWL_SWING_HEIGHT,
            "command_name": "base_velocity",
            "command_deadband": COMMAND_DEADBAND,
        },
    )
    stance_feet_slide = RewTerm(    #惩罚错误步态，支撑脚在地上滑
        func=mdp.crawl_stance_feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
            "frequency_hz": CRAWL_FREQUENCY_HZ,
            "duty_factor": CRAWL_DUTY_FACTOR,
            "threshold": 1.0,
            "max_value": 2.0,
            "command_name": "base_velocity",
            "command_deadband": COMMAND_DEADBAND,
        },
    )
    stand_base_still = RewTerm(  #零速度命令时 base 不应平移
        func=mdp.stand_still_base_vel_l2,
        weight=-1.5,
        params={"command_name": "base_velocity", "command_deadband": COMMAND_DEADBAND},
    )
    stand_joint_still = RewTerm(  #零速度命令时关节不要持续摆动
        func=mdp.stand_still_joint_vel_l2,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "command_deadband": COMMAND_DEADBAND,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
        },
    )
    stand_default_pose = RewTerm(  #零速度命令时回到默认站姿附近
        func=mdp.stand_still_joint_pose_exp,
        weight=0.8,
        params={
            "command_name": "base_velocity",
            "command_deadband": COMMAND_DEADBAND,
            "sigma": 0.12,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True),
        },
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)   #保持 base 水平（加重惩罚）
    pitch_l1 = RewTerm(func=mdp.pitch_l1, weight=-5.0)   #俯仰重锤！10°→-0.87/步，15°→-1.29/步
    base_ang_vel_xy = RewTerm(func=mdp.base_ang_vel_xy_l2, weight=-0.3)    #允许行走时轻微晃动
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)     #不要上下乱跳
    lateral_yaw_vel = RewTerm(func=mdp.lateral_yaw_vel_l2, weight=-0.5)    #别横移/乱转
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)  #适度平滑（太大会让腿不敢抬）
    joint_vel = RewTerm(   #关节速度惩罚（减轻，不压制必要的快速摆腿）
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
    )
    torques = RewTerm(  #别用太大力矩
        func=mdp.joint_torques_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTUATED_JOINTS, preserve_order=True)},
    )


@configclass
class TerminationsCfg:
    """Basic failure conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(    #倾斜过大直接终止
        func=mdp.bad_orientation,
        params={"limit_angle": 0.3},  #约 17°（原来28°太松了）
    )


@configclass
class BennettQuadCrawlFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Flat crawl environment with scheduler-conditioned observations."""

    scene: BennettQuadCrawlFlatSceneCfg = BennettQuadCrawlFlatSceneCfg(
        num_envs=4096, env_spacing=1.5, clone_in_fabric=False
    )
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
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BennettQuadCrawlFlatEnvCfg_PLAY(BennettQuadCrawlFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
