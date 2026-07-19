from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 用于动态拼接当前文件所在目录下的子文件或文件夹的绝对路径
# Path(__file__)：将当前执行的 Python 脚本文件封装为路径对象。
# .resolve()：将相对路径转换为绝对路径，并解析任何符号链接。
# .parent：获取该文件所在的父级目录路径（即文件夹）。
# / "..."：使用 / 运算符（Python 路径拼接语法）将父目录与目标子文件/夹 ... 组合。
BENNETT_USD_PATH = str(Path(__file__).resolve().parent / "Urdf_Bennett_1" / "urdf" / "Urdf_Bennett_1.usd")
# BENNETT_USD_PATH = str(Path(__file__).resolve().parent / "Urdf_Bennett_2" / "urdf" / "Urdf_Bennett_2.usd")
# Keep the newly rebuilt asset opt-in so existing Bennett tasks retain their
# original dynamics. Go2-8 is currently the only task using BENNETT_CFG_V5.
BENNETT_3_USD_PATH = str(Path(__file__).resolve().parent / "Urdf_Bennett_3" / "urdf" / "Urdf_Bennett_3.usd")



def _make_bennett_cfg(usd_path: str, actuator_cfg: dict, joint_pos: dict) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_thigh", ".*_calf"],
                **actuator_cfg,
            ),
        },
    )


BENNETT_CFG_V1 = _make_bennett_cfg(
    usd_path=BENNETT_USD_PATH,
    joint_pos={
        "FL_thigh": +0.14, 
        "FR_thigh": -0.14,
        "RL_thigh": +0.14,
        "RR_thigh": -0.14,
        "FL_calf": -0.28,
        "FR_calf": -0.28,
        "RL_calf": -0.28,
        "RR_calf": -0.28,
    },
    actuator_cfg=dict(
        effort_limit=8,
        saturation_effort=8,
        velocity_limit=30.0,
        stiffness=40.0,
        damping=1.5,
        friction=0,
    ),
)


BENNETT_CFG_V2 = _make_bennett_cfg(
    usd_path=BENNETT_USD_PATH,
    joint_pos={
        "FL_thigh": 0.3,
        "FR_thigh": -0.3,
        "RL_thigh": 0.3,
        "RR_thigh": -0.3,
        "FL_calf": -0.6,
        "FR_calf": -0.6,
        "RL_calf": -0.6,
        "RR_calf": -0.6,
    },
    actuator_cfg=dict(
        effort_limit=8,
        saturation_effort=8,
        velocity_limit=30.0,
        stiffness=40.0,
        damping=1.5,
        friction=0,
    ),
)


BENNETT_CFG_V3 = _make_bennett_cfg(
    usd_path=BENNETT_USD_PATH,
    joint_pos={
        
    },
    actuator_cfg=dict(
        effort_limit=8,
        saturation_effort=8,
        velocity_limit=30.0,
        stiffness=40.0,
        damping=1.5,
        friction=0,
    ),
)


BENNETT_CFG_V4 = _make_bennett_cfg(
    usd_path=BENNETT_USD_PATH,
    joint_pos={
        "FL_thigh": +0.12,
        "FR_thigh": -0.12,
        "RL_thigh": +0.12,
        "RR_thigh": -0.12,
        "FL_calf": -0.24,
        "FR_calf": -0.24,
        "RL_calf": -0.24,
        "RR_calf": -0.24,
    },
    actuator_cfg=dict(
        effort_limit=15,
        saturation_effort=15,
        velocity_limit=30.0,
        stiffness=30.0,
        damping=2,
        friction=0,
    ),
)


BENNETT_CFG_V5 = _make_bennett_cfg(
    # old: usd_path=BENNETT_USD_PATH,
    usd_path=BENNETT_3_USD_PATH,
    joint_pos={
        "FL_thigh": +0.1,
        "FR_thigh": -0.1,
        "RL_thigh": +0.1,
        "RR_thigh": -0.1,
        "FL_calf": -0.2,
        "FR_calf": -0.2,
        "RL_calf": -0.2,
        "RR_calf": -0.2,
    },
    actuator_cfg=dict(
        effort_limit=17,
        saturation_effort=17,
        velocity_limit=30.0,
        stiffness=30.0,
        damping=2,
        friction=0,
    ),
)
