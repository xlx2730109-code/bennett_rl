# Bennett Crawl Gap Analysis

日期：2026-06-30

范围：只读审计当前 `source/bennett_rl/bennett_rl/tasks/manager_based/quad_leg_velocity` 任务，判断它距离“超慢速、静态 crawl、Sim2Real 优先”方案还差什么。没有修改训练代码、机器人资产、配置或结果文件。

## 结论

当前 `quad_leg_velocity` 仍然是普通 velocity tracking 任务的 Bennett 适配版，不是 crawl 任务。它已经做了一些 Sim2Real 友好的改动，例如去掉 actor 的 `base_lin_vel`、限定 8 个主动关节、调整 foot body regex 和降低 rough terrain 高度；但它没有 gait scheduler、没有 desired contact、没有三足支撑约束、没有 crawl 接触奖励，也没有训练端到部署端的 observation/action metadata 校验。

最关键的下一步不是直接调 reward，而是新建一个 Bennett crawl 专用任务包，在其中先实现并测试 gait scheduler。不要继续把现有 velocity task 的速度范围改小后当作 crawl 使用。

## 当前任务结构

任务注册在 `source/bennett_rl/bennett_rl/tasks/manager_based/quad_leg_velocity/__init__.py`：

- `Isaac-BennettRL-QuadVelocity-Flat-v0`
- `Isaac-BennettRL-QuadVelocity-Flat-Play-v0`
- `Isaac-BennettRL-QuadVelocity-Rough-v0`
- `Isaac-BennettRL-QuadVelocity-Rough-Play-v0`

环境结构：

- `rough_env_cfg.py` 继承官方 `LocomotionVelocityRoughEnvCfg`。
- `flat_env_cfg.py` 继承 Bennett rough 配置，再关闭 terrain generator 和 height scan。
- `mdp/rewards.py` 目前没有自定义 reward，只是说明当前使用 Isaac Lab 内置 locomotion velocity MDP。
- `agents/rsl_rl_ppo_cfg.py` 配置普通 PPO。

这说明当前任务主体仍由官方 velocity task 决定：command、action、observation、reward、termination 的大部分逻辑来自 `D:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\velocity_env_cfg.py` 和 Isaac Lab 内置 MDP。

## Robot / Joint Mapping

当前 velocity 任务使用：

```python
ACTIVE_JOINTS = [
    "FL_thigh", "FL_calf",
    "FR_thigh", "FR_calf",
    "RL_thigh", "RL_calf",
    "RR_thigh", "RR_calf",
]
```

动作配置把 `joint_pos.joint_names` 设为上述 8 个关节，`scale=0.25`，仍使用官方 `JointPositionActionCfg` 的 `use_default_offset=True`。

动作公式为：

```text
q_target = default_joint_pos + raw_action * 0.25 rad
```

重要风险：

- `rough_env_cfg.py` 没有给 action 和 observation 的 `SceneEntityCfg` 设置 `preserve_order=True`。
- 其他 Bennett 任务如 `quad_leg_track` 和 `quad_leg_standing_2` 已显式使用 `preserve_order=True`。
- Crawl / Sim2Real 必须固定训练端、ONNX、部署端的关节顺序，不能依赖默认匹配顺序。

机器人资产：

- `quad_leg_velocity` 使用 `BENNETT_CFG_V3`。
- `BENNETT_CFG_V3` 的 `joint_pos={}` 为空，默认姿态很可能来自 USD/URDF 导入默认零位。
- `quad_leg_standing_2` 使用 `BENNETT_CFG_V1`，有明确默认站姿：左侧 thigh `+0.14`，右侧 thigh `-0.14`，四个 calf `-0.28`。

建议：crawl 第一版不应直接沿用 `BENNETT_CFG_V3` 空默认姿态；应先明确采用已验证站立默认姿态，或者把 crawl 专用 default pose 写成独立配置。

## URDF Body / Joint Names

URDF 中 link 名称包括：

```text
base
FL_thigh, FL_calf, FL_3, FL_1, FL_2
FR_thigh, FR_calf, FR_3, FR_1, FR_2
RL_thigh, RL_calf, RL_3, RL_1, RL_2
RR_thigh, RR_calf, RR_3, RR_1, RR_2
```

当前 velocity 任务把 foot body 设置为 `.*_1`，这与站立任务里的 `FOOT_BODIES = [".*_1"]` 一致。

官方 velocity 原始配置里的 `.*FOOT`、`.*THIGH` 不适合 Bennett。当前已经把 `feet_air_time` 改成 `.*_1`，并且关闭了 `undesired_contacts`。

风险：

- 关闭 `undesired_contacts` 会让 thigh/calf/link 意外触地没有明确惩罚。
- Crawl 需要区分 stance foot、swing foot、base contact、非足端连杆接触；仅有 `base_contact` termination 不够。

## Observation

官方 rough velocity policy 默认观测顺序是：

```text
base_lin_vel
base_ang_vel
projected_gravity
velocity_commands
joint_pos
joint_vel
actions
height_scan
```

Bennett rough velocity 修改：

- `base_lin_vel = None`
- `joint_pos` 限定 8 个主动关节
- `joint_vel` 限定 8 个主动关节
- rough 保留 `height_scan`
- flat 关闭 `height_scan`

因此当前实际观测大致为：

Rough:

```text
base_ang_vel: 3
projected_gravity: 3
velocity_commands: 3
joint_pos_rel: 8
joint_vel_rel: 8
last_action: 8
height_scan: 约 160 点，取决于 ray grid 1.6 x 1.0 / 0.1
```

Flat:

```text
base_ang_vel: 3
projected_gravity: 3
velocity_commands: 3
joint_pos_rel: 8
joint_vel_rel: 8
last_action: 8
合计约 33 维
```

与 crawl 目标的差距：

- 没有 `global_phase sin/cos`。
- 没有四腿 `leg_phase` 或 `desired_contact`。
- 没有 `gait_frequency`、`duty_factor`、`swing_height`。
- 没有 observation history。
- 没有 critic 特权观测分组。
- rough 的 height scan 真机不可直接获得，第一版 crawl 平地任务应先关闭。
- joint_pos/joint_vel 没有 `preserve_order=True`，部署一致性不足。
- 当前没有导出的 observation 名称、顺序、scale、clip metadata。

## Command

当前 command 继承官方 `UniformVelocityCommandCfg`：

```text
lin_vel_x = [-1.0, 1.0]
lin_vel_y = [-1.0, 1.0]
ang_vel_z = [-1.0, 1.0]
heading = [-pi, pi]
heading_command = True
rel_standing_envs = 0.02
rel_heading_envs = 1.0
resampling_time = 10 s
```

Isaac Lab 的 `UniformVelocityCommand` 会对 `rel_standing_envs` 采样到的环境把 command 全部置零。

与 crawl 目标的差距：

- 速度范围过大，不是超慢 crawl 课程。
- 一开始就包含 `vy`、`yaw`、heading command，不适合 Stage 1 固定前进 crawl。
- stand 是概率 zero-command，不是明确 crawl/stand 模式。
- 没有 gait frequency、duty factor、phase offset、swing height command。
- 没有每周期位移目标。

## Reward

当前继承的主要 reward：

```text
track_lin_vel_xy_exp
track_ang_vel_z_exp
lin_vel_z_l2
ang_vel_xy_l2
dof_torques_l2
dof_acc_l2
action_rate_l2
feet_air_time
flat_orientation_l2
dof_pos_limits
```

Bennett rough 覆盖：

- `feet_air_time.body_names = ".*_1"`
- `feet_air_time.weight = 0.01`
- `undesired_contacts = None`
- `dof_torques_l2.weight = -0.0002`
- `track_lin_vel_xy_exp.weight = 1.5`
- `track_ang_vel_z_exp.weight = 0.75`

Bennett flat 进一步覆盖：

- `flat_orientation_l2.weight = -2.5`
- `feet_air_time.weight = 0.25`

重要风险：

- 官方 `feet_air_time` 内部有硬门控：`norm(command[:2]) > 0.1`。如果 crawl 目标包含 `0.02 ~ 0.10 m/s`，这一项会完全没有奖励信号。
- `feet_air_time` 奖励长腾空步，不等价于“指定 swing window 内抬指定腿”，和三足支撑 crawl 有冲突。
- flat 里 `feet_air_time.weight=0.25` 对 crawl 偏大，可能鼓励动态步态或多腿离地。
- 当前没有 `desired_contact` 匹配奖励。
- 当前没有 `less_than_three_contacts`、`multiple_swing_legs` 惩罚。
- 当前没有 stance foot slip、touchdown impact、swing clearance、cycle displacement。
- 当前没有 support polygon / stability margin metric。
- 当前速度 tracking 的 `std=sqrt(0.25)=0.5` 对超慢速度过宽。目标 `0.03 m/s` 时，站着不动也可能拿到较高线速度奖励。

## Termination

当前继承：

- `time_out`
- `base_contact`

Bennett 把 `base_contact.body_names` 设置为 `base`。

与 crawl 目标的差距：

- 没有低 base height termination。
- 没有 excessive tilt termination。
- 没有 joint limit termination。
- 没有少于三足支撑的 violation metric / termination。
- 没有多腿同时摆动 violation metric / termination。

建议第一版训练中不要把所有接触违规都直接 termination；先作为 metric 和 reward 记录，确认接触检测稳定后再决定是否终止。

## Event / Domain Randomization

当前 rough：

- physics material startup 固定摩擦区间。
- base mass 随机加 `[-1.0, 3.0] kg`。
- base COM randomization 被关闭。
- push_robot 被关闭。
- reset base pose/velocity 全部固定 0。
- reset joints position_range 固定 `(1.0, 1.0)`。
- rough terrain 高度被缩小。

与 crawl 目标的关系：

- 对第一阶段 crawl 来说，关闭 push、固定 reset 是合理的。
- base mass `[-1, 3] kg` 对小型 Bennett 是否过宽，需要按整机实测质量决定。
- 关闭 COM randomization 可以先保留，但最终 crawl 必须测量真实 COM 并做窄范围随机化。
- rough terrain 不应作为第一版 crawl 起点。应先 flat、固定 vx、固定 gait。

## PPO / Training Config

当前 PPO：

- rough: `num_steps_per_env=24`，`max_iterations=1500`，网络 `[512, 256, 128]`，`learning_rate=1e-3`，`schedule=adaptive`。
- flat: `max_iterations=300`，网络 `[128, 128, 128]`。

风险：

- 当前 PPO 是普通 velocity 任务配置，没有针对 crawl 的长期周期完成、contact F1、低速命令课程设计。
- 当前没有固定 crawl 评估脚本，也没有 1000 episode 通过闸门。

## Export / Deployment Gap

当前脚本行为：

- `scripts/rsl_rl/train.py` 会保存 `params/env.yaml` 和 `params/agent.yaml`。
- `scripts/rsl_rl/train.py` 支持 `--export_io_descriptors`，但默认关闭。
- `scripts/rsl_rl/play.py` 会导出 `exported/policy.pt` 和 `exported/policy.onnx`。
- 当前仓库没有看到 `policy_metadata.yaml`、`joint_mapping.yaml`、`observation_builder.py` 或训练/部署共同校验工具。
- 当前仓库没有 `Bennett_quad_velocity_*` 训练日志目录。

与 crawl / Sim2Real 的差距：

- 没有单一 metadata 记录 observation order、joint order、action scale、default pose、control dt。
- 没有 ONNX/JIT 与训练端 observation parity 测试。
- 没有部署端拒绝启动机制来检查 motor id、sign、zero offset、policy input/output dim。

## 可复用资产

可以复用：

- `quad_leg_standing_2` 的 Sim2Real-oriented observation 结构：`base_ang_vel + projected_gravity + joint_pos + joint_vel + last_action`。
- `quad_leg_standing_2` 的 `preserve_order=True` 写法。
- `quad_leg_standing_2` 的 foot body 定义 `FOOT_BODIES = [".*_1"]`。
- `quad_leg_track` 的相位观测、参考 offset、分腿 reward logging、2D action 展开思路。
- `quad_leg_track` 的 action rate limit 思路：`max_joint_speed * env.step_dt`。

不建议直接复用：

- `quad_leg_track` 的 trot-like `LEG_SIGNS` 作为 crawl 步态逻辑。
- 当前 `quad_leg_velocity` 的 full velocity command 范围。
- 当前 `flat_env_cfg.py` 的 `feet_air_time.weight=0.25`。

## Gap Matrix

| 项目 | 当前状态 | Crawl 需要 | 优先级 |
|---|---|---|---|
| 任务隔离 | velocity task | 独立 crawl task id/package | 高 |
| gait scheduler | 无 | global phase、leg phase、desired_contact、desired_swing | 高 |
| 三足支撑约束 | 无 | min stance >= 3, max swing <= 1 | 高 |
| action | 8D joint residual, scale 0.25 | 可保留，但需 preserve_order、可考虑限速 | 高 |
| default pose | V3 空 joint_pos | 明确 crawl/stand 默认姿态 | 高 |
| actor obs | 无 phase/contact | 加 phase/contact/gait params | 高 |
| command | -1~1 m/s + yaw/heading | 固定低速 vx、固定 gait，逐步课程 | 高 |
| feet_air_time | 使用普通项 | 关闭或替换为 phase-gated swing reward | 高 |
| slow deadband | feet_air_time 有 0.1 门控 | crawl reward 不应被 0.02~0.1 清零 | 高 |
| contact reward | 无 | contact match/F1/early/late touchdown | 高 |
| support margin | 无 | 先 metric，后 reward | 中 |
| deployment metadata | 无 | policy_metadata.yaml + 校验工具 | 高 |
| sim2sim | 无 | MuJoCo 通过闸门 | 中 |

## 建议执行顺序

### Step 1：新增 crawl skeleton，不接训练

新建独立任务包，例如：

```text
source/bennett_rl/bennett_rl/tasks/manager_based/quad_leg_crawl/
```

先只放：

- `__init__.py`
- `crawl_env_cfg.py`
- `mdp/gait_scheduler.py`
- `mdp/__init__.py`
- `agents/rsl_rl_ppo_cfg.py`

不要改官方 Isaac Lab 源码，不要改现有 `quad_leg_velocity`。

### Step 2：先实现 gait scheduler 单元测试

要求：

- 腿序固定 `[FL, FR, RL, RR]`。
- 支持 `stand` 和 `crawl` 即可，其他 gait 以后再加。
- crawl 输出 `global_phase`、`leg_phase`、`desired_contact`、`desired_swing`。
- 对一整个周期断言：
  - `desired_contact.sum(dim=1) >= 3`
  - `desired_swing.sum(dim=1) <= 1`
- 生成一个 contact schedule 表或图。
- 暂时不接环境。

### Step 3：接入 flat crawl 环境

第一版只做：

```text
flat ground
vx fixed around 0.08~0.15 m/s
vy = 0
yaw = 0
fixed frequency
fixed duty factor around 0.85
height_scan = off
domain randomization = minimal
```

### Step 4：替换 reward

先关闭普通 `feet_air_time`，新增 crawl 专用：

- stance contact match
- swing no-contact match
- less than three contacts penalty
- multiple swing legs penalty
- stance slip
- base height / upright
- action rate / joint vel / torque
- cycle displacement metric

### Step 5：部署一致性

在训练能跑通前就设计 metadata，不要等真机前补：

- joint order
- observation order
- action scale
- default pose
- control dt
- policy input/output dim
- motor id/sign/zero placeholder

## 目前不建议做的事

- 不建议直接把 `quad_leg_velocity` 的 `lin_vel_x` 改成 `0.02~0.15` 后开始训练。
- 不建议在现有 velocity reward 上叠一堆 crawl reward。
- 不建议第一版加入 rough terrain、yaw、vy、后退、起停。
- 不建议一开始做大范围 domain randomization。
- 不建议当前阶段改真机部署脚本。

## 下一步建议

建议下一步执行：

```text
实现 quad_leg_crawl/mdp/gait_scheduler.py
+ scratch/单元测试验证一个周期 contact schedule
+ 暂不接入训练环境
```

通过后再接入 `crawl_env_cfg.py` 的 observation。
