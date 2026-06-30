# 四足机器人“超慢速、静态爬行步态、高成功率 Sim2Real”训练方案

> 面向项目：Bennett 连杆四足机器人  
> 推荐技术栈：Isaac Lab + RSL-RL + ROS 2 + Jetson/MC02  
> 整理日期：2026-06-30  
> 核心目标：**慢速优先、稳定优先、可解释接触时序优先、Sim2Real 成功率优先，而不是追求高速和花哨动作。**

---

## 0. 先给结论

你要训练的并不是普通的“把速度指令设得很小的四足速度跟踪策略”，而是：

> **具有明确足端接触时序约束的静态爬行步态（static walk / crawl gait / creep gait）策略。**

它的关键特征是：

- 任意时刻**不得少于 3 条腿处于可靠支撑状态**；
- 一次最多抬起一条腿；
- 抬腿前先进行机身/质心转移，使质心投影落在另外三足形成的支撑三角形内；
- 占空比 `duty factor β` 通常应不小于 `0.75`；
- 为提高真机鲁棒性，建议先用 `β = 0.80～0.90`，允许短暂四足支撑重叠，而不是严格卡在 `0.75`；
- 低速命令、起步、停车、转弯均必须单独训练，不能期待普通 trot 策略自动学会。

### 最推荐的总体方案

不要一开始采用完全自由的端到端强化学习。对你的目标，更合适的是：

```text
确定性步态调度器
    ↓ 给出
期望接触状态 + 步态相位 + 当前允许摆动的腿
    ↓
强化学习策略
    ↓ 输出
关节位置残差 / 足端轨迹残差 / 机身姿态补偿
    ↓
低层 PD / 电机控制
```

也就是：

> **“固定接触时序 + RL 学习平衡和残差补偿”**，而不是让 RL 自己猜该先抬哪条腿。

这种方案牺牲了一些动作自由度，但更容易获得：

- 三足支撑保证；
- 可解释的步态；
- 更小的训练搜索空间；
- 更高的重复性；
- 更低的 Sim2Real 风险；
- 出问题时容易判断究竟是步态调度、动力学、奖励还是部署代码的问题。

---

# 1. 你链接中的 walk 与机器人领域术语

你给出的页面：

- Quadruped Gaits — Animator Notebook  
  https://www.animatornotebook.com/learn/quadrupeds-gaits

该页面主要是四足动物动画和落足顺序参考。机器人领域对你描述的步态更常用以下名称：

| 名称 | 中文理解 | 特征 |
|---|---|---|
| Static walk | 静态行走 | 每次抬一条腿，强调静态稳定 |
| Crawl gait | 爬行步态 | 四拍步态，速度慢、支撑时间长 |
| Creep gait | 匍匐/缓行步态 | 与 crawl 高度接近，部分资料混用 |
| Walk gait | 行走步态 | 含义较宽，不一定严格保证始终三足支撑 |

后续代码、论文和项目检索时，建议优先使用：

```text
quadruped static walk
quadruped crawl gait
quadruped creep gait
statically stable quadruped locomotion
quadruped support polygon
gait-conditioned locomotion
contact-conditioned locomotion
```

## 1.1 “三条腿永远着地”的工程化定义

不建议把它写成“始终恰好三足着地”，更合理的约束是：

```text
任意时刻有效支撑足数量 >= 3
任意时刻摆动足数量 <= 1
```

原因：

- 换腿瞬间最好存在短暂的四足支撑；
- 真机接触检测会抖动，不可能每次都在同一毫秒完成切换；
- 四足重叠支撑能够给下一条腿的抬起留出安全时间；
- 严格“恰好三足”反而会迫使机器人过早抬腿。

建议第一版参数：

```yaml
gait_name: crawl
frequency_hz: 0.35 ~ 0.70
duty_factor: 0.82 ~ 0.90
swing_legs_max: 1
min_stance_legs: 3
four_leg_overlap: true
```

以上是适合初次训练的工程建议，不是必须固定不变的标准答案。

---

# 2. 开源项目筛选与推荐等级

## 2.1 第一梯队：直接作为你的工程基础

### A. Isaac Lab

项目：

- https://github.com/isaac-sim/IsaacLab
- 官方文档：https://isaac-sim.github.io/IsaacLab/

适合你的原因：

- 你已经在使用；
- Manager-Based 环境适合把步态调度、观测、奖励、事件、课程分别模块化；
- 官方已有四足速度跟踪任务；
- 支持 RSL-RL；
- 有接触传感器、域随机化、执行器模型、并行训练等基础设施。

建议：

- **不要换训练框架**；
- 在现有 Bennett 速度任务上增加 crawl gait command、接触目标、支撑稳定性奖励；
- 继续保持项目为 Isaac Lab 外部 extension，避免直接改 Isaac Lab 源码。

### B. RSL-RL

项目：

- https://github.com/leggedrobotics/rsl_rl

用途：

- PPO 训练；
- 非对称 actor-critic；
- observation history、teacher/student、RND、distillation 等能力可按版本使用。

对当前任务的建议：

- 第一阶段使用普通 PPO；
- critic 可以看到域随机化参数和真值状态；
- actor 只使用真机能够获得的本体感知观测；
- 不要一开始叠加太多复杂算法，否则难以定位问题。

### C. basic-locomotion-isaaclab

项目：

- https://github.com/iit-DLSLab/basic-locomotion-isaaclab

这是当前非常值得参考的工程项目，主要价值不是直接复制奖励，而是参考完整链路：

- Isaac Lab 四足 locomotion；
- sim2sim；
- sim2real；
- MuJoCo；
- ROS 2；
- 状态估计；
- RMA；
- 对称性；
- PACE 参数辨识。

建议重点研究目录：

```text
任务环境定义
机器人资产配置
策略导出
MuJoCo 验证
ROS 2 真机部署
观测量排序
状态估计
参数辨识接口
```

### D. robot_lab + rl_sar

项目：

- https://github.com/fan-ziqi/robot_lab
- https://github.com/fan-ziqi/rl_sar

用途划分：

```text
robot_lab：Isaac Lab 内训练和任务模板
rl_sar：仿真验证、ROS/ROS2、libtorch/ONNX、真机部署
```

值得借鉴：

- 自定义机器人如何挂入 Isaac Lab；
- 如何组织不同机器人任务；
- 推理中间层；
- ROS 2 和 C++ 部署；
- 策略观测/action 对接；
- MuJoCo/Gazebo 验证。

注意：

- 不能只复制配置文件；
- 关节名称、关节顺序、正方向、默认角度、action 顺序必须和 Bennett 完全一致；
- 策略导出后要生成“观测顺序清单”和“关节顺序清单”，由训练端和部署端共同读取。

---

## 2.2 第二梯队：重点借鉴其步态设计

### E. Walk These Ways

项目：

- https://github.com/Improbable-AI/walk-these-ways

论文：

- Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior  
  https://arxiv.org/abs/2212.03238

这是本任务最值得研究的“步态接口”项目之一。它具有：

- gait phase；
- gait frequency；
- foot phase offsets；
- gait duration / duty factor 类参数；
- foot swing height；
- 期望接触状态；
- 多步态条件输入；
- 策略历史观测；
- 域随机化和延迟模拟；
- Go1 真机部署。

**建议移植的是思想和步态调度代码，不是整仓库替换。**

重点借鉴：

```text
_step_contact_targets()
gait_indices
phase / offsets / bounds
desired_contact_states
clock inputs
gait-conditioned commands
```

需要警惕：

- 项目建立在较旧的 Isaac Gym 技术栈上；
- 代码中的默认速度范围、步频、抬脚高度针对 Go1 和动态步态；
- 某些配置会把很小的平面速度命令直接置零；
- 这对你的超慢速任务可能是致命问题；
- 必须检查所有 command deadband、standing mask 和 curriculum threshold。

### F. IsaacLab Tutorial

项目：

- https://github.com/Lab-of-AI-and-Robotics/IsaacLab-Tutorial

适合用来补齐：

- 从 Go2 基础任务开始；
- 奖励设计；
- 速度课程；
- 稳定性；
- ActuatorNet；
- Sim2Real 思路。

它比很多零散视频更适合作为 Codex 阅读材料，因为代码按章节组织。

---

## 2.3 第三梯队：提高 Sim2Real 成功率

### G. PACE Sim2Real

项目：

- https://github.com/leggedrobotics/pace-sim2real
- 项目主页：https://pace.filipbjelonic.com/

作用：

- 使用真机关节编码器数据辨识执行器和关节动力学；
- 优化仿真参数；
- 减少靠拍脑袋设置摩擦、阻尼、转子惯量等参数；
- 可接入 Isaac Lab 工作流。

对于你的 DM8006 + Bennett 自研机器人，PACE 的价值很高。你的机器人不是成熟商业整机，以下误差更可能明显：

- 电机实际输出与目标力矩不一致；
- 驱动器内部滤波未知；
- 电机控制延迟；
- 减速器和连杆摩擦；
- 关节装配阻力；
- 左右腿机械差异；
- 编码器零偏；
- 结构柔性和间隙。

建议优先顺序：

```text
先测量和辨识
    ↓
建立可信名义模型
    ↓
围绕测量不确定性做适度域随机化
```

不要采用：

```text
模型不准
    ↓
把所有参数随机得极宽
    ↓
希望策略自动解决
```

超宽域随机化可能得到“什么都能勉强走、但哪里都走不好”的保守策略。

### H. SPI-Active

项目：

- https://github.com/LeCAR-Lab/SPI-Active

用途：

- 主动系统辨识；
- 识别基座质量、质心、惯量；
- 模块化电机动力学模型。

当前代码主要面向 Unitree Go2，因此更适合作为研究思路和二期工具，不建议第一阶段强行接入 Bennett。

### I. 经典 Sim2Real 参考

论文：

- Sim-to-Real: Learning Agile Locomotion for Quadruped Robots  
  https://arxiv.org/abs/1804.10332

其核心思路至今仍然重要：

- system identification；
- accurate actuator model；
- latency simulation；
- domain randomization；
- disturbances；
- compact observations；
- 必要时使用 open-loop reference 引导步态。

---

## 2.4 第四梯队：独立仿真验证

### J. Unitree RL Lab / Unitree MuJoCo

项目：

- https://github.com/unitreerobotics/unitree_rl_lab
- https://github.com/unitreerobotics/unitree_mujoco

即使你不是 Unitree 机器人，也值得借鉴其流程：

```text
Isaac Lab 训练
    ↓
MuJoCo sim2sim
    ↓
真机 sim2real
```

对 Bennett 需要自己建立 MuJoCo 模型，但这一步非常有价值：

- 可发现只对 PhysX 有效的策略；
- 检查碰撞、摩擦和执行器依赖；
- 检查推理代码与训练代码是否一致；
- 验证策略是否依赖仿真器漏洞。

### K. MuJoCo Playground

项目：

- https://github.com/google-deepmind/mujoco_playground

可作为现代 MuJoCo/MJX 训练和验证参考，但你目前没有必要从 Isaac Lab 迁移过去。

结论：

> MuJoCo 对你更适合充当**第二仿真器和部署前闸门**，而不是替代现有 Isaac Lab 主训练环境。

---

## 2.5 传统控制/轨迹生成参考

### L. CHAMP

项目：

- https://github.com/chvmp/champ

用途：

- 传统四足 gait、足端轨迹、IK 和 ROS；
- 可作为不依赖 RL 的基准控制器；
- 帮助验证 URDF、关节方向、零位和基础落足顺序。

### M. TOWR

项目：

- https://github.com/leggedrobotics/towr

包含 contact sequence / gait 相关设计，可用于：

- 生成参考落足序列；
- 研究 walk、walk overlap、trot、bound、gallop 等接触安排；
- 生成离线参考轨迹。

### N. Crocoddyl

项目：

- https://github.com/loco-3d/crocoddyl

适合研究：

- 接触序列下的最优控制；
- support foot / swing foot；
- 生成机身和足端参考；
- 后续做 MPC/RL 混合控制。

对第一阶段而言，CHAMP/TOWR/Crocoddyl 都是参考或基线，不应同时塞入训练主线。

---

# 3. 对你的项目最推荐的路线

## 3.1 主路线

```text
现有 Bennett Isaac Lab Manager-Based 速度任务
        +
自定义 CrawlGaitCommand / ContactSchedule
        +
RSL-RL PPO
        +
明确的接触时序奖励与约束
        +
低速专用 command curriculum
        +
执行器/延迟/摩擦建模
        +
MuJoCo sim2sim
        +
ROS 2/C++ 真机部署
```

## 3.2 为什么不建议直接训练“通用速度策略”

普通 velocity locomotion 环境通常主要奖励：

- 线速度跟踪；
- 角速度跟踪；
- 姿态；
- 能耗；
- 平滑；
- 足端腾空时间。

但它并不保证：

- 抬腿顺序；
- 一次只抬一条腿；
- 三足支撑；
- 质心转移；
- 低速时仍然完成周期性步态。

当速度很小时，策略可能通过以下方式骗取奖励：

- 原地站立；
- 四脚拖动；
- 高频小碎步；
- 两条腿同时轻微离地；
- 依赖仿真静摩擦；
- 以很小幅度抖动获得平均速度；
- 不完成完整 gait cycle。

所以必须加入**接触条件**。

---

# 4. 静态爬行步态调度器

## 4.1 基础状态

建议定义：

```python
phase:               [num_envs]      # 总步态相位，范围 [0, 1)
leg_phase:           [num_envs, 4]   # 各腿局部相位
desired_contact:     [num_envs, 4]   # 期望接触，0/1
desired_swing:       [num_envs, 4]
gait_frequency:      [num_envs]
duty_factor:         [num_envs]
phase_offsets:       [num_envs, 4]
swing_height:        [num_envs]
gait_id:             [num_envs]
```

腿顺序必须固定。例如：

```text
[FL, FR, RL, RR]
```

训练端、导出端、真机端均不得改变。

## 4.2 一个简单的相位计算

```python
phase = (phase + control_dt * gait_frequency) % 1.0
leg_phase = (phase[:, None] + phase_offsets) % 1.0

# 假设局部相位 [0, duty_factor) 为支撑期
desired_contact = leg_phase < duty_factor
desired_swing = ~desired_contact
```

为了保证一次最多一条腿摆动：

```python
assert desired_swing.sum(dim=1).max() <= 1
assert desired_contact.sum(dim=1).min() >= 3
```

注意：

- 断言适合开发和测试；
- 大规模训练中可改为统计指标或违规终止；
- 接触相位边界最好加入平滑过渡，不能直接让策略面对完全不连续的目标。

## 4.3 Crawl 相位示例

腿顺序：

```text
[FL, FR, RL, RR]
```

一种示例摆动顺序：

```text
FL → RR → FR → RL
```

对应摆动开始相位示例：

```python
phase_offsets = [0.00, 0.50, 0.75, 0.25]
```

但这只是初始候选，不是所有机器人通用的唯一顺序。最终必须根据：

- Bennett 机身质心；
- 前后腿安装位置；
- 足端可达区域；
- 左右腿机构镜像；
- 电池、Jetson、驱动板布局；
- 真机支撑裕度；

验证每一次抬腿前，质心投影是否处于三足支撑三角形内部。

## 4.4 更稳妥的 8 阶段状态机

纯相位调度器能工作，但对于超慢静态步态，推荐显式 8 阶段：

```text
1. 向支撑三角形内部移动机身/质心
2. 抬起 FL
3. FL 落地并确认接触
4. 再次移动机身/质心
5. 抬起 RR
6. RR 落地并确认接触
7. 后续 FR、RL 同理
8. 完成整周期
```

实际可细分为：

```text
SHIFT_COM
LIFT_LEG
SWING_LEG
TOUCHDOWN
CONFIRM_CONTACT
```

真机安全逻辑：

```python
if touchdown_expected and contact_confirmed:
    advance_phase()
elif timeout:
    lower_foot_safely()
    stop_or_recover()
```

RL 训练中仍可用固定时间相位，部署时再加入接触确认；但必须把由此产生的相位延迟也纳入仿真随机化。

---

# 5. 推荐的观测量

## 5.1 Actor 可使用的观测

```text
1. base angular velocity
2. projected gravity
3. commanded vx, vy, yaw rate
4. joint position error relative to default pose
5. joint velocity
6. previous action
7. gait phase sin/cos
8. four leg phase sin/cos，或四足 desired_contact
9. gait frequency
10. duty factor
11. commanded swing height
12. gait id / one-hot gait code
13. 适量 observation history
```

建议不要直接向 actor 提供真机无法稳定获得的真值：

- 世界坐标绝对位置；
- 无噪声 base linear velocity；
- 真值地面摩擦；
- 真值质心位置；
- 仿真器专有接触信息，除非真机有对应可靠传感器。

### 关于线速度

真机若没有可靠的 base linear velocity，可选：

1. 使用状态估计器；
2. 使用 observation history 让策略隐式推断；
3. teacher 使用真值速度，student 不使用；
4. 先在平地低速任务中验证不含线速度的策略。

## 5.2 Critic 特权观测

critic 可加入：

```text
base linear velocity truth
foot contact truth
friction coefficient
payload/base mass offset
CoM offset
motor strength
Kp/Kd scale
latency
external force
terrain height
joint friction
encoder bias
```

这属于 asymmetric actor-critic：

- critic 在训练中获得更完整信息；
- actor 保持为真机可部署观测。

## 5.3 历史观测

建议第一版：

```yaml
history_length: 5 ~ 15
```

历史可帮助策略处理：

- 延迟；
- 速度估计；
- 接触状态；
- 电机响应差异；
- IMU 噪声。

但必须注意：

- 训练与真机 history buffer 初始化方式一致；
- reset 后填零、复制首帧或重复默认帧必须一致；
- 堆叠顺序必须一致；
- 每帧归一化必须一致。

---

# 6. 动作空间

## 6.1 推荐第一版：关节位置残差

```python
q_target = q_default + action_scale * policy_action
```

优点：

- 和常见电机 PD 控制兼容；
- 易限制；
- 真机安全性比直接力矩高；
- 更容易实现仿真/真机一致。

建议：

```text
action_scale 不宜过大
不同关节可设置不同 scale
髋外展、髋俯仰、膝关节分开限制
加入 action rate limit
加入 joint soft limit
加入 torque/current saturation
```

## 6.2 更适合“固定 gait + RL 残差”的动作

可以让策略输出：

```text
机身 roll/pitch/height residual
四足 x/y/z 足端轨迹 residual
或关节位置 residual
```

参考轨迹由 gait scheduler + IK 生成。

这样可以把：

- 抬哪条腿；
- 何时抬腿；
- 基本摆动轨迹；

交给确定性模块，RL 负责：

- 平衡；
- 接触适应；
- 模型误差；
- 姿态修正；
- 轻微地形扰动。

## 6.3 暂不推荐：直接力矩策略

不是不能做，而是你的当前目标是“迁移成功率极高”。直接力矩对以下误差更敏感：

- 电机扭矩常数；
- 驱动器电流环；
- 摩擦；
- 总线延迟；
- 力矩饱和；
- 关节柔性；
- 结构装配差异。

第一版先使用位置目标 + 可辨识低层 PD 更稳妥。

---

# 7. 奖励函数设计

不要只看总 reward。必须监控每个原始物理量。

## 7.1 速度跟踪

```text
track_lin_vel_xy
track_ang_vel_z
```

超慢速时注意：

- 速度误差的核宽度不能沿用高速任务；
- 例如目标只有 0.03 m/s，而奖励容忍范围是 0.20 m/s，站立也可能拿到很高奖励；
- 应根据慢速范围缩小 `std` 或误差尺度；
- 同时加入“每个 gait cycle 的净位移”指标，防止原地抖动。

## 7.2 接触时序奖励

必须新增：

```text
stance_contact_match
swing_no_contact_match
gait_contact_f1
early_liftoff_penalty
late_touchdown_penalty
multiple_swing_legs_penalty
less_than_three_contacts_penalty
```

示意：

```python
measured_contact = contact_force_z > threshold

stance_match = desired_contact & measured_contact
swing_match = (~desired_contact) & (~measured_contact)

contact_reward = (
    w_stance * stance_match.float().mean(dim=1)
    + w_swing * swing_match.float().mean(dim=1)
)
```

接触不能只用一个硬阈值，真机和仿真都建议加入：

- 滞回；
- 最短确认时间；
- 低通；
- touchdown/liftoff 分开阈值。

## 7.3 静态稳定裕度

静态步态真正重要的是：

> 质心在地面的投影，与当前支撑多边形边界之间的最小距离。

定义：

```text
stability_margin > 0：质心投影在支撑多边形内
stability_margin = 0：位于边界
stability_margin < 0：已在外部
```

奖励可以是：

```python
reward = clamp(stability_margin / desired_margin, 0, 1)
penalty = relu(-stability_margin)
```

第一版若实现凸包过于复杂，可先用替代项：

- 机身投影靠近期望支撑中心；
- 抬腿前向三足支撑三角形中心移动；
- roll/pitch 小；
- stance foot 不滑；
- 支撑腿法向力不过度偏置。

但最终仍建议实现真实 support polygon margin，用于评估和部署安全。

## 7.4 支撑足滑移

超慢步态的核心指标之一：

```python
slip_velocity = foot_velocity_xy * measured_contact
```

可使用：

```text
stance_foot_slip_l2
slip_distance_per_meter
max_stance_foot_speed
```

不要只给很大的惩罚而不看物理原因。滑移可能来自：

- 摩擦系数不匹配；
- foot collision 形状；
- 质心未转移；
- 足端目标变化过快；
- PD 太硬；
- action 抖动；
- 接触求解参数；
- 机体高度不合适。

## 7.5 摆动足奖励

仅在期望 swing 时启用：

```text
foot_clearance
foot_forward_progress
swing_trajectory_tracking
touchdown_position_error
```

重要：

> 不要直接照搬很大的 `feet_air_time` 奖励。

普通 `feet_air_time` 经常鼓励更长腾空、动态步态甚至跳跃，与“至少三足支撑”的目标可能冲突。

应改成：

```text
期望摆动窗口内抬脚
达到适度高度
按时落地
非摆动腿不得腾空
```

## 7.6 着地冲击

```text
touchdown vertical velocity
peak contact force
contact impulse
```

超慢步态应尽量轻柔落足：

- swing 脚落地前降低 z 速度；
- 接触后快速建立稳定法向力；
- 避免膝关节瞬时冲击。

## 7.7 机身稳定

```text
flat_orientation
base_roll_pitch
lin_vel_z
ang_vel_xy
base_height
```

静态爬行允许有计划的轻微机身平移和侧摆，所以：

- 不要把所有 roll/pitch/xy 位移都惩罚到接近零；
- 否则会阻止必要的质心转移；
- 更合理的是跟踪 gait scheduler 给出的期望机身轨迹。

## 7.8 平滑、能耗与安全

```text
action_rate_l2
action_acceleration
joint_acc_l2
joint_torque_l2
joint_power
joint_pos_limits
joint_vel_limits
torque_limits
undesired_contacts
```

调权重顺序：

1. 先保证不摔；
2. 再保证接触时序；
3. 再保证能前进；
4. 再降低滑移和冲击；
5. 最后优化能耗与动作美观。

不能一开始把能耗惩罚设得过大，否则最省能的策略就是不走。

---

# 8. Command 设计：超慢速任务最容易踩坑的地方

## 8.1 不要让 deadband 抹掉慢速命令

必须全局搜索：

```text
command_norm < threshold
small command
standing_env
stand_still
clip command
zero command
0.1
0.2
```

某些公开 gait-conditioned 项目会把很小的平面速度命令直接设为 0，以便训练站立；如果你目标范围本身就是 `0.02～0.15 m/s`，这种逻辑会把训练命令全部消掉。

建议分开：

```yaml
stand_command_probability: 0.10
stand_deadband_mps: 0.005
crawl_vx_range_initial: [0.08, 0.14]
crawl_vx_range_final: [0.02, 0.18]
```

## 8.2 为什么不应一开始从 0.01 m/s 训练

过小速度下：

- 一个周期净位移很小；
- 速度测量噪声与目标同量级；
- 策略更容易选择不走；
- 摩擦和接触误差占主导；
- 训练信号弱。

更好的课程：

```text
先在 0.08～0.15 m/s 学会正确 crawl
再逐步降低到 0.02～0.08 m/s
最后加入 0、起步、停车和微速切换
```

## 8.3 使用“每周期位移”作为附加目标

例如：

```python
cycle_displacement = base_x_at_cycle_end - base_x_at_cycle_start
target_displacement = vx_cmd / gait_frequency
```

优点：

- 不容易被瞬时抖动欺骗；
- 更符合超慢四拍步态；
- 可以判断机器人是否完成了真正的整周期前进。

---

# 9. 建议的训练课程

## Stage 0：站立策略

目标：

- 默认姿态稳定；
- 轻微外力不摔；
- 零速度无抖动；
- 足端无滑移；
- 开机进入姿态平滑。

训练内容：

```text
固定平地
较窄参数随机化
随机轻推
质量/质心小范围随机
关节零偏和 IMU 噪声
```

通过后冻结 checkpoint：

```text
bennett_stand_v1.pt
```

## Stage 1：固定速度 Crawl

目标：

```text
仅 vx > 0
固定步频
固定 duty factor
固定落足顺序
固定平地
一次一条腿
```

建议起始：

```yaml
vx: 0.08 ~ 0.15 m/s
vy: 0
yaw: 0
frequency: 0.45 ~ 0.65 Hz
duty_factor: 0.85
swing_height: 按腿长的约 3%～6% 起步
```

绝对抬脚高度需根据 Bennett 尺寸确定，不应机械照搬 Go1 的数值。

## Stage 2：降低速度

逐步扩展：

```text
0.15 → 0.10 → 0.06 → 0.03 → 0.02 m/s
```

加入：

- 不同 gait frequency；
- 不同 duty factor；
- 起步；
- 停车；
- phase reset；
- 切换命令时仍保持落足顺序。

## Stage 3：转弯、侧向和后退

顺序建议：

1. 小角速度原地/小半径转向；
2. 前进 + yaw；
3. 小范围 vy；
4. 后退；
5. 组合命令。

不要在 Crawl 尚未稳定时就同时加入全部三自由度命令。

## Stage 4：鲁棒性

逐步加入：

```text
质量和质心偏移
摩擦变化
电机强度
PD 增益偏差
延迟
观测噪声
编码器零偏
地面小坡度
低矮不平整
随机推力
负载变化
```

所有随机化都应有“测量依据 + 分阶段扩大”。

## Stage 5：Sim2Sim

至少在另一物理引擎验证：

```text
Isaac/PhysX → MuJoCo
```

通过后才能上真机。

## Stage 6：真机分级测试

```text
悬空/吊架检查关节方向
→ 脚接触但机身受保护
→ 软垫/低增益
→ 平整高摩擦地面
→ 正常地面
→ 小扰动
→ 不同摩擦
```

每一级失败都应回到模型或训练修正，而不是在真机上反复赌。

---

# 10. 其他步态的训练顺序

你希望“训完一个步态再训练下一个”，这是合理的，尤其适合自研机器人。

推荐：

```text
1. Stand
2. Static crawl / creep
3. Amble / overlapping walk
4. Slow trot
5. Normal trot
6. Pace
7. Bound
8. Pronk 或其他动态步态
```

优先级说明：

- Crawl 最慢、支撑最多，适合建立第一套迁移链路；
- Trot 是大量开源项目默认步态，后续参考资料最多；
- Pace 对侧向稳定性和结构误差更敏感；
- Bound/Pronk 属于动态步态，不符合当前“高安全慢速”主目标，可最后做。

## 10.1 相位参数示例

腿序：

```text
[FL, FR, RL, RR]
```

常见相位示意：

| 步态 | phase offset 示例 | 说明 |
|---|---:|---|
| Crawl | `[0.00, 0.50, 0.75, 0.25]` | 示例四拍顺序，需结合质心验证 |
| Trot | `[0.00, 0.50, 0.50, 0.00]` | 对角腿同步 |
| Pace | `[0.00, 0.50, 0.00, 0.50]` | 同侧腿同步 |
| Bound | `[0.00, 0.00, 0.50, 0.50]` | 两前腿/两后腿成对 |
| Pronk | `[0.00, 0.00, 0.00, 0.00]` | 四腿同步 |

具体相位正负和腿序取决于代码约定，复制前必须画出一整周期接触图。

## 10.2 每个步态先单独策略

为了迁移成功率，建议：

```text
stand_policy
crawl_policy
amble_policy
trot_policy
pace_policy
...
```

每个策略通过：

```text
训练收敛
→ 评估
→ sim2sim
→ 真机
```

之后再考虑：

- gait-conditioned single policy；
- policy distillation；
- skill switching；
- 多步态统一策略。

不要一开始把所有 gait 混在同一策略里。多任务会引入：

- 梯度冲突；
- gait 间折中；
- 难以排查失败；
- 真机切换不稳定。

---

# 11. Sim2Real 必须建模的项目

## 11.1 执行器与控制

至少考虑：

```text
motor strength scale
torque saturation
current limit
Kp/Kd mismatch
control latency
action latency
observation latency
zero-order hold
packet jitter
low-pass filter
joint damping
Coulomb friction
viscous friction
rotor/armature inertia
dead zone
backlash/structure compliance（若明显）
battery voltage influence
```

关键原则：

> 真机端增加的滤波、插值、限速、斜坡和保护逻辑，仿真端必须有等价实现。

否则“为了让真机更稳”临时加一个滤波器，也可能因额外延迟让策略失稳。

## 11.2 机械参数

测量：

```text
每个连杆质量
整机质量
质心
惯量估计
足端材料
关节摩擦
结构偏差
左右腿差异
线缆牵引
电池/Jetson/控制板位置
```

尤其是 Bennett 连杆机构：

- 闭链或等效约束建模是否准确；
- 传动关系；
- 输出关节角与电机角的映射；
- 机构奇异位形；
- 等效惯量随构型变化；
- 关节限位与真实干涉；
- 左右镜像方向。

## 11.3 传感器

建模：

```text
IMU bias
IMU noise
IMU mounting rotation error
gyro scale/bias
encoder zero offset
encoder quantization
velocity estimate filtering
timestamp mismatch
CAN feedback delay/drop
```

IMU 坐标系必须从真实板卡安装方向转换到机器人 base 坐标系，不能只看画面中“似乎朝前”。

## 11.4 接触

低速策略对接触模型尤其敏感：

```text
static friction
dynamic friction
restitution
foot collision shape
contact offset/rest offset
solver iterations
contact force threshold
foot sole compliance
ground unevenness
```

建议足端碰撞体：

- 简洁；
- 左右一致；
- 与真实接触区域相似；
- 不要使用过度复杂网格；
- 检查视觉模型与 collision model 是否错位。

---

# 12. 域随机化建议

## 12.1 第一阶段窄范围

先围绕可信名义模型：

```yaml
mass_scale: ±3% ~ ±5%
com_offset: 小范围
friction: 围绕实测地面
motor_strength: ±5%
kp_kd_scale: ±5%
latency: 围绕实测均值
encoder_offset: 围绕标定误差
```

## 12.2 第二阶段扩大

策略学会正确 Crawl 后再扩大：

```yaml
mass_scale: ±10% 或按实际负载
friction: 覆盖不同地面
motor_strength: ±10% ~ ±20%
latency: 覆盖偶发抖动
pushes: 从小到大
slope: 从 0° 逐步增加
terrain_height: 从毫米级逐步增加
```

范围必须按实机测试修正，上述仅为启动建议。

## 12.3 不建议的做法

```text
一上来 friction 0.1～2.0
质量 ±50%
COM 任意漂移
延迟随机几十个控制周期
强推力频繁触发
```

这种策略可能很“抗随机”，但会牺牲：

- 低速精度；
- gait 时序；
- 平顺性；
- 能效；
- 可调试性。

---

# 13. 控制频率建议

你当前已使用过：

```python
sim.dt = 0.005
decimation = 4
```

对应策略控制周期：

```text
0.005 × 4 = 0.020 s
policy frequency = 50 Hz
```

50 Hz 可以作为第一版策略频率。

推荐架构：

```text
策略：50～100 Hz
低层关节控制/电机环：尽可能更高
传感器反馈：稳定、时间戳一致
```

不能只比较标称 Hz，还要测：

- 实际周期均值；
- 最大周期；
- jitter；
- CAN 发送到电机执行的延迟；
- 反馈帧年龄；
- Jetson 调度抖动。

部署端每次推理应记录：

```text
policy inference time
control loop period
observation age
action age
CAN send duration
CAN receive duration
missed deadline count
```

---

# 14. 评估指标与通过闸门

不能只看 TensorBoard 总奖励或“看起来能走”。

## 14.1 Crawl 核心指标

建议记录：

```text
minimum number of stance feet
percentage of time stance_feet >= 3
percentage of time swing_feet <= 1
contact schedule accuracy/F1
early liftoff rate
late touchdown rate
support margin minimum/mean
stance foot slip velocity
slip distance per meter
velocity MAE/RMSE
yaw tracking error
roll/pitch RMS and 95th percentile
base height variation
touchdown impact
joint torque margin
joint limit violation
fall rate
cycle completion rate
```

## 14.2 建议验收门槛

以下是工程目标，可按硬件能力调整：

```text
名义平地：1000 个评估 episode 无摔倒
接触时序：>99% 时间满足至少三足支撑
一次多腿摆动：接近 0
命令 0.02～0.15 m/s 均能完成完整周期
零命令下稳定站立，无持续小碎步
不同随机种子结果一致
MuJoCo 中可连续运行
推力、质量、摩擦扰动后可恢复
```

真机阶段不要宣称“迁移成功率”而不定义分母。建议定义：

```text
成功 = 完成规定时长/距离
且无人工扶持
且未触发保护
且未出现少于三足支撑
且姿态和滑移指标在阈值内
```

例如：

```text
30 次独立启动，每次完成 10 m 或 60 s
成功次数 / 30
```

---

# 15. 高频踩坑清单

## 15.1 把“低速度”误认为“静态 Crawl”

错误：

```text
把 vx 范围调成 0.05 m/s
```

但没有 gait schedule。

结果：

- 策略可能站着；
- 拖脚；
- 小碎步；
- 使用 trot 的微小版本；
- 不保证三足支撑。

## 15.2 `feet_air_time` 权重过大

结果：

- 鼓励腿腾空更久；
- 学出动态步态；
- 多腿同时离地；
- 与 crawl 目标冲突。

改法：

- 使用目标 swing window；
- 只奖励指定腿在指定时间抬起；
- 惩罚其他腿离地。

## 15.3 慢速命令被 standing mask 清零

必须查训练代码中所有命令阈值，尤其是：

```python
norm(command[:2]) < 0.1
norm(command[:2]) < 0.2
```

超慢任务应显式采样 `stand`，而不是用很大的 deadband 推断站立。

## 15.4 速度奖励尺度沿用高速任务

若目标为 0.03 m/s，但奖励容忍 0.2 m/s：

- 站立和正确跟踪差别很小；
- PPO 没有动力学会爬行。

## 15.5 接触传感器抖动

不要把单帧 `Fz > threshold` 当作绝对真相。应有：

- touchdown hysteresis；
- liftoff hysteresis；
- minimum contact duration；
- substep contact accumulation；
- 接触状态滤波。

## 15.6 质心不在几何中心却按对称步态训练

自研机器人电池、Jetson、驱动板会造成质心偏移。必须：

- 实测整机质心；
- 写入 USD/MuJoCo；
- gait scheduler 的 COM shift 以真实质心为准；
- 域随机化覆盖装配变化。

## 15.7 训练端与部署端关节顺序不一致

典型表现：

- 仿真正常；
- 真机一上电腿乱动；
- 左右腿动作相反；
- 某两个关节互换。

必须自动校验：

```text
joint_names
joint_ids
action indices
observation joint order
motor CAN ID
motor sign
zero offset
gear ratio
```

生成单一配置文件，让训练与部署共用。

## 15.8 URDF/USD 默认角和真机零位混淆

要明确四种量：

```text
电机编码器零位
机械安装零位
URDF joint=0
策略 default_joint_pos
```

推荐统一公式：

```python
q_policy = sign * (q_motor - q_motor_zero) + q_urdf_offset
q_motor_target = inverse_mapping(q_policy_target)
```

部署前逐关节悬空测试。

## 15.9 action scale 太大

会导致：

- 策略频繁饱和；
- 真机冲击；
- 仿真里依靠极硬 PD；
- 微速控制分辨率差。

检查 action histogram，长期贴近 ±1 说明可能存在问题。

## 15.10 域随机化过早、过宽

先确认：

- 名义模型能稳定训练；
- gait 接触正确；
- 奖励方向正确；
- 速度可跟踪。

再逐项随机化，每加一项都做消融实验。

## 15.11 只做 PhysX 评估

一个策略可能利用：

- 接触参数；
- 数值积分；
- 摩擦实现；
- 碰撞穿透；
- 求解器偏差。

必须进行 MuJoCo sim2sim。

## 15.12 归一化不一致

检查：

```text
obs scale
clip
mean/std
command scale
joint position relative/default
angular velocity unit
action scale
history order
```

最好把归一化参数随 ONNX 一同导出。

## 15.13 真机临时增加滤波但仿真没有

滤波会增加相位滞后。必须把相同滤波器放到仿真 loop 中重新训练/评估。

## 15.14 只看平均奖励

平均奖励可能掩盖：

- 少量环境频繁摔倒；
- 某条腿从不抬；
- 左右步态不对称；
- 极端参数全部失败。

必须画分项指标、分布和最差分位数。

---

# 16. 推荐实验消融

每次只改变一项：

```text
A0：普通 velocity baseline
A1：+ gait phase observation
A2：+ desired contact schedule
A3：+ contact matching rewards
A4：+ support margin
A5：+ slip and touchdown rewards
A6：+ observation history
A7：+ measured latency
A8：+ actuator model
A9：+ calibrated domain randomization
A10：+ sim2sim
```

每个版本保存：

```text
git commit
config snapshot
random seed
checkpoint
exported policy
evaluation CSV
视频
```

否则无法判断是哪一步提高或破坏了迁移。

---

# 17. 推荐代码结构

```text
source/bennett_rl/bennett_rl/
├─ assets/
│  └─ bennett.py
├─ tasks/manager_based/locomotion/
│  └─ crawl/
│     ├─ __init__.py
│     ├─ crawl_env_cfg.py
│     ├─ mdp/
│     │  ├─ commands.py
│     │  ├─ observations.py
│     │  ├─ rewards.py
│     │  ├─ events.py
│     │  ├─ curricula.py
│     │  ├─ terminations.py
│     │  ├─ gait_scheduler.py
│     │  └─ metrics.py
│     ├─ agents/
│     │  └─ rsl_rl_ppo_cfg.py
│     └─ config/
│        ├─ crawl_flat_cfg.py
│        ├─ crawl_rough_cfg.py
│        ├─ trot_flat_cfg.py
│        └─ pace_flat_cfg.py
├─ deployment/
│  ├─ policy_metadata.yaml
│  ├─ observation_builder.py
│  ├─ joint_mapping.yaml
│  ├─ export_onnx.py
│  └─ ros2/
└─ tests/
   ├─ test_joint_order.py
   ├─ test_gait_schedule.py
   ├─ test_observation_parity.py
   ├─ test_action_parity.py
   └─ test_policy_onnx_parity.py
```

---

# 18. Codex 可直接执行的任务清单

## Task 1：审计现有 Bennett 环境

```text
请遍历当前 Bennett Isaac Lab 项目，输出：
1. observation 的精确排列、维度、缩放和裁剪；
2. action 到关节目标的完整公式；
3. joint_names、joint_ids、默认角、正方向；
4. policy dt、physics dt、decimation；
5. 所有 reward 的原始物理量、权重和启用条件；
6. command 采样范围和所有小速度置零逻辑；
7. domain randomization；
8. contact sensor 配置；
9. policy export 和真机 inference 的差异。
只做审计，不修改代码。输出 BENNETT_LOCOMOTION_AUDIT.md。
```

## Task 2：实现 gait scheduler

```text
在不修改 Isaac Lab 源码的前提下，为 Bennett Manager-Based locomotion 新增 gait_scheduler.py。
要求：
- 固定腿序 [FL, FR, RL, RR]；
- 支持 stand、crawl、trot、pace、bound、pronk；
- 输出 global_phase、leg_phase、desired_contact、desired_swing；
- crawl 任意时刻 desired_contact 数量不少于 3；
- crawl 任意时刻 desired_swing 数量不超过 1；
- 支持 frequency、duty_factor、phase_offsets、swing_height；
- 支持批量 num_envs；
- 使用 torch，避免 Python for-loop；
- 添加单元测试并画一整个周期的 contact schedule 图；
- 暂时不要接入训练环境。
```

## Task 3：接入观测

```text
将 gait scheduler 接入 Bennett crawl 环境。
Actor 新增：
- global phase sin/cos；
- 四腿 leg phase sin/cos，或 desired_contact；
- gait frequency；
- duty factor；
- swing height；
- gait id。
更新 observation dimension、RSL-RL config、export metadata。
添加测试，确保训练和 ONNX 部署观测顺序一致。
```

## Task 4：接触奖励

```text
新增 crawl 专用奖励：
- desired stance contact；
- desired swing no-contact；
- early liftoff；
- late touchdown；
- multiple swing legs；
- less than three stance legs；
- stance foot slip；
- swing foot clearance；
- touchdown vertical velocity；
- whole-cycle displacement。
每个奖励返回 [num_envs] tensor，并为每项添加原始物理量日志。
不要一次给出最终权重，先提供初始建议和消融开关。
```

## Task 5：支持多边形稳定裕度

```text
基于足端世界坐标和当前有效支撑足，计算 base CoM 在地面的投影到支撑多边形边界的带符号最小距离。
要求：
- 支持 3 足和 4 足支撑；
- 批量 torch 实现；
- 输出 margin、inside 标志、最危险边；
- 可视化支撑多边形、CoM 投影；
- 添加规则几何单元测试；
- 先作为 metric，不立刻加入 reward。
```

## Task 6：慢速 command curriculum

```text
实现 crawl 慢速课程：
阶段 1：vx 0.08～0.15 m/s；
阶段 2：逐步扩展到 0.04～0.18 m/s；
阶段 3：扩展到 0.02～0.18 m/s；
阶段 4：加入零命令、起步和停车；
阶段 5：加入 yaw；
阶段 6：加入 vy 和后退。
检查并删除会把 0.02～0.15 m/s 命令误置零的 deadband。
stand 状态必须通过独立采样标志控制。
```

## Task 7：部署一致性自动检查

```text
生成 policy_metadata.yaml，至少包含：
- joint_names/order；
- motor ids；
- joint signs；
- zero offsets；
- default joint positions；
- action scale；
- observation names/order/scales；
- control dt；
- history length/order；
- policy input/output dimensions；
- ONNX sha256。
编写训练端和 ROS2/C++ 部署端共同校验工具，发现不一致时拒绝启动电机。
```

## Task 8：Sim2Real 随机化分层

```text
将 domain randomization 划分为：
DR0 nominal；
DR1 measured narrow；
DR2 moderate；
DR3 stress test。
参数包含 mass、CoM、friction、motor strength、Kp/Kd、joint friction、encoder offset、IMU noise/bias、action latency、observation latency、push、slope。
所有范围写入独立 YAML，并在 TensorBoard 记录每个 episode 实际采样值。
```

## Task 9：评估程序

```text
实现 evaluate_crawl.py：
- 至少 1000 个并行/连续 episode；
- 多随机种子；
- 输出 fall rate、contact F1、min stance feet、support margin、slip、velocity RMSE、orientation、impact、torque margin；
- 按 nominal、DR1、DR2、DR3 分组；
- 保存 CSV、JSON、Markdown 报告；
- 自动判定是否允许进入 MuJoCo sim2sim。
```

## Task 10：MuJoCo 验证

```text
为 Bennett 建立 MuJoCo 模型和推理适配层。
确保：
- 同样的关节顺序；
- 同样的 default pose；
- 同样的 PD；
- 同样的 observation/action scaling；
- 同样的 latency/filter；
- 同样的 policy frequency。
运行 crawl 策略并生成与 Isaac Lab 相同的评估指标。
```

---

# 19. 第一版配置建议

以下仅作为启动参数，需根据 Bennett 尺寸、电机和真机测量修改。

```yaml
sim:
  dt: 0.005

control:
  decimation: 4
  policy_frequency_hz: 50
  action_type: joint_position_offset
  action_scale:
    hip_abduction: small
    hip_flexion: medium
    knee: medium
  action_rate_limit: enabled

gait:
  type: crawl
  frequency_hz: [0.45, 0.65]
  duty_factor: [0.84, 0.90]
  max_swing_legs: 1
  min_stance_legs: 3
  four_leg_overlap: true
  swing_height_ratio_to_leg_length: [0.03, 0.06]

commands:
  initial_vx: [0.08, 0.15]
  final_vx: [0.02, 0.18]
  initial_vy: [0.0, 0.0]
  initial_yaw: [0.0, 0.0]
  stand_probability: 0.10
  stand_deadband: 0.005

observations:
  base_ang_vel: true
  projected_gravity: true
  commands: true
  joint_pos_error: true
  joint_vel: true
  previous_action: true
  gait_phase: true
  desired_contacts: true
  history_length: 10

rewards:
  velocity_tracking: enabled
  yaw_tracking: enabled
  contact_schedule: enabled
  less_than_three_contacts: strong_penalty
  multiple_swing_legs: strong_penalty
  stance_slip: enabled
  support_margin: metric_first
  swing_clearance: phase_gated
  touchdown_impact: enabled
  orientation: enabled
  vertical_velocity: enabled
  action_rate: enabled
  joint_limits: enabled
  undesired_contacts: enabled
  feet_air_time_generic: disabled

termination:
  illegal_body_contact: true
  excessive_tilt: true
  base_height: true
  joint_limit: true
```

---

# 20. 真机安全清单

上电前：

```text
[ ] 急停可用
[ ] 吊架/保护绳
[ ] 单关节方向逐个验证
[ ] 电机 ID 与 joint mapping 校验
[ ] 编码器零位校验
[ ] IMU 坐标系校验
[ ] 关节位置/速度/力矩限制
[ ] 电流和温度保护
[ ] 控制超时 watchdog
[ ] CAN 掉线保护
[ ] 策略输入 NaN/Inf 检查
[ ] ONNX metadata 校验
[ ] 默认姿态缓慢插值进入
[ ] 动作幅度从 10% 逐步放开
[ ] 初次 gait frequency 降低
```

运行中：

```text
[ ] 每周期 deadline
[ ] 最新反馈时间
[ ] 电机故障码
[ ] 接触异常
[ ] roll/pitch
[ ] base height
[ ] 多腿意外离地
[ ] joint saturation
[ ] policy output saturation
```

触发异常时：

```text
不要直接全部失能使机器人自由坠落；
应根据硬件设计进入安全阻尼、趴下或受控停机状态。
```

---

# 21. 最终建议优先级

## 必须先做

1. 审计关节顺序、零位和 action mapping；
2. 实现 Crawl gait scheduler；
3. 加入 gait phase 与 desired contact 观测；
4. 关闭/修改通用 `feet_air_time`；
5. 排查慢速 command deadband；
6. 加入接触匹配、滑移和整周期位移；
7. 从固定平地、固定速度、固定步频开始；
8. 建立训练端—ONNX—ROS2 观测一致性测试。

## 第二阶段

1. 支撑多边形和静态稳定裕度；
2. observation history；
3. 延迟与执行器模型；
4. 窄范围域随机化；
5. PACE/系统辨识；
6. MuJoCo sim2sim。

## 第三阶段

1. 更低速度；
2. 起停；
3. 转弯、侧向、后退；
4. 地形和推力；
5. 逐个训练 trot、pace 等步态；
6. 多 gait conditioned policy 或蒸馏。

---

# 22. 一句话方案

> 对 Bennett 先训练“确定性 Crawl 接触调度器约束下的关节位置残差策略”：固定一次只摆一条腿，使用 gait phase 和 desired contact 作为观测，以接触匹配、三足支撑、支撑裕度、低滑移、低冲击和整周期位移为核心指标；先完成可信模型、系统辨识和 MuJoCo sim2sim，再上真机。不要把普通 velocity task 的速度范围改小就当成静态 Crawl。

---

# 23. 主要资料链接

## 基础框架

- Isaac Lab  
  https://github.com/isaac-sim/IsaacLab
- RSL-RL  
  https://github.com/leggedrobotics/rsl_rl
- Legged Gym  
  https://github.com/leggedrobotics/legged_gym
- IsaacLab Tutorial  
  https://github.com/Lab-of-AI-and-Robotics/IsaacLab-Tutorial

## 步态与强化学习

- Walk These Ways  
  https://github.com/Improbable-AI/walk-these-ways
- Walk These Ways paper  
  https://arxiv.org/abs/2212.03238
- basic-locomotion-isaaclab  
  https://github.com/iit-DLSLab/basic-locomotion-isaaclab
- robot_lab  
  https://github.com/fan-ziqi/robot_lab
- Animator Notebook Quadruped Gaits  
  https://www.animatornotebook.com/learn/quadrupeds-gaits

## 部署与验证

- rl_sar  
  https://github.com/fan-ziqi/rl_sar
- Unitree RL Lab  
  https://github.com/unitreerobotics/unitree_rl_lab
- Unitree MuJoCo  
  https://github.com/unitreerobotics/unitree_mujoco
- MuJoCo Playground  
  https://github.com/google-deepmind/mujoco_playground

## 系统辨识与 Sim2Real

- PACE Sim2Real  
  https://github.com/leggedrobotics/pace-sim2real
- SPI-Active  
  https://github.com/LeCAR-Lab/SPI-Active
- Sim-to-Real: Learning Agile Locomotion for Quadruped Robots  
  https://arxiv.org/abs/1804.10332

## 传统轨迹与接触规划

- CHAMP  
  https://github.com/chvmp/champ
- TOWR  
  https://github.com/leggedrobotics/towr
- Crocoddyl  
  https://github.com/loco-3d/crocoddyl

---

# 24. 给 Codex 的总提示词

```text
你现在负责把现有 Bennett 四足 Isaac Lab Manager-Based velocity locomotion 任务，改造成“超慢速、高静态稳定性、高 Sim2Real 成功率”的 crawl gait 任务。

核心硬约束：
1. 任意时刻期望支撑腿数量 >= 3；
2. 任意时刻期望摆动腿数量 <= 1；
3. gait scheduler 决定允许哪条腿摆动，RL 不得自由改变落足顺序；
4. actor 观测必须能在真机获得；
5. critic 可使用特权信息；
6. 第一版动作为 default joint position 上的关节位置残差；
7. 禁止使用会鼓励动态跳跃的大权重通用 feet_air_time reward；
8. 必须检查小速度命令是否被 deadband/standing mask 清零；
9. 训练端、ONNX、ROS2/C++ 部署端的 observation 和 joint order 必须自动校验；
10. 所有域随机化范围必须集中配置、可关闭、可做消融；
11. 先完成固定平地、固定前进速度、固定 crawl，再逐步降低速度和增加扰动；
12. 进入真机前必须通过另一个物理引擎的 sim2sim；
13. 不要直接修改 Isaac Lab 源码，要在 Bennett extension 内实现；
14. 每一步提交可独立运行的代码、测试和 Markdown 说明；
15. 每次只完成一个小阶段，不要一次大范围重构。

第一步只做项目审计，生成：
- 环境目录结构；
- observation/action 精确清单；
- joint mapping；
- command deadband；
- reward；
- domain randomization；
- contact sensor；
- export/deployment pipeline；
- 当前与 crawl 目标之间的差距。
输出 BENNETT_CRAWL_GAP_ANALYSIS.md，不修改任何文件。
```
