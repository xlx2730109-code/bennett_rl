# Bennett Stair1

`quad_leg_stair1` 是一个独立、从零训练的 Bennett 单向上楼任务。它不继承
Trot1、Go2-10、Gravel 或其他自定义任务；只使用 Isaac Lab 通用速度任务组件和
`BENNETT_CFG_V6` 机器人资产。

## 已锁定的任务合同

- 只训练世界坐标 `+X` 方向上楼，不训练下楼。
- 每条楼梯6级，前方有0.60 m平地起步区和顶部平台。
- 10个课程等级精确对应每级台阶高1、2、...、10 cm。
- 11种踏面深度覆盖25、26、...、35 cm，并行分布在不同训练列。
- 前进命令采用Slope4已训练成功的0.16–0.24 m/s范围。
- 横向速度目标为0；训练环境根据世界+X航向误差生成±0.40 rad/s以内的偏航
  修正命令，使真实IMU航向闭环可以复现走偏后的纠偏输入。
- 课程从1 cm开始。只有一批运动环境中至少70%完成“全四脚进入顶部平台并
  稳定0.25 s”，才整体升到下一个高度；失败不会误升级。

## 真机可迁移观测与动作

部署 actor 固定为33维，只包含真机已有信号：

1. IMU机身角速度3维；
2. IMU投影重力3维；
3. 速度命令3维；
4. 8个关节相对位置；
5. 8个关节速度；
6. 上一时刻8维动作。

actor 不读取基座线速度、地形高度扫描或足端接触力。训练 critic 额外读取基座
线速度、高度扫描、四足接触状态和课程等级；这些特权信息不会进入导出的 actor。
偏航修正仍位于原有3维速度命令中，因此actor维度保持33维；真机部署时应由IMU
航向与上楼目标航向计算同类yaw命令。

动作合同为50 Hz策略频率、8维关节位置残差、`scale=0.20 rad`、runner动作裁剪
`±3.5`，并保留绝对关节目标安全范围。执行器使用实测合同：Kp=28、Kd=2、
8 Nm持续/20 Nm峰值、19.896753 rad/s速度限制。

## Sim2Real随机化

训练包含以下域随机化：

- 静摩擦0.6–1.3、动摩擦0.5–1.1、恢复系数0–0.02；
- base质量0.90–1.10倍；
- base质心x/y方向±2 cm、z方向±1 cm；
- 刚度0.80–1.20倍、阻尼0.70–1.30倍；
- 关节初始角±0.04 rad、初始速度±0.05 rad/s；
- 起点、横向位置、朝向及初速度小范围随机；
- 每8–14 s施加一次有限速度扰动；
- IMU与关节观测使用Isaac Lab通用观测噪声。

关节顺序、动作尺度、控制频率和电机力矩限制不随机，因为它们属于部署时必须
严格一致的动作合同。

## 从Slope4吸收但不继承的部分

- 保留相同的前进速度区间、世界+X航向纠偏、动作/执行器合同和域随机化量级。
- 不加载Slope4 checkpoint，也不复制它的50维相位/目标接触观测。
- 不规定对角腿序或固定摆动周期；宽速度跟踪核、轻量通用离地时间奖励和低权重
  台阶净空奖励只负责帮助从零发现运动，精细速度核负责后续跟踪精度。
- 低机身失败阈值放宽到0.16 m用于早期恢复，但到达顶部仍要求至少0.20 m净空。

## 训练

必须从新run开始，不得加载Trot1或其他任务的checkpoint：

```powershell
Set-Location 'E:\Project\Isaaclab\bennett_rl'

D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\rsl_rl\train.py `
  --task Isaac-BennettRL-QuadLeg-Stair1-v0 `
  --max_iterations 250 `
  --run_name stair_guard_250 `
  --headless
```

250轮门槛训练至少应满足：学习率始终为`0.0003`；noise std保持有限且为正；
`base_contact`、`low_base_clearance`和`outside_lane`没有持续恶化；课程不会在无
`stair_success`时升级；`uphill_velocity_progress`不应长期接近0；确定性策略能在
1 cm台阶产生真实前进、抬脚和四足接触切换。未通过这些门槛时不要直接跑满
3000轮。

门槛通过后，可从同一个Stair1 run继续附加2750轮。不得混入旧任务checkpoint。

## 可视化验证

```powershell
D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\rsl_rl\play.py `
  --task Isaac-BennettRL-QuadLeg-Stair1-Play-v0 `
  --num_envs 1
```

Play配置关闭观测噪声、质量/质心/增益随机化和外部推扰，用于观察名义策略。
