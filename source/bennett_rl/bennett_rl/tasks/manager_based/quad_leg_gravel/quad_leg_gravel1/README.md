# Bennett Gravel1 Sim2Real

独立的 Bennett 固定碎石崎岖地形任务。它不继承其他 Bennett 实验、不加载旧
checkpoint，也不修改 GO2 或共享 locomotion 配置。

## 真机接口契约

- Actor 输入保持原有 33 维：机身角速度、投影重力、3 维速度命令、8 关节位置、
  8 关节速度、8 个上一时刻动作。
- Actor 不读取仿真 base 线速度或高度图。训练导出的 actor 因而不依赖真机上
  不存在的传感器。
- Critic 额外读取 3 维 base 线速度和高度扫描，形成非对称 actor-critic；这些
  privileged observations 不进入导出策略。
- 动作顺序固定为 `FL/FR/RL/RR`，每条腿 `thigh, calf`；动作缩放 `0.20 rad`，
  runner clip 为 `±3.5`，目标角继续受任务中的绝对关节限位约束。
- 电机模型保持 Bennett 真机标定值：`Kp=28`、`Kd=2`、连续力矩 `8 N·m`、
  饱和力矩 `20 N·m`、速度上限 `20 rad/s`。

## 域随机化

- 静/动摩擦：`0.6–1.3 / 0.5–1.1`，恢复系数 `0–0.02`。
- base 质量缩放：`0.9–1.1`，并重算惯量。
- base 质心：x/y `±0.02 m`，z `±0.01 m`。
- PD 增益缩放：刚度 `0.8–1.2`，阻尼 `0.7–1.3`。
- reset 时关节位置/速度、机身平动与角速度施加小扰动；出生位置和朝向在中央
  平台内随机化。
- 不使用持续外力或定时 push 掩盖地形步态问题。

## 命令、奖励和课程

- 一个 20 秒回合只使用一个显式命令模式，覆盖站立、前进、后退、左右横移、
  原地转向以及平移加转向；避免旧版两个 10 秒命令的世界位移互相抵消。
- 线速度范围 x `±0.35 m/s`、y `±0.12 m/s`，角速度 `±0.50 rad/s`。
- 奖励不指定固定步态相位，但要求机身高度/姿态、至少两足支撑、摆腿离地、
  各腿参与和支撑足少滑动，抑制低趴拖行策略。
- 地形为 `10 rows × 20 columns` 固定随机网格，石块半高度从约 0 增至
  `0.04 m`。训练从 0–1 级开始。
- 课程依据完整回合的线速度/转向跟踪积分以及是否存活来升降级，不再使用净
  位移。TensorBoard 中重点检查 `Curriculum/terrain_levels/mean_level`、
  `promotion_rate`、`demotion_rate`、`linear_score` 和 `yaw_score`。

## 任务与命令

- Train: `Isaac-BennettRL-QuadLeg-Gravel1-v0`
- Play: `Isaac-BennettRL-QuadLeg-Gravel1-Play-v0`

## PPO 数值安全与训练门槛

- 使用 `log_std`，避免 RSL-RL 的无约束 scalar std 在优化中变成负数。
- 初始 std 为 `0.6`；学习率固定为 `3e-4`，不允许 adaptive schedule 将其
  放大到 `1e-2`。
- entropy 系数为 `0.003`，100-step rollout 对应 2 秒，覆盖各腿参与奖励的
  `0.65–1.20 s` 状态时间尺度。
- `2026-08-11_12-00-08` 使用旧 scalar std，已在 iteration 186 崩溃，且
  `model_150` 确定性测试存在低净空重置；不得从该 run `--resume`。

先做全新的 250 轮门槛训练（不要加 `--resume`）：

```powershell
Set-Location 'E:\Project\Isaaclab\bennett_rl'
D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\rsl_rl\train.py `
  --task Isaac-BennettRL-QuadLeg-Gravel1-v0 `
  --max_iterations 250 `
  --run_name ppo_guard_250 `
  --headless
```

250 轮结束后，只有同时满足以下条件才继续长训：

- `Loss/learning_rate` 始终为 `0.0003`；
- `Policy/mean_noise_std` 有限、为正且没有持续冲向 0 或快速增大；
- `Episode_Termination/low_base_clearance` 呈下降趋势，不能继续维持在约 0.5；
- `Train/mean_episode_length` 明显增长；
- 使用 `model_200.pt` 或最终保存的 `model_249.pt` 做确定性
  forward/reverse/yaw rollout，
  实际机身速度和四足接触切换均有效，不接受只看训练 reward。

门槛通过后可以续接同一 run，保留已经验证为有限值的 log-std 和 fixed-LR
optimizer，避免浪费前 250 轮：

```powershell
Set-Location 'E:\Project\Isaaclab\bennett_rl'
D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\rsl_rl\train.py `
  --task Isaac-BennettRL-QuadLeg-Gravel1-v0 `
  --resume `
  --load_run <时间戳>_ppo_guard_250 `
  --checkpoint model_249.pt `
  --max_iterations 1750 `
  --headless
```

这里的 `--max_iterations 1750` 是续跑的附加轮数，不是绝对终止轮数。若门槛不
通过则不要续跑，也不要把旧 scalar-std checkpoint 转换后混入新 optimizer。

可视化检查某个新 run：

```powershell
Set-Location 'E:\Project\Isaaclab\bennett_rl'
D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\rsl_rl\play.py `
  --task Isaac-BennettRL-QuadLeg-Gravel1-Play-v0 `
  --load_run <新训练目录名> `
  --checkpoint <model_xxx.pt>
```

正式长训至少同时满足以下条件再进入吊架/低速真机测试：课程平均等级持续高于 6，
后半程升级率不再长期为 0，平均回合长度接近 20 秒，且 Play 中最高等级不会
持续低趴、拖脚或频繁触发 base/低净空终止。真机首测仍需吊架、低速命令和急停。
