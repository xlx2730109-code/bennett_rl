# Bennett Go2 模板 model_2250 诊断报告（平地矩阵 + 训练地形矩阵）

checkpoint：`model_2250.pt`（本目录上一级），任务 `Isaac-BennettRL-Rough-BennettGo2-Play-v0`，
BENNETT_CFG_V1 资产（12.2629 kg，足端 FL_1/FR_1/RL_1/RR_1），effort ±20 N·m 选型配置，
命令包络 ±0.50 / ±0.25 / ±0.60（m/s, m/s, rad/s），评估不截断动作（与训练一致）。

数据由 `scripts/rsl_rl/collect_bennett_policy_diagnostics.py` 确定性采集（seed 42，单环境，
每场景 1.5 s 静置 + 5 s 命令 + 1.5 s 恢复，1 间隔回放）。

## flat_matrix/ — 平地全方向命令矩阵

- CSV：`Isaac-BennettRL-Rough-BennettGo2-Play-v0_20260906_044737.csv`
- 场景：stand + 6 方向 × 3 档速度（慢/中/快 = 训练包络的 25%/57%/91%）= 19 场景
- 采集地面：平面 mesh（关掉训练地形，看纯指令跟踪与电机负载基线）
- 图：`01_torque_timeseries` ~ `09_velocity_tracking`（含义见 `plot_motor_report.py` 文件头）

## terrain_matrix/ — 六种训练地形矩阵（rough）

- CSV：`Isaac-BennettRL-Rough-BennettGo2-Play-v0_rough_20260906_051433.csv`
- 场景：重建训练 curriculum 地形（proportion 均分，列=地形一一对应），
  6 地形 × Level 0 / Level 2 两档难度 × 6 命令（stand、前进中/快、左移中、左/右转中）
  = 72 段；每段把机器人钉到对应 curriculum patch 上再 reset
- 地形（列序=curriculum 列序，机器人视角）：
  | 列 | 名字 | 生成参数 |
  |---|---|---|
  | 0 | stairs 上行 | 台阶 0.05–0.23 m，宽 0.3 m |
  | 1 | stairs 下行 | 同上（倒置） |
  | 2 | boxes | 0.45 m 网格，高 0.025–0.1 m |
  | 3 | random_rough | 噪声 0.01–0.06 m，步长 0.01 |
  | 4 | slope 上行 | 坡度 0–0.4 rad |
  | 5 | slope 下行 | 同上（倒置） |
  Level 0 ≈ 训练 curriculum 第 0 行（最易），Level 2 ≈ 第 2 行（难度 ~50%）
- 图：`00_terrain_command_overview`（总览：6 命令 × 6 地形，L0/L2 vs 指令，格内数字=RMS）、
  `01_torque_heatmap_terrain`、`02_torque_speed_terrain`、`03_gait_rhythm_terrain`、
  `04_cot_by_terrain`、`05_body_attitude_terrain`
- **通过扭矩（选型导向，`scripts/analysis/plot_pass_torque.py`）**：只取 fwd mid/fast
  24 个穿越段（每段实际通过 1.4–2.3 m，全部完整走完），
  `06_pass_torque_envelope`（逐地形最坏关节 |τ| 时序包络 vs DM8006 8/20 参考线）、
  `07_torque_demand_hist`（逐地形 |τ| 样本分布，>8 N·m 占比 / 触 20 计数）、
  `pass_torque_summary.csv`（24 段汇总表：通过距离/峰值/RMS/超 8 占比/触顶数）。
  结论：下台阶需求最高（L2 fast 峰值 20 触顶 17 次，>8 占 14.9%），
  random_rough 与 boxes L2 fast 峰值 19.8–19.9 无触顶，
  上台阶/坡面峰值 14.1–17.0；峰值关节几乎全为 RL_calf。

## 绘图脚本

- `scripts/analysis/plot_motor_report.py --csv <flat CSV>`
- `scripts/analysis/plot_terrain_report.py --csv <rough CSV> --mass 12.2629`
