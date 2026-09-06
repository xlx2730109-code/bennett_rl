# 6 张新图，每张都按地形对比组织
图	内容
00_terrain_command_overview	总览：6 命令 × 6 地形 36 格，每格画 Level 0（浅色）/ Level 2（深色）实际 vs 指令，格内数字=跟踪 RMS
01_torque_heatmap_terrain	8 关节 × 12（地形×难度）峰值扭矩热图，标数字
02_torque_speed_terrain	工作点按地形着色 + DM8006 包络，量程扩到 ±20 露出触顶带
03_gait_rhythm_terrain	12 行步态 raster（每地形×难度一行），右侧 duty
04_cot_by_terrain	CoT 分组条形：5 命令组 × 12 条
05_body_attitude_terrain	快行俯仰/横滚时序，同列统一刻度跨地形可比

# 关键发现
扭矩：平地峰值 14.5 N·m；Level 2 下台阶快行时后腿 RL/RR 的 thigh+calf 四处打满 ±20 限幅——地形对电机的需求比平地高一整个档，选型余量就靠 peak 区间撑
跟踪：上台阶中速 RMS 0.027，与平地同级，练得最扎实；弱项是下台阶快行（RMS 0.365，两次俯冲到 -0.85 m/s）和 random_rough 上的原地转（RMS 0.19-0.26）
能耗：台阶行走 CoT 0.49-0.73，坡面最省（0.35-0.53）；random_rough 上 yaw 的 CoT 反而全场最低（1.1-1.6）——因为它在那根本没怎么转
姿态：台阶上俯仰摆到 ±0.45-0.65（约 30°），其余地形 ±0.15 以内
RR 拖腿（duty 0.70-0.87）在所有地形上都延续，是策略固有习惯