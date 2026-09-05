# 模块结构
# 文件	作用
dm8006_envelope.py	纯 numpy/torch 包络核心(不依赖 isaaclab,可独立测试/绘图)：从数字化 CSV 取扫频下降支(13 N·m@73rpm → 9 N·m@118.6rpm),两端接说明书锚点(堵转 20 N·m、空载 190rpm),平台抖动行丢弃，两端无数据区间用保守直线弦
damiao.py	DamiaoMotorCfg + DamiaoMotor 执行器:DelayedPDActuator 基类(min/max_delay 默认 0,CAN 延迟想开就改两个数)，compute() = 标准 MIT-PD,_clip_effort() = 先 ±effort_limit 平削(电流限)再 τ_max(|ω|) LUT 包络削(电压限)，四象限对称——和真机 DM MIT 模式管线逐位一致
plot_dm_j8006_envelopes.py	四象限包络验证图(上图)：达妙 LUT vs 现役 DCMotor 8/20/19.9 + 全部数字化点 + 额定星标，独立运行无需开 sim
数据四件套(保留)	官方曲线 PNG、说明书 PDF、数字化 CSV(含效率/电流/功率)、参数契约 YAML
__init__.py	只导出 numpy 核心——因为根包 bennett_rl/__init__ 会拉起整个 isaaclab,执行器类须 from bennett_rl.assets.motor.damiao import DamiaoMotorCfg 直接导入
已删(git 41cdf2f 里可恢复)：7 月的两个一次性脚本 plot_dm_j8006_curve.py(曲线复现验证，已跑过一次定格为 CSV)、plot_dm_j8006_actuator_envelopes.py(A/B 选型，决策已定型进 bennett.py)及 generated/ 旧图。

# 验证结果(全过)
数值：torch 插值 vs numpy.interp 最大误差 2.7e-6 N·m;包络单调性断言通过
AppLauncher 冒烟(真实实例化+compute):6 组削剪全 PASS——|ω|=3→削到 8(电流限 binds)、12.57→8、15→5.89(包络 binds)、19.9→0、22(超空载)→0、负向对称 −5.89;延迟变体(min1/max2)正常
关键数字：包络 @堵转 20 / @额定 120rpm 8.82 / @15rad/s 5.89;重要结论——effort_limit=8 时新旧模型行为几乎一致(包络处处 ≥8,平削先 bind),只有像 bennett_go2 那样放开到 ±20 才显差异
# 怎么用
以后哪个任务要真机保真，把执行器组换掉即可(参数用法与 DCMotorCfg 完全同构):


from bennett_rl.assets.motor.damiao import DamiaoMotorCfg

"base_legs": DamiaoMotorCfg(
    joint_names_expr=[".*_thigh", ".*_calf"],
    effort_limit=8.0, velocity_limit=19.8967,
    stiffness=30.0, damping=2.0,
    # min_delay=1, max_delay=2,   # 可选:CAN 往返延迟随机化
)