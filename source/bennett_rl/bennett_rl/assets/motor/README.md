# bennett_rl/assets/motor — 达妙 DM-J8006-2EC 电机资产

三个子目录,按"训练/部署真正用到的"与"参考资料/复现工具"分开:

```
motor/
├── __init__.py                     # 包入口:只导出 numpy 包络核心(不拉 isaaclab)
├── dm8006_envelope.py              # 核心:纯 numpy/torch τ-ω 包络(见下)
├── damiao.py                       # 核心:DamiaoMotorCfg/DamiaoMotor 执行器(Isaac Lab)
├── dm_j8006_24v_120rpm_curve.csv   # 核心:数字化扫频数据(被 dm8006_envelope 按同目录路径读取,勿挪)
├── README.md                       # 本文件
├── docs/                           # 参考文档(纯人读,无代码读取)
│   ├── DM-J8006-2EC V1.1减速电机说明书V1.0.pdf
│   ├── 24V 120RPM 8006电机性能曲线图.png   # 厂家性能曲线(CSV 的数字化来源)
│   └── dm_j8006_2ec_v1_1_24v.yaml  # 参数契约:规格+来源+待辨识项清单
└── tools/                          # 不常用:复现绘图工具 + 其输出
    ├── plot_dm_j8006_envelopes.py  # 厂家形态复现图(曲线三件套+包络对比),独立运行无需开 sim
    └── generated/                  # 上脚本的输出(git 忽略)
```

## 核心三件(训练/部署链路)

**`dm8006_envelope.py`** — 纯 numpy/torch 包络核心(不依赖 isaaclab,可独立测试/绘图)。
从数字化 CSV 取扫频下降支(13 N·m@73rpm → 9 N·m@118.6rpm),两端接说明书锚点
(堵转 20 N·m、空载 190rpm),平台抖动行丢弃,两端无数据区间用保守直线弦。
关键常数:RATED 8 N·m@120rpm、PEAK 20 N·m、NO_LOAD 190rpm,全部关节侧(6:1 减速后)。

**`damiao.py`** — `DamiaoMotorCfg` + `DamiaoMotor` 执行器:`DelayedPDActuator` 基类
(min/max_delay 默认 0,CAN 延迟想开就改两个数),`compute()` = 标准 MIT-PD,
`_clip_effort()` = 先 ±effort_limit 平削(电流限)再 τ_max(|ω|) LUT 包络削(电压限),
四象限对称——和真机 DM MIT 模式管线逐位一致。

**`dm_j8006_24v_120rpm_curve.csv`** — 唯一被代码读取的数据文件
(`dm8006_envelope.CURVE_CSV = Path(__file__).parent / "..."`,必须与本模块同目录)。
含效率/电流/功率列,官方曲线 PNG 的数字化产物。

## 它是真电机吗?(直白版)

**一句话:曲线是真的,但你现在训练还没用它。**

`DamiaoMotor` 干的事:每个力矩输出前过两道削剪——
1. 先按 ±effort_limit 平削(= 真机电流限);
2. 再按官方实测曲线削:转速越高、能给的最大力矩越低(空载 190rpm 时 0 → 堵转时 20 N·m)。
这条曲线就是从 `docs/` 里那张官方 PNG 数字化来的,四象限对称。**只要任务把执行器换成
`DamiaoMotorCfg`,训练时的电机能力就按这条真机曲线约束**——DCMotor 8/20/19.9 那种
两参数直线,高速段给力给多了,拟不出"转快了没劲"。

但它没建的部分,别高估:
- 堵转段(0→73rpm)和空载段(118.6→190rpm)厂家没给扫频数据,是**保守直线弦**;
- 只有稳态边界:没有电流环带宽、**没有热模型**(真机 20 N·m 只能短时,仿真里持续用不会过热);
- 没有减速箱背隙/摩擦、编码器噪声;CAN 延迟默认 0,想开要在 cfg 里配 min/max_delay。

**现状(2026-09-08 查过)**:全仓库没有任何任务 import `DamiaoMotorCfg`,
free_gait4 等所有任务仍在用 `DCMotorCfg 8/20/19.9`。两者关系:
机器人工作点 ≤3 rad/s 时行为一致(包络处处 ≥8,电流限先 bind,碰不到曲线);
高转速才有差别(如 15 rad/s:Damiao 削到 5.89、DCMotor 削到 4.92,DCMotor 反而更保守)。

**什么时候换**:哪个任务要真机保真(尤其要放开 ±20 的),把执行器组换掉即可(用法见下)。
**一次只动一个变量**:奖励还在调的任务别同时换执行器,不然出了问题分不清谁的锅。

## 验证结果(全过,2026-07 定型)

- 数值:torch 插值 vs numpy.interp 最大误差 2.7e-6 N·m;包络单调性断言通过
- AppLauncher 冒烟(真实实例化+compute):6 组削剪全 PASS——|ω|=3→8(电流限 binds)、
  12.57→8、15→5.89(包络 binds)、19.9→0、22(超空载)→0、负向对称 −5.89;延迟变体(min1/max2)正常
- 关键数字:包络 @堵转 20 / @额定 120rpm 8.82 / @15rad/s 5.89;
  **effort_limit=8 时新旧模型行为几乎一致**(包络处处 ≥8,平削先 bind),
  只有像 bennett_go2 那样放开到 ±20 才显差异

## 怎么用

以后哪个任务要真机保真,把执行器组换掉即可(参数用法与 DCMotorCfg 完全同构):

```python
from bennett_rl.assets.motor.damiao import DamiaoMotorCfg

"base_legs": DamiaoMotorCfg(
    joint_names_expr=[".*_thigh", ".*_calf"],
    effort_limit=8.0, velocity_limit=19.8967,
    stiffness=30.0, damping=2.0,
    # min_delay=1, max_delay=2,   # 可选:CAN 往返延迟随机化
)
```

注意:`bennett_rl` 根包 `__init__` 会拉起 isaaclab,所以在无 sim 的离线脚本里,
执行器类必须 `from bennett_rl.assets.motor.damiao import DamiaoMotorCfg` 直接导入
(包入口 `__init__.py` 只导出 numpy 核心,不会把 isaaclab 拖进来)。

## 工具

重画包络对比图(验证 CSV→包络→图全链路,兼作回归):

```
python source/bennett_rl/bennett_rl/assets/motor/tools/plot_dm_j8006_envelopes.py            # 出图到 tools/generated/
python source/bennett_rl/bennett_rl/assets/motor/tools/plot_dm_j8006_envelopes.py --validate-only   # 只打印关键数字
```

已删(git 41cdf2f 里可恢复):7 月的两个一次性脚本 plot_dm_j8006_curve.py(曲线复现验证,
已跑过一次定格为 CSV)、plot_dm_j8006_actuator_envelopes.py(A/B 选型,决策已定型进 bennett.py)。
