# Bennett(四足 · DM8006 · 50Hz)—— 从零到真机部署(sim2real)全流程清单

> 本清单是"从 USB 口插上真机到稳稳走动"的完整检查表。分阶段、逐项打勾,**每一行都标注了当前状态**
> (✅ 已做 / ⚠️ 部分或待补 / ⬜ 未做 / 🔜 以后做)。目标:尽量不出错,像 sim2sim 那样每一步都验证。
>
> 两条主线贯穿全篇:
> - **两阶段部署**:阶段A = 用**电脑**驱动真机做带权验收;阶段B = 真机走稳后**换 Jetson** 做机载生产部署。
> - **可观测 + 随机化 + 执行器对齐** 是能否迁移的三支柱——你的 `quad_leg_trot1` 已经做到前两根,第三根(执行器/延迟)还没做,是本文的重点。

> 📌 **2026-09-05 重大更新**:经审计 `E:\HuanCun\Desktop\u2canfd`,**真机通信桥其实早就建好了**,而且不止会点通信 —— 里面有一个**能闭环推理策略 + 读 IMU + 高帧率写 CSV** 的成熟运行时(`quad_leg_xu/bennett_deploy`)。同时你确认了:**DM8006 额定 8N·m / 峰值 20N·m,±8 是对的**,以及 **obs 冻结、只做离线校验不改**。本文据此全面修订。

---

## 0. 总体一句话:缺什么?

| 内容 | 状态 | 说明 |
|---|---|---|
| 仿真训练(Isaac Lab + rsl_rl) | ✅ | 多任务已训练,含 DR + 硬件可观测 |
| sim2sim 保真(MuJoCo 复现 obs→action) | ✅ | trot1 / track / trace 三任务跑通,foot-trail 可视化 |
| hardware-observable 契约 | ✅ | trot1 等删除了 `base_lin_vel` / `height_scan` |
| 域随机化 | ✅ | 摩擦/质量/执行器增益/随机外力/推挤 |
| 策略导出(play.py → .pt / .onnx) | ✅ | 每 run `exported/` 里都有 |
| **真机通信桥(CAN-FD / DM SDK + IMU)** | ✅ 已有 | 在 `E:\HuanCun\Desktop\u2canfd`,非仓库内;见 `bennett_deploy`(详见阶段4) |
| **系统辨识(执行器有效增益/摩擦/惯量/背驱/延迟)** | ⬜ 未做 | 数据手册明确要求做;**但桥已自带高帧率 CSV 日志,正好用它采集** |
| **真实延迟测量 + 补偿** | ⬜ 未做 | 你还没做过;本文给出逐步方法 |
| **真机测试序列(站立→零速→慢→全速→扰动)** | ⬜ 未做 | 有规划文档但没跑过 |
| 部署代码"硬件无关化"(PC/Jetson 同套) | ⚠️ 部分 | `bennett_deploy` 已是"硬件无关 runtime"形态,但换 Jetson 要去掉 Windows 专属 DLL/定时器/键盘 shim(详见阶段6) |
| 速度估计器 | ✅ 不需要 | trot1 只用 `base_ang_vel` + `projected_gravity`(IMU 可测),无 `base_lin_vel` 特权量 |

> **最大的三个"坑"**(先记住,后面各节展开):
> 1. ⚠️ **已修正理解**:仿真 `forcerange -8 8` **是对的,不要改成 ±20**。DM8006 **额定 8N·m(可持续)、峰值 20N·m(仅瞬态)**。策略指令通常在几 N·m 以内(DR 推挤外力 ≤3N),±8 持续上限完全够;训练端 `saturation_effort=20` 只是给网络留的峰顶余量,**真机不需要也不应该持续顶到 20**。部署端用 benerick 的"软件力矩包络"把指令 clamp 在安全区即可,别去改 sim 的 forcerange。
> 2. 🔑 **真正的关键点(比之前想的好)**:真机不是"不可控的内部位置闭环",而是 **MIT 阻抗位置指令** `control_mit(kp, kd, q, dq, tau)`,其内部 `tau_cmd = kp*(q − q_actual) + kd*(dq − dq_actual) + tau` —— **这跟 sim 的位置执行器 `kp*err + kv*(−v)` 逐项同构**。手册确认位置模式是电机**内置真实 PD 伺服**,`Kp∈[0,500] Kd∈[0,5]`(sim 的 28/2 都在范围内),且**增益每周期随包下发、是你自己定的**。但要注意:内部 PD 跑在**电机自己的编码器**上 —— 电机是 **6:1 减速 + 14bit 磁编**,所以给定的 kp/kd 在**关节输出端**等效的刚度/阻尼会被减速比重新折算,而且编码器可能在电机轴(关节角 = $q/6$)也可能在输出端(关节角 = $q$)。所以真正要辨识的是**"给定 MIT kp/kd,关节输出端实际等效刚度/阻尼是多少 N·m/rad",以及编码器到底测的是哪端** —— 这两件事定了,才能对齐 sim 的 28/2。
> 3. **URDF 里所有关节限位是 0(冻结)** —— 真实限位只存在于 config/MJCF。真机部署必须在**软件里重新强制**这些限位,不能只依赖电机出厂默认。

---

## 阶段 1 · 硬件核对(进行中)

> 目标:确认"真机能提供的观测"和"训练时的观测"在**坐标系、顺序、方向、数值范围**上完全一致,并确立基础安全。

- [ ] **关节顺序 = 训练顺序**:trot1 驱动关节 8 个,顺序为 `FL_thigh, FL_calf, FR_thigh, FR_calf, RL_thigh, RL_calf, RR_thigh, RR_calf`(见 `sim2sim/configs/quad_leg_trot1.py`)。SDK 读到的编号/顺序必须映射到这套顺序。
  - 📌 **桥里已认关节**:`damiao.py` 注释写明 `FL_thigh=canid1 … RR_calf=canid8`,两路 CAN-FD 各 4 个。`bennett_deploy` 的 `contract.py` 有 `JOINT_ORDER`/`JOINT_SPECS`。**核对 go2 契约的关节顺序 ≠ trot1 顺序**(镜像符号/排布),别照搬。
- [ ] **关节方向(正负号)镜像核对**:每个 thigh/calf 的转动方向、零位,与 URDF 一致。镜像腿(FR/RL)若方向不同,在映射表里翻转,别改策略。
- [ ] **CAN 类型核对(易坑)**:J8006 手册 = 标准 **CAN 2.0A / 1 Mbps / 11bit ID / 无 CAN-FD**;但桥的 USB 适配器与 `damiao.py` 默认 `canfd=True, brs=True`(CAN-FD 配置,5M 数据段)。**需确认桥实际下发的帧是标准 CAN**(FD 适配器向下兼容),否则 J8006 收不到。核对 `Motor_Control(...)` 及各脚本里的 `canfd` 开关。
- [ ] **编码器↔关节映射(首个真机测量)**:J8006 是 **6:1 减速 + 14bit 磁编(16384/圈,multi-turn 计数 "2")**,但桥两份文档对关节指令系数**自相矛盾**(`技术文档_1.md` 写 `q_motor = default + 6.0*q_joint`,`技术文档_2.md` 写 `scale=1.0`)。**上电后先发一个小目标角(如 +0.05rad),看回读 `q_real` 是 0.05 还是 0.30** → 决定 obs 的 `joint_pos` 到底是输出端 rad 还是电机轴 rad($q/6$),以及部署端要不要乘 6。**这个错了,后面全崩。**
- [ ] **IMU 坐标约定**:`projected_gravity` = 在**机体系**投影的重力方向;`base_ang_vel` = 机体系角速度。桥里 IMU 是 `DMImuSerialReader`(USB 串口 19 字节帧,reg2 陀螺 / reg3 欧拉角,CRC16 校验),已做符号/零偏 + 一阶低通(默认 12Hz)。**必须确认轴系与 Isaac base 系一致**(最常见的翻车点)。
- [ ] **默认站姿对齐**:trot1 默认站姿 thigh `±0.08`、calf `-0.16`。上电后应先把 8 个关节归到这套站姿,而不是各自零位。桥的 `runtime.py` 已有 `ramp_default → warmup → policy` 状态机(阶段A 直接复用)。
- [ ] **关节限位软件化**:thigh `(-0.80, 0.80)` rad、calf `(-0.90, 0.55)` rad。在部署代码里对目标值做 clamp,别等电机自己顶死。
- [ ] **SDK 能力确认**(逐项打勾):
  - [ ] 以 **50Hz/1kHz** 周期下发 MIT 目标角 ✅(`_hardware_run` 1kHz 主循环,策略按 `policy_rate_hz`)
  - [ ] 读回关节角 ✅(`Get_Position()`/`Get_Velocity()`)
  - [ ] 读回力矩 ✅(`Get_tau()`,标称 Nm)
  - [ ] 读/设内部增益 ✅(`read_motor_param`/`write_motor_param`/`change_motor_param`,DM_REG: Damp=11 / Inertia=12 / KT=1)
  - [ ] 读回错误码/心跳 ✅(`Get_err()` 4bit、`getTimeInterval()` 帧龄)
- [ ] **电机使能 + 零点/回零**:确认 Home/零点定义;`set_zero_position()` 发 `0xFE`。避免一上电跳。
- [ ] **基础安全垫**(参考 GO2 审计第100/184行,桥里已有 `_validate_feedback` / `_software_effort_limited_targets`):
  - [ ] 急停(物理 + 软件)
  - [ ] 限流 / 过流保护
  - [ ] 过温保护
  - [ ] 通信丢失 → 立即停并锁零(`feedback_age` 超时判断)
  - [ ] 目标值跳变限幅(slew-limit)
- [ ] **初始以"关节逐个、小步"测试**:手动发一个小目标角(比如 +0.05rad),确认方向、反馈、实际到位,再扩展。

---

## 阶段 2 · 系统辨识(未做 · 核心)

> 依据:数据手册 `bennett_rl/assets/motor/dm_j8006_2ec_v1_1_24v.yaml` 第66行明确列出需要辨识的项。本文把它展开成一张"做什么 / 为什么 / 怎么测 / 优先级"表。
>
> **原则**(网上实战经验):① 这是**回报最高**的一步,值得先投 2–3 天;② **只辨识几个高杠杆参数**;③ 理想**只用关节编码器**(PACE 做到无需力矩传感器);④ 激励要**动态**(chirp/摆/斜坡)。
>
> 💡 **采集工具已就绪**:桥里的 `async_csv.py` 已在每个策略周期写全量字段 —— `q_real / dq_real / tau_nm / q_cmd / feedback_age`,CSV 头见 `CSV_HEADER`。**系统辨识只需让它沿激励轨迹跑,把 CSV 导出即可**,无需自己再写采集。

### 2.1 辨识项清单

| # | 辨识项 | 为什么 | 怎么测 | 优先级 |
|---|---|---|---|---|
| 1 | **执行器有效刚度 kp / 阻尼 kv**(MIT 阻抗) | 策略发目标角,实际扭转 = `kp*(q_d−q)`。**MIT 的 kp/kd 是电机自配单位**,kp=28 未必 = 28 N·m/rad | 位置**阶跃响应**:发 target 阶跃,用 CSV 的 `q_real` 记录上升/超调/稳态误差 → 反推等效 kp/kd | 🔴 高 |
| 2 | **环路延迟 T_d**(指令→动作) | 相位错位是振荡/不稳常见根因 | 见**阶段3**;CSV 里的 `feedback_age` 可直读反馈延迟 | 🔴 高 |
| 3 | **粘性阻尼 + 库仑摩擦(Stribeck)** | 决定低速/换向误差与抖动;sim 默认 friction=0 | 恒速斜坡测电流-速度曲线;或正弦往返拟合 | 🔴 高 |
| 4 | **有效惯性**(转子+连杆) | 影响加速与谐振 | 几何/称重 + 摆衰减 + 阶跃下角加速度拟合 | 🟡 中 |
| 5 | **背驱(backdrive)q̃_b** | 外力能否反推关节;影响低增益柔顺 | 断电/低增益加外扭矩,测位移+速度 | 🟡 中 |
| 6 | **峰值/饱和力矩** | 已确认**额定8 / 峰值20**;sim ±8 正确,不要改 | 读手册 + 让电机带限输出看力矩包络;确认持续不超 8 | ✅ 已确认(不再是不一致) |
| 7 | **齿轮传动比 / 效率 / 齿隙** | 高减速比执行器模型不准;齿隙造成换向死区 | 读手册减速比 + 低速微幅正弦看死区 | 🟡 中 |
| 8 | **力矩→角度映射 & scale 一致性** | 防"发10°实际8°"比例误差 | 多位置小阶跃,target-vs-actual 线性拟合斜率 | 🟡 中 |

### 2.2 方法与流程

0. **第0步(最关键):定"编码器↔关节"映射 + 每电机 kp/kd 单位。**
   - 先发小阶跃,读 `q_real` → 判定 1:1 还是 6:1(见阶段1);这决定 obs 的 `joint_pos` 与部署目标角是否要乘 / 除 6。
   - 在同一阶跃里,由 `q_real` 的上升/超调/稳态误差,反推**当前 kp/kd 在关节端等效的刚度/阻尼**;换个 kp 再测,拟合出"MIT 数值 ↔ 关节端真实 N·m/rad"的标定曲线。**这一步产出的标定表,就是后面每次训练对齐 sim 增益的依据。**
1. **收集激励数据**:让真机沿**动态激励轨迹**运动 —— chirp(扫频)最佳(~20s),或组合摆/斜坡/往返。真机和仿真跑**同一条**激励指令,对比轨迹。
2. **直接实验定基础参数**:称重/几何定惯量;摆衰减定阻尼;阶跃定增益+延迟;外扭矩定背驱。
3. **优化拟合残余**:对比"真机实测轨迹" vs "sim2sim(MuJoCo) 复现轨迹",用 **CMA-ES**(零阶、适合非可微仿真)或 Nelder-Mead 最小化误差,反推 `[I_a, d, τ_f, q̃_b, T_d]`(PACE 最小参数集)。进阶:先 CMA-ES 粗拟合,再最大化 FIM 选激励、采新数据精修。
4. **更新仿真模型**:把辨识出的 kp/kd(按真实单位换算回 sim)、摩擦、惯量、力矩上限写回 `bennett_3.xml`(以及训练端执行器增益)。**forcrange 保持 ±8,不要改成 ±20。**
5. **重放验证**:相同激励做 sim vs 真机轨迹叠加;若真机出现更高频振荡 → 阻尼或延迟要调。

### 2.3 落点(写进哪)

- `sim2sim/models/bennett_3/bennett_3.xml`(自由基,行走) 与 `bennett_3_fixed.xml`(trace)
- 训练侧执行器增益:`quad_leg_trot1/flat_env_cfg.py:79-83`(kp28/kv2/effort8/saturation20)

---

## 阶段 3 · 真实延迟(未做 · 一步步来)

> 参考:MEVIUS 四足(50Hz 策略,CAN-USB 异步 150Hz 伺服)通过对比**目标 vs 实际角**时间偏移,估出延迟约 `0.02–0.06s`,并**把动作执行平移若干步**补偿,才稳。
>
> **我们照做,逐步。采集很方便**:桥的 `async_csv.py` 同步写入 `q_cmd`(你下发)与 `q_real`(回读),且 `feedback_age` 字段直读每电机反馈帧龄 —— 延迟测量基本等于"拿 CSV 做互相关"。

### 3.1 先定义"延迟"包含哪些
`推理时间(PC/Jetson)` + `指令下发(SDK/CAN 吞吐)` + `电机内部闭环响应` + `编码器回传` + `IMU 采样/滤波` + `数据封装`。分开测,别笼统一个数。

### 3.2 测量方法(同步记波形)
理论:系统延迟 = 输入序列 x 与输出序列 y 的**时间平移**(互相关最大处)。

1. **最直接**:在 50Hz 主循环里同步记录 `q_cmd` 与 `q_real` 两个时间序列(CSV 已有)。
2. 对真机做**阶跃/扫频**,画 `q_cmd vs q_real`;二者**时间偏移** = 有效环路延迟。`feedback_age` 还直接告诉你"下发→回读"的往返量。
3. **分解延迟**:
   - 下发延迟:发一条、读回,量总线往返;
   - 推理延迟:策略一次 forward 计时(PC 通常 ~1ms;Jetson 差异大,要重测);
   - 电机内部:目标下发后实际角多快跟上(阶跃响应)。
4. **已知现象作对照**:若 MuJoCo 里**不加延迟**的 sim2sim 出现周期性振荡的 target-vs-actual 误差 → 相位裕度不足;真机只会更糟,说明必须建模延迟。

### 3.3 怎么补偿
- **训练端注入延迟**(鲁棒化,首选):legged_lab 有 `ActionDelayCfg(enable=False, max_delay=5)`,打开重训(或保留原策略用部署端补偿)。这样策略天然抗延迟。
- **部署端平移补偿**:把延迟换算成 50Hz 步数,动作应用时**平移 N 步**(MEVIUS: 0.02–0.06s ≈ 1–3 步)。
- **两案都要先在 MuJoCo 里复现**:sim2sim 加同样延迟/平移,确认现象一致、策略仍稳,再上真机。

### 3.4 阶段A vs 阶段B 差异
- **PC 延迟**和 **Jetson 延迟**不同(Jetson 推理可能更快,但 I/O/总线/电源/散热加抖动)。**换板必须重测、重注入、重跑验证序列。**

---

## 阶段 4 · 部署脚手架(⚠️ 桥已存在 · 需对齐 trot1)

> **重大更新:部署脚手架已经建好了。** 在 `E:\HuanCun\Desktop\u2canfd\quad_leg_xu\bennett_deploy\`:
> - `runtime.py` → `DeploymentRunner._hardware_run()`(1kHz 主循环)+ 状态机(`ramp_default→warmup→policy`)
> - `policy.py` → `PolicyPipeline.infer()`:obs→raw→clamp→desired→rate-limit→`target=train_default+applied`
> - `policy.py` → `ObservationBuilder.build()`(构造 **50 维** obs)
> - `contract.py` → `DeploymentContract.load()`:读 `env.yaml/agent.yaml`,算出 `policy_rate_hz = 1/(sim_dt*decimation)`
> - `dm_can.py` → `Go2MotorBus`(USB-CANFD 双路,SN `D6977...`,`snapshot()` 原子读 / `command(q, kp, kd)` / disable / close)
> - `async_csv.py` → 高帧率 CSV(见阶段2/3)
> - `imu.py` → `DMImuSerialReader`(19 字节帧,CRC16,陀螺+欧拉角,一阶低通)
>
> **所以阶段A 的骨架是现成的。缺的只是"把它对齐到 trot1"这一层对比。** 不要照搬 go2 契约。

- [ ] **obs 重建(严格 50 维,顺序不可乱)** —— 以 `trot1` 的 `sim2sim.py:259-286` 为准,核对 `bennett_deploy` 的 `ObservationBuilder`:
  `base_ang_vel(3) + projected_gravity(3) + command(3) + joint_pos_rel(8) + joint_vel(8) + last_action(8) + phase_sin_cos(2) + leg_phase_sin_cos(8) + desired_contacts(4) + gait_params(3)`
  - `joint_pos_rel` = `(joint_pos − default)`,default = thigh `±0.08` / calf `-0.16`
  - `last_action` = 上一策略输出的**原始 8 维**(非缩放/限幅后的目标角 —— 之前确认过是 `env.action_manager.action` 原值)
  - gait 时钟/相位/触点/步态参数 = **软件自产**,按 `sim2sim.py` 同样的公式算
  - `projected_gravity` / `base_ang_vel` = IMU 读(务必匹配基系)
  - ⚠️ **go2 的 obs 顺序/维度可能与 trot1 有细微差异**(尤其 phase 的 2 维 base 相位 + 8 维腿相位),**逐项比对,别当相同**。
- [ ] **scaling**:`target_angle = clamp(raw_policy * 0.20, clip) + default_joint_pos`,clip = thigh `±0.80` / calf `(0.55, -0.90)`(注意 calf 下限 -0.90)。核对 contract 里 `joint_pos.scale`。
- [ ] **joint 映射**:8 维目标角按顺序映射到 SDK 电机 id;核对 `JOINT_ORDER` 与 trot1 一致(镜像腿符号)。**并套用阶段2 第0步标定的 6:1/1:1 系数** —— 三处(obs `joint_pos`、`target_angle`、写回 `q_cmd`)必须同用一个系数,别一处乘 6 一处不乘。
- [ ] **闭合腿(并联 pantograph)说明**:真机上的"闭环"是**物理四杆**(thigh→calf + thigh→1→2 在踝汇聚),**不需要** `<equality><connect>` —— 那只是仿真里把两连杆焊成闭环的手段。真机 8 个驱动关节就是 thigh/calf;`_1/_2` 被动杆,`_3` 脚端。obs 只含 8 个驱动关节,不含脚端。
- [ ] **50Hz 恒定循环**:`_hardware_run` 已是 1kHz 主循环 + 策略按 `policy_rate_hz`;_`_sleep`(L724-732)保证周期;统计 jitter。
- [ ] **推理**:阶段A `torch.jit.load(policy.pt).forward(obs)` 即可(有 torch);阶段B 才上 ONNX/TensorRT。
- [ ] **观测自检(离线)**:把真机一段日志 replay 到 sim2sim 或直接喂策略,看输出是否合理、obs 是否在训练分布内。**这一步能抓住 90% 的顺序/坐标错误。**
- [ ] **安全垫(部署里已部分内置)**:关节 clamp、软件力矩包络(`_software_effort_limited_targets`,顺带把指令压在 8N·m 内)、slew-limit、`_validate_feedback`、通信丢失停。

> ⭐ **先做"观测自检"再连电机。** 你 sim2sim 很稳是因为 obs→action 严丝合缝;真机坑多半在坐标/顺序/单位,离线自检能拦住。
>
> 🔒 **obs 已冻结(用户要求)。** 以上只做**离线校验**(把真机日志 replay 看是否在训练分布内),**不改 obs / 训练 / 对比逻辑**;任何改动都要能 git 回退。

---

## 阶段 5 · 真机测试序列(阶段A · 逐步解锁)

> 参考业内通用 + iit-DLSLab 部署 checklist。**严格按顺序,每步通过再进下一步;全部挂 harness + 急停在手上。** 桥的状态机已含 `ramp_default→warmup→policy`,上层再加这个序列。

1. [ ] **站立起身**:从倒地/下蹲起身到站姿,平滑、无抽搐。
2. [ ] **零指令静止**:`(vx,vy,wz)=(0,0,0)`,看是否自稳、无持续振荡(振荡→回阶段3 查延迟/阻尼)。
3. [ ] **慢速(≈0.2 m/s)**:看步态相序(触地/摆动)与接触时序是否和 sim 一致。
4. [ ] **全速(至 0.35 m/s)+ 转向 wz(−0.6,0.6)**:先直行再加转向。
5. [ ] **扰动**:轻推一下,看能否恢复平衡(DR 的 `push_robot` 就是为此)。
6. [ ] **地形(可选)**:沙/草/缓坡 —— 对应已训的 gravel/slope。
7. [ ] **量化记录**:轨迹平滑度(jerk)、接触力、摔倒恢复、多次方差 —— 验证"sim 与实际一致"的硬指标;不一致就回阶段2/3 调参,别硬走。

---

## 阶段 6 · Jetson 生产化(阶段B · 以后做)

> 桥已基本跨平台(审计确认)。**要改的 Windows 专属部分**:
> - Windows DLL 加载(`os.add_dll_directory`/`ctypes.CDLL`)→ arm64 的**厂商 `.so`(gitee 下载)**
> - `_WindowsTimerPeriod`(timeBeginPeriod)键盘输入 `msvcrt`(`KeyboardCommandVelocity.poll` 非 win 降级)
> - USB-CANFD 在 Linux 走 `libusb` 预加载(`quad_leg_go2-4.py:24-30` 已有补丁参考)
> - pyserial / pyusb / torch(TorchScript `map_location="cpu"`)均跨平台 ✓

- [ ] **策略导出**:`policy.pt` → **ONNX**(play.py 已支持),ONNX Runtime / TensorRT(FP16);量测推理耗时 < 20ms 留余量。
- [ ] **部署代码复用**:阶段4 的 obs/scaling/joint 映射**原样**搬(所以阶段4 必须硬件无关)。
- [ ] **重测延迟**:Jetson 上重做阶段3.2(推理/I/O/总线/电源差异)。
- [ ] **实时性**:实时线程、CPU 亲和、免抢占;必要时 RT 调度。
- [ ] **供电与热**:强力矩下供电压降、温度(热降频影响实时性)。
- [ ] **重跑阶段5 序列**:新平台 = 新延迟,可能需重新注入/微调。
- [ ] 可选:接 ROS2 driver 或 unitree-style SDK。

---

## 阶段 7 · 闭环迭代

- [ ] 每次真机测试**留下数值记录**(放 `outputs/` 或任务 README)。
- [ ] 若 sim 预测 ≠ 实测 → 记下来,回**阶段2(系统辨识)** 或 **阶段3(延迟)** 改参数,再回 **sim2sim** 复验,最后才改策略。
- [ ] **改过仿真(辨识值/延迟)后,先重跑 sim2sim 三任务**,确认无回归。
- [ ] 长远:更强地形/鲁棒上 **RMA / teacher-student 蒸馏** 或 **PACE 式系统辨识**;现阶段 trot1 的"删特权量 + DR"已够走平/缓坡。

---

## 附 · 需要你确认/实测的开放问题

- ✅ **[已测 2026-09-05] MIT kp ↔ 关节端静态刚度标定**(悬空阶跃,稳定段 τ/err;`u2canfd\sysid_battery.py`,输出在 `u2canfd\sysid_out\kp_*.png/.csv`):
  * FL_thigh:kp=20/28/40 → kp_static ≈ **21.8 / 30.7 / 50.0** N·m/rad
  * FR_thigh(更紧):kp=20/28/40 → ≈ **15.9 / 23.0 / 31.1** N·m/rad
  * **工作点 kp=28 → 关节端刚度 ≈ 23~31 N·m/rad,与 sim kp=28 基本对齐(差 <25%)** → 模拟/真机执行器刚度已大致一致,**无需改 kp**。
  * ⚠️ 可靠标定范围 = **kp 20~40**。>40 时 `err_st` 塌到 ~0.001 rad(伺服能到目标),静态 τ/err 落在编码器噪声底、比值乱跳(FR 图上 kp=60/100 变负);要测更高刚度需"外加已知力矩测偏转"的独立方法。
  * ⚠️ 以上全是**静态**刚度;**阻尼 kd 由构造决定** —— 部署设 kd=2、与 sim 的 kv=2 一致,伺服内部直接施加,**不用再单独测**(位置正弦回归根本收不回 kd:跟踪太好、err 落在噪声底;kp 会错测成 ~1.8)。**别用全程瞬态回归**,那会混入 kd*dq 把 kp 错测成 ~2.7。
- ✅ **[已测 2026-09-05] 编码器↔关节映射 = 1:1,且 8 关节符号全对**(`u2canfd\sysid_out\map_*.png`):
  * 8 电机各 +0.05 sim 小阶跃 → **Δsim 全部为正(0.66~0.98),无一为负** → `MOTOR_SPECS` 的 `sim_to_motor`(±1,仅符号)**对全部 8 关节都正确**(某关节符号错则该处比值为负)。
  * 比值 <1 是**重力/摩擦柔度**而非齿轮比:悬空、kp=28,伺服在 `kp·err=重力矩` 处平衡,稳态 `actual<cmd`。稳定段比值如 FR_thigh=0.923、RR_thigh=0.658(最松的一条)。
  * `Get_Position()` 即**关节(输出)端 rad**,obs `joint_pos` 直接用回读值,**部署系数不乘 6**。`技术文档_1` 的 ×6 **是错的**,弃用。
- ✅ **[已测 2026-09-05] 其余 sysid 电池**(`u2canfd\sysid_battery.py`,CSV+PNG 在 `u2canfd\sysid_out\`):
  * **alive**:按 `3_quad_leg_track.py` 的唤醒(refresh+纯阻尼)后 **8/8 电机全回传**;所有电机 `err=1` 即**正常码**(runner 只拒绝非 (0,1)),**非故障**。之前"4~8 无响应"是只读脚本没唤醒的假象。
  * **lin(FR_thigh)**:斜率 0.938,0.05~0.45 rad 无饱和,线性良好(0.06 偏移为均匀重力柔度)。
  * **damp**:定位正弦回归**无法恢复 kp**(伺服跟踪太好、err 落噪声底);库仑摩擦 fc≈0.09~0.15 N·m(粗略)。**kd 由部署值直接给定**。
  * **lat**:阶跃互相关环路延迟 FL=93ms / FR=129ms —— 这是**伺服闭环整定响应延迟**(上升+稳定),**不是**纯 CAN/策略传输延迟(50Hz 策略才 20ms/帧);**纯传输延已用反馈帧龄确认 ≈0.0~0.2ms**(alive 的 age 列),CAN 传输极快。
  * **encnoise(静止)**:FR/RR 大腿 pos std≈0.08~0.11 mrad、峰峰≈0.38 mrad(≈1 个 14bit LSB=2π/16384)→ **obs `joint_pos` 噪声底 ≈0.005°,几乎无噪**。两腿静止一致;松腿回差只在**运动/换向**显现(故 map/damp 松腿更飘,这里不飘)。
  * 🔧 **机械观察**:FL/RL/RR 的映射比值偏低(FL_thigh 0.702、RR_thigh 0.658),FR 干净(0.923),与你"其他腿有点松动"一致 → 松的腿回差/摩擦大,sysid 数值更飘。紧完丝建议在 FR/RR 复测。
- ✅ **[已核实 2026-09-05] MIT 指令打包**:t_ff 的 **12bit 打包在 `damiao.py control_mit` 内部完成**(`_float_to_uint(tau,−tau_max,tau_max,12)`,字节 526~527 拼帧);kp 12bit(0~500)、kd 12bit(0~5)、q 16bit、dq 12bit。**我们只传 float,从不手打字节;runner 能用 = 已正确**,12bit 量化分辨 ≈0.01 N·m 可忽略。(其余"回零/使能协议"仍靠 `控制协议 V1.4.pdf`,另留。)
- ✅ **[已核实 2026-09-05] 电机限位**:你出厂**未设限位**、URDF=0;`bennett_deploy/contract.py` 只是从 env.yaml 读 `velocity_limit/effort_limit` 作为**软件 clamp**。当前**已装配、不周转**;sysid 已在软件里 clamp 到关节物理限位,不会打限位。
- ✅ **[已测 2026-09-05] IMU(桥)—— 轴序校准权威闭环**:桥 `projected_gravity` = 把 IMU reg3 的 roll/pitch(±`--roll_offset_deg`/`--pitch_offset_deg`)经 `projected_gravity_from_rpy_deg` 换算(reg2 陀螺给 `base_ang_vel`)。**确系 COM6**(reg1=原始加速度计/reg2 陀螺/reg3 欧拉,1000Hz,CRC 0 错;COM7 别的)。三方面验证通过:
  * **硬件轴序(非循环)**:`imu_axis_cal.py` 后台录 90s,最后 3s 出现俯仰。物理绕**机体 y 轴**俯仰时,**独立信号——陀螺 `wy` 主导**(|wy|=0.445°/s,wx/wz≈±0.06 无交叉),且 reg3 **pitch** 随之变化(roll 只是小交叉到 ~4°)→ **IMU 的 pitch 轴 = 机体 y 轴,无 90° 转置**(若转置,物理 y 轴旋转会显示成 reg3 roll)。
  * **公式 ≡ sim obs(代码推导)**:桥公式 `(sin p,−sin r·cos p,−cos r·cos p)` 从标准 XYZ(roll-x/pitch-y,`base FORWARD=(1,0,0), GRAVITY_VEC_W=(0,0,-1)`)逆旋精确推出 = IsaacLab `projected_gravity_b = quat_apply_inverse(quat, (0,0,-1))`(见 `IsaacLab/.../rigid_object_data.py:343`)。→ **符号/轴序/约定与策略在 sim 里吃的 `projected_gravity` obs 完全一致,无需改**。数据点实测:peak pitch=+24.62° → gx=+0.4165=sin(24.62°)=+0.4165 逐位吻合,gz=−0.907=−cos(roll)cos(pitch);roll=−4.83° → gy=+0.0836 也逐位吻合。
  * **电平/单位**:水平 gz≈−1,|g|≈0.9998;静态 roll≈−0.96°、pitch≈−0.46°(可归零到 `--roll_offset_deg/--pitch_offset_deg`);yaw≈−157.6° 因公式忽略 yaw 而不影响;reg1 加速度计 |a|=9.81 m/s²(独立对照源,需要时可再交叉)。
  * **`base_ang_vel` 单位已核实(非循环,reg2=d(reg3)/dt)**:爬升段 `wy` / `dPitch/dt` 比率 ≈0.0145~0.0173,聚在 **0.01745 = 1/57.30**(低通 12Hz 恒略低于此,低通只衰减不放大)→ **reg2 陀螺直接输出 rad/s**。因此桥默认 `--gyro_unit rad_s`(不换算)= **正确**,`base_ang_vel` 单位与 sim 一致,**无 ×57.3 错**。(若某天 raw reg2 给 deg/s,才需该参数换成 deg_s。)
  * 工具:`u2canfd\imu_axis_cal.py`(录 reg2/3→CSV)、`probe_imu.py`(找口)、`analyze_tail.py/analyze_resid.py/analyze_gyro_unit.py`(离线分析)。
- ⚠️ **[2026-09-05 发现] 两处 `sim_to_motor` 符号表冲突**:`3_quad_leg_track.py` MOTOR_SPECS **FL_thigh=−1、RL_thigh=−1**;`bennett_deploy/contract.py` JOINT_SPECS **FL_thigh=+1、RL_thigh=+1**,其余 6 关节一致。我的电池**采用 3_quad_leg_track.py**(=你正在跑的策略)。**两者只能有一个匹配 trot1/quad_leg_track 的 sim 轴定义** → 跑 trot1 前务必确认用哪套,别混用(obs `joint_pos`/写回 `q_cmd`/target 三处一致)。
- ⚠️ **[2026-09-05 已逐项比对] trot1 contract vs go2** —— **结论:trot1 有两处硬校验过不去,不能直接用 go2 contract**。详见 **`TROT1_CONTRACT_CHECKLIST.md`**。要点:
  * ① **观测项命名**:go2 期望 `crawl_phase/crawl_leg_phase`,trot1 是 **`trot_phase/trot_leg_phase`** → `observation_order` 不匹配即失败。
  * ② **gait 参数形状**:contract 用 `_consistent_under(...,"frequency_hz"/"duty_factor")` 读**扁平标量**;trot1 是速度相关的 `min/max_frequency_hz` + `low/high_speed_duty_factor`(范围)→ **无扁平 `.frequency_hz`/`.duty_factor`,直接抛错**。
  * ③ **obs 构建器**:`policy.py::ObservationBuilder` 写死调 `crawl_phase_terms`、把 freq/duty 当标量传入;trot1 的 gait 是 `commanded_trot_*`(按 `velocity_command` 实时解算)→ 得换成 trot1 的 gait 函数并**按当前速度解算**该帧 freq/duty。
  * 其余(8 关节名/`use_default_offset`/`preserve_order`/`offset=0`/`JointPositionAction`/`joint_pos_rel`/`scale=0.2`/`dt.005·decimation4=50Hz`/`stiffness28·damping2`/`effort8·saturation20`/`velocity20`)**全部通过**。
  * ⚠️ **符号表**:trot1/quad_leg_track 都用 `Urdf_Bennett_3.usd` 同名同轴向,但 `contract.py` `FL/RL_thigh=+1` vs `3_quad_leg_track` `-1`(实机已验证 -1 全对)→ **跑 trot1 前必须把符号统一到实机验证过的 -1,并三处(obs `joint_pos`/写回 `q_cmd`/target)用同一套**。

---

## 参考来源
- PACE(ETH 系统辨识):`github.com/leggedrobotics/pace-sim2real`(chirp / CMA-ES / FIM / `[I_a,d,τ_f,q̃_b,T_d]`)
- legged_gym / rsl_rl(ETH RSL,ANYmal 真机验证):`github.com/leggedrobotics/legged_gym`
- MEVIUS 四足(目标 vs 实际角测延迟 + 步数平移):arXiv 2409.14721
- 你的桥:`E:\HuanCun\Desktop\u2canfd\quad_leg_xu\bennett_deploy\`(runtime/policy/contract/dm_can/async_csv/imu)
- 手册:`bennett_rl/assets/motor/dm_j8006_2ec_v1_1_24v.yaml`(第66行系统辨识硬性要求;额定8/峰值20)
