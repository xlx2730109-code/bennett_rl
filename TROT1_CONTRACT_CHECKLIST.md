# TROT1 契约比对清单(trot1 env.yaml vs go2 contract)

> 目的:判断 `bennett_deploy/contract.py`(go2-10 硬编码)能否直接跑 `quad_leg_trot1` 策略;
> 不能的话,具体要改哪些。数据来源全部为实读两份文件,非推测。

## 0. 先分清三套东西(别混)

| 文件 | 是什么 | obs 维数 | FL/RL_thigh 符号 |
|---|---|---|---|
| `3_quad_leg_track.py` | 独立自包含 runner,**不走 contract.py** | 36(phase2+ref8+err8+pos8+vel8+last2) | **-1**(MOTOR_SPECS) |
| `bennett_deploy/contract.py` | 现代桥 go2 专用契约(`runtime.py` 用) | 50 | **+1**(JOINT_SPECS) |
| `quad_leg_trot1` 策略 | gait-trot 训练产物 | 50 | (obs 用 _rel;动作 JointPositionAction) |

`3_quad_leg_track.py` 能用 = 它**绕过了** go2 校验(自带 `load_train_alignment` 只读 dt/decimation/stiffness/scale),
自己手拼 36 维 obs、用 `leg_signs {FL:+1,FR:-1,RL:-1,RR:+1}`(小腿按大腿取反?否——是 base 项直接乘符号)。
它跑的是 **quad_leg_track(2 维 base 动作)**,不是 trot gait 策略。

## 1. contract.py 对 trot1 的逐条硬检查结果

来源:trot1 `2026-07-27_20-08-08\params\env.yaml` vs `contract.py::DeploymentContract.load`

| contract.py 硬校验(值) | trot1 实际值 | 结果 |
|---|---|---|
| `actions.joint_pos.joint_names == JOINT_ORDER` | FL/FR/RL/RR × thigh/calf 同序 | ✅ 通过 |
| `use_default_offset == true` | `true` | ✅ 通过 |
| `preserve_order == true` | `true` | ✅ 通过 |
| `offset == 0` | `0.0` | ✅ 通过 |
| `class_type == isaaclab...JointPositionAction` | 恰为此类 | ✅ 通过 |
| `observations.policy.joint_pos.func == joint_pos_rel` | 是 | ✅ 通过 |
| `observations.policy.joint_vel.func == joint_vel_rel` | 是 | ✅ 通过 |
| obs `joint_names{preserve_order}` 与 JOINT_ORDER 一致 | 是 | ✅ 通过 |
| **观测项名字 == go2** `(base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions, **crawl_phase, crawl_leg_phase**, desired_contacts, gait_params)` | 前 6 项一致,但 gait 三项是 **`trot_phase, trot_leg_phase`** | ❌ **失败**(名字不匹配) |
| `_consistent_under(env,"observations.policy.","frequency_hz")` | trot 用的是 `min_frequency_hz`/`max_frequency_hz`(范围),**无扁平 `.frequency_hz`** → 空列表 | ❌ **失败**(抛 ValueError) |
| `_consistent_under(...,"duty_factor")` | 用 `low_speed_duty_factor`/`high_speed_duty_factor` → 无扁平 `.duty_factor` | ❌ **失败**(抛 ValueError) |
| `_consistent_under(...,"swing_height")` | 0.045(×4 个 term 一致) | ✅ 通过 |
| `_consistent_under(...,"command_deadband")` | 0.025(×4 一致) | ✅ 通过 |
| `actions.joint_pos.scale=0.2` | 0.2 | ✅(action_scale_rad=0.2) |
| `sim.dt=0.005, decimation=4` | 一致 | ✅ policy_rate_hz=50Hz |
| `stiffness=28, damping=2` | 一致 | ✅ 对应 MIT kp/kd=28/2 |
| `effort_limit=8.0, saturation_effort=20.0` | 一致 | ✅ 与"±8 不改 ±20"约束吻合 |
| `velocity_limit=20.0` | 一致 | ✅ |

**结论:trot1 有两处硬性过不去 —— ① 观测项命名(`crawl_*` vs `trot_*`),② gait 参数形状(扁平 `frequency_hz/duty_factor` vs `min/max_frequency_hz` + `low/high_speed_duty_factor`)。**

## 2. 更深一层:obs 构建也过不去

`bennett_deploy/policy.py::ObservationBuilder.build` 目前**写死**调用 `from .gait import crawl_phase_terms`,且把
`frequency_hz / duty_factor / swing_height / command_deadband` 当作**标量**传入。

trot1 的 gait 是"速度相关的参相 treadmill"型:`commanded_trot_*` 在 `bennett_rl...quad_leg_trot.mdp.gait`,
按 `min/max_frequency_hz`、`low/high_speed_duty_factor`、`min/max_equivalent_speed`、`yaw_equivalent_radius` 生成。
**合同里单值 `frequency_hz/duty_factor` 无法表达"随速度变化"**,所以即使改完 obs 名字,ObsBuilder 也得换成
`commanded_trot_*`(并按当前 velocity_command 算出该帧的 freq/duty),不能再用 `crawl_phase_terms`。

## 3. ⚠️ 符号冲突(最要命,和 trot1/go2 无关,但部署必踩)

- `contract.py` `JOINT_SPECS`:**FL_thigh=+1, RL_thigh=+1**(其余 6 关节一致)。
- `3_quad_leg_track.py` `MOTOR_SPECS`:**FL_thigh=-1, RL_thigh=-1**。
- 两者都映射**同一台物理机**(m 都用 `Urdf_Bennett_3.usd`,同名关节同轴向)。
- **本会话系统辨识已实机验证**:按 `3_quad_leg_track` 的符号表,8 个关节全部正响应(无一个为负)。
- => `contract.py` 的 `FL_thigh/RL_thigh=+1` **相对实机是反的**,若直接按它部署,会把这俩关节往**反方向**打。

**行动项(部署前必须定,别混用三处):**
1. 确认 trot1 在 sim 里 FL_thigh/RL_thigh 的**正方向**对应物理哪个方向(看 `URDF` 关节轴向 + `init_state FL_thigh=+0.08`)。
2. 以**实机验证过**的 `3_quad_leg_track` 符号表为准,把 contract.py 的 `FL_thigh/RL_thigh` 改回 **-1**;
   或反过来,统一到 contract.py 的 **+1** 并**重新在实机跑一遍 map 验证**(不建议,会推翻已验证结果)。
3. obs 侧 `joint_pos_rel`/`q_cmd`/`target` 三处必须用同一份符号,不能一半 +1 一半 -1。

## 4. 要跑 trot1,建议的最小改动量

1. **contract.py** 新增 trot1 分支(或复制成 `trot1_contract.py`):
   - 观测名映射:`crawl_phase/crawl_leg_phase` → `trot_phase/trot_leg_phase`(其余 6 项不变)。
   - gait 标量抽取改为**允许范围**:读 `min/max_frequency_hz`、`low/high_speed_duty_factor`、`swing_height`、`command_deadband`。
2. **policy.py / gait.py** ObsBuilder 换成 trot1 的 `commanded_trot_*`,gait 项按 `velocity_command` 实时解算。
3. **符号表**按第 3 节统一。
4. 复用现有全部 `sysid_battery.py` 结论(kp=28 对齐、编码器 1:1、obs 无噪、CAN 传输 0.2ms)。

## 5. 附:默认策略与参数

- `3_quad_leg_track.py` 默认 `DEFAULT_POLICY` = `quad_leg_track\2026-06-29_15-16-03\exported\policy.pt`
  (该 run env.yaml:sim.dt≈1/300、decimation=6、scale=0.349rad=20°、max_joint_speed=1.047rad/s = 60°/s)。
- trot1 默认参数:`scale=0.2rad(0.2π≈36°...)`、`stiffness=28 / damping=2 / effort_limit=8 / saturation_effort=20 / velocity_limit=20`。

## 6. 已落地(2026-09-05)—— trot1 专属契约本体 + 步态时钟

已按本节最小改动写并**离线验证通过**:
- **`bennett_deploy/trot1_contract.py`** —— `TrotDeploymentContract.load()` 对真 trot1 env.yaml 逐条硬校验,**全部通过**,产出 `id=c0ee8699`:**(2026-09-05 实机推翻早前 -1 结论:FL/RL_thigh 用 +1)
  - 观测顺序(50 维)= `base_ang_vel(3)+projected_gravity(3)+velocity_commands(3)+joint_pos(8)+joint_vel(8)+actions(8)+trot_phase(2)+trot_leg_phase(8)+desired_contacts(4)+gait_params(3)`。
  - 携带**速度条件化步态配置** `TrotGaitConfig`:deadband=0.025、freq=[0.75,1.35]Hz、speed=[0.08,0.35]m/s、duty=[0.54,0.62]、swing=0.045、yaw_r=0.2。
  - **符号表统一为 Bennett 实测 FL/RL_thigh=−1**(非 go2 的 +1);train_default(thigh ±0.08、calf −0.16)、action_scale=0.2、clip=3.5(agent.yaml)、执行器 28/2/8/20/20、policy_rate=50Hz。
- **`bennett_deploy/trot1_gait.py`** —— `TrotGaitClock`,标量复现 sim 的**状态积分相位时钟**(非 `elapsed_s` 纯函数),且 `leg_phase` 用 **trot1 分组布局**(sin×4 在前 cos×4 在后,非 go2 逐腿交错),对角偏移 `(0,0.5,0.5,0)`。
  - 实测 @cmd=(0.20,0,0):freq=1.017Hz、duty=0.584、对角接触 FL=1/FR=0/RL=0/RR=1、静止 phase=(0,1) → 与 sim 逐位一致。
- **已验证结论**:三处 sign **已在实机上定为 FL/RL_thigh=+1**(2026-09-05,取 go2/contract.py;`3_quad_leg_track.py` 的 −1 会把 trot1 步态镜像);obs 命名/形状与 sim 逐位对齐;这些对换训练(含改无约束步态)仍复用,**只有 `trot1_gait.py` + 那四个 gait 观测项需要换**。

### ⚠️ 已知残留(非阻塞)
env.yaml 动作 clip 是**按关节不对称**的:`.*_thigh ∈ [-0.8,0.8]`、`.*_calf ∈ [-0.9,0.55]`。部署端只用标量 `action_clip=3.5`(×0.2=±0.7 rad)。calf **上界 +0.55** 会在 policy 输出饱和到 +3.5 raw 时被 sim 更早截断(0.7>0.55),造成 sim/deploy 微弱不一致。极少数饱和情形才触发;后续若要完全一致,可在部署侧镜像逐关节 clip(留着做,非本次)。

## 7. 未做(刻意延迟,属当前 trot 专用、最可能被后续换步态砍掉的部分)
- `runtime.py` 接线(`--policy-kind trot1` 选择 `TrotPolicyPipeline`)+ `trot1_policy.py` 完整管线(重建 obs → 解码 action → MIT-PD)。
- 待未来步态方向(如无约束 RL)定了再补;届时只需换 `trot1_gait.py` + 删四个 gait 观测项,obs 回到 ~33 维。
