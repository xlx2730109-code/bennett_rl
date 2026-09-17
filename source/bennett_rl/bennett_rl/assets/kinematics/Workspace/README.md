# Workspace — Bennett 腿工作空间分析(Fig.5 / Fig.6 一族)

fig5_workspace_natural 同款布局的工作空间图 + 动画的**代码、产物、缓存、草稿**都在这里。
原始参照图的内容从未改动(只做了归位搬运)。

## 目录结构

```
Workspace/
├── README.md          ← 本文件
├── code/              ← 全部脚本(封闭的互相 import 族,整族搬入,可直接运行)
│   ├── bennett_leg_fk.py            腿部 FK/被动角求解(几何常量取自 bennett_1.xml)
│   ├── plot_bennett_workspace.py    基础模块:配色/三视图/扫描缓存(WS_DIR 在这里定义)
│   ├── plot_fig5_paper.py           论文 Fig.5 解析曲面静态图(生成 natural 原图的脚本,勿改)
│   ├── plot_fig5_natural.py         natural 版工作空间(原图的生成脚本,勿改)
│   ├── plot_fig5_paper_anim.py      ★ 动画主脚本(GIF 交付物)
│   ├── plot_fig6_manipulability.py  Fig.6 可操作性
│   └── bennett_leg_gui.py           交互式腿部查看器
├── animation/         ← GIF 成品
│   ├── fig5_workspace_real_anim.gif       ★ 当前动画(可达曲面 + p2 底部闭环)
│   ├── fig5_workspace_real_anim_foot.gif  ★ 同款,跟踪真足端(--marker foot)
│   └── fig5_workspace_natural_anim.gif    旧版动画(原件,内容未动)
├── figures/           ← 静态图(含 natural 原件,内容未动)
│   ├── fig5_workspace_natural.png/.svg   ★ 参照原件
│   ├── fig5_workspace.png/.svg, fig5_workspace_full.*   早期版本
│   ├── fig5b_workspace_limited.*      硬件限位版
│   └── fig6_manipulability.png/.svg
├── scan/              ← 扫描缓存(workspace_*.npz,删了会触发重扫,几分钟)
└── scratch/           ← 归档草稿(见其中 README)
```

## 运行(仓库根目录)

```powershell
conda activate env_isaaclab
python source/bennett_rl/bennett_rl/assets/kinematics/Workspace/code/plot_fig5_paper_anim.py
```

产物自动写入 `Workspace/animation/`。快速预览加 `--frames 24 --path 96 --dpi 80 --gif _preview.gif`。

## plot_fig5_paper_anim.py 调参速查(行号为 2026-09-15 有限回转版)

| 想改什么 | 位置 | 现值 |
|---|---|---|
| 连杆粗细 | `LINKS` 表第 4 列(L106-110) | 2.0 / 2.0 / 2.0 / 2.0 |
| 关节点大小 | `JOINTS`(L112-115):hip 36,p1/p3 **4**,"m" 42 | p1/p3 = 直径²,黑色同连杆 |
| 红点大小/颜色 | `JOINTS` 末元组 `("m", 42, RED)`,跟 `--marker` | RED="#d03b3b" |
| 轴线颜色 | `AXIS_C = INK`(L99) | 黑 |
| 轴线长度 | `--axis_half`(L268,使用处 ~L565) | ±55 mm |
| 轴线样式 | ~L503-507,`ls=(0,(6,3,1,3))`,lw 0.9 | 点画线,z1 同粗 |
| 轨迹模式 | `--traj local`(默认)/`circle`(L270) | local = 底部闭环 |
| 圈形状 | `--q1_amp 0.30` `--q2_amp 0.35` `--q2_c 0.0`(L258-263) | ~200×170 mm |
| 跟踪点 | `--marker p2`(默认)/`foot`(L285) | foot = _1 上真足端,低 ~50 mm |
| 曲面范围 | 默认只回转 q1∈[−0.8,+0.8](大腿 XML 限位) | `--full_revolve`(L288)恢复全 360° 理论碗 |
| 曲面来源 | `--surface real`(默认)/`paper`(L282) | real = 生成线绕 z1 有限回转 |
| 清晰度 | `--dpi 200`(L277) | 2080×1260,GIF 帧延迟是厘秒,**fps>100 无意义**(L274) |
| 曲面网格 | `--n_rot 240`(L292) | 240 |

## 可达曲面(2026-09-15 用户反馈定稿)

全 360° 回转碗在侧视图上方有大片**足端到不了的区域**(用户红线标注,"很严重")。
现默认只画**可达区**:生成线(标记点 q2∈[−0.9,+0.55] 的真实轨迹,q1=0 体坐标)
绕 z1 只回转 q1∈[−0.8,+0.8] rad —— 即大腿关节 XML 限位内的实际摆动范围。
底部闭环的 q1∈[−41°,+10°] ⊂ [−46°,+46°],严格落在补丁上(实测贴面 ≤2.1 mm)。
`--full_revolve` 可看理论碗(诊断用,不用于交付)。

## 约定(用户指令)

- **不画 _3 杆**(闭链多余杆);_2 直接闭合到 calf 端。
- **natural 原图(figures/ 里两个 + animation/ 里旧 GIF)内容一律不改**;重跑
  `plot_fig5_paper.py` / `plot_fig5_natural.py` 会写到 assets/kinematics 根目录,
  不会覆盖这里的内容,但会产生副本——跑完记得删或手动归位。
- `bennett_kinematics.py`、`kin_style.py`、`plot_0*.py`、`generated/` 是上一代
  kinematics 分析(且 bennett_kinematics 被 `__init__.py` 运行时引用),不在本族内,未搬动。
