# scratch/ — 归档的草稿(2026-09-15 从 outputs/ 迁入)

这轮 Fig.5 工作空间动画调参时的临时文件,按用户要求归档保留,不参与正式流程。

| 文件 | 是什么 |
|---|---|
| `_loop_scan.py` / `_loop_scan2.py` | 底部闭环参数扫描(A1/A2/q2c/ψ 网格),结论已固化进 `../code/plot_fig5_paper_anim.py` 的默认值(A1=0.30, A2=0.35, q2c=0.0) |
| `_extract_smoke.py` / `_extract_final.py` / `_extract_smoke3.py` | 用 PIL 从 GIF 抽帧检查的小工具 |
| `_smk/` `_smk2/` `_anim_frames/` `_ax_f*.png` `smoke3_f*.png` | 各轮抽出来的检查帧 |

注意:两个 `_loop_scan*.py` 里的 `sys.path` 还指向老位置
`scripts/analysis`(bennett_leg_fk 已搬到 `../code/`),要重跑需把路径改成
`../code`。
