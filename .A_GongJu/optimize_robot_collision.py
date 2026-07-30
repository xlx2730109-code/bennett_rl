"""
优化 Bennett 机器人 USD 的碰撞网格。

对 USD 中所有碰撞 prim 施加 PhysX collision approximation，
将高精度 STL 三角面碰撞体简化为 low-poly 近似，
大幅减少 PhysX 碰撞检测计算量。

用法：
  python scripts/tools/optimize_robot_collision.py

需要 Isaac Sim 环境（因为依赖 pxr / omni）。
如果直接运行报模块找不到，就用 Isaac Sim 的 Python 启动：
  D:\Conda\envs\env_isaaclab\python.exe scripts/tools/optimize_robot_collision.py
"""
# python scripts/tools/optimize_robot_collision.py


import argparse
import os
import sys

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="优化机器人 USD 碰撞网格")
parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="输入 USD 路径（默认：Bennett V3 USD）",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="输出 USD 路径（默认覆盖输入文件）",
)
parser.add_argument(
    "--approximation",
    type=str,
    default="convexHull",
    choices=["convexDecomposition", "convexHull", "meshSimplification",
             "boundingCube", "boundingSphere", "triangleMesh", "sdf", "none"],
    help="碰撞近似方法 (default: convexHull — 动态body只能用这个)",
)
parser.add_argument(
    "--voxel-resolution",
    type=int,
    default=None,
    help="凸分解体素分辨率 (仅 convexDecomposition，默认 50000)",
)
# 添加 Isaac Sim 启动参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim (headless)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""启动完成，下面导入 pxr 相关模块"""

from pxr import Usd, UsdGeom, UsdPhysics

# 默认路径：Bennett V4 USD（已备份，可以修改）
# 自动向上查找项目根目录（含 .git 或 source/bennett_rl 的目录）
def _find_repo_root(start: str) -> str:
    d = os.path.dirname(os.path.abspath(start))
    for _ in range(10):  # 最多向上找10层
        if os.path.isdir(os.path.join(d, ".git")) or os.path.isdir(os.path.join(d, "source/bennett_rl")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(os.path.dirname(os.path.abspath(start)))  # fallback

_REPO_ROOT = _find_repo_root(__file__)
BENNETT_DEFAULT_USD = os.path.join(
    _REPO_ROOT,
    "source/bennett_rl/bennett_rl/assets/robots/Urdf_Bennett_4/urdf/Urdf_Bennett_3.usd",
)

input_path = args_cli.input or BENNETT_DEFAULT_USD
output_path = args_cli.output or input_path

if not os.path.exists(input_path):
    print(f"[ERROR] 输入文件不存在: {input_path}")
    sys.exit(1)

print(f"[INFO] 打开 USD: {input_path}")
stage = Usd.Stage.Open(input_path)
if not stage:
    print(f"[ERROR] 无法打开 USD 文件")
    sys.exit(1)

# 统计
total_prims = 0
collision_modified = 0

print(f"[INFO] 碰撞近似方法: {args_cli.approximation}")

# 先遍历一次，找出所有碰撞体
# URDF导入器生成的USD中，碰撞mesh是Mesh prim，被Xform包裹，
# 但没有显式的CollisionAPI。我们找所有Mesh prim来施加属性。
mesh_prims = []
for prim in stage.Traverse():
    total_prims += 1
    if prim.IsA(UsdGeom.Mesh):
        mesh_prims.append(prim)

print(f"[INFO] 总 prims: {total_prims}，其中 Mesh prims: {len(mesh_prims)}")

for prim in mesh_prims:
    prim_path = str(prim.GetPath())

    # 施加 API 并获取对象；Apply() 返回 API 对象本身
    # 施加 MeshCollisionAPI 并设置碰撞近似
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.GetApproximationAttr().Set(args_cli.approximation)

    collision_modified += 1
    if collision_modified <= 5:  # 只打印前5个
        print(f"  [OK] {prim_path} → {args_cli.approximation}")

print(f"\n[INFO] 总 prims: {total_prims}，其中 Mesh prims: {len(mesh_prims)}")
print(f"[INFO] 修改碰撞近似: {collision_modified} 个 Mesh prims")

if collision_modified == 0:
    print("[WARN] 没有找到任何 Mesh prim!")
else:
    # 保存
    stage.GetRootLayer().Save()
    print(f"\n[INFO] 已保存到: {output_path}")
    print(f"[INFO] 下次训练 PhysX 将自动使用简化碰撞网格！")

# 关闭
simulation_app.close()
