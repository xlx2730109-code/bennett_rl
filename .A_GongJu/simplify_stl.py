# STL 面数简化工具
# 基于 trimesh + 二次边折叠算法
#
# 参数说明:
#   MAX_FACES  面数红线，超过就简化    180000（留余量）
#   face_count 目标面数                 int(MAX_FACES * 0.9)
#   aggression 速度/质量权衡(0-10)      0=质量最高

import trimesh
import os

# ============================================
#  配置
# ============================================
MODE = 2  # 1=单文件 | 2=遍历文件夹

# 单文件路径（MODE=1 时使用）
STL_FILE = r"D:/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/AnLi/unitree_rl_gym-main/resources/robots/Bennett_test2/meshes/base1.STL"

# 文件夹路径（MODE=2 时使用）
STL_DIR = r"E:/HuanCun/Desktop/ls/31/Bennett_test2/meshes"

# MuJoCo 允许的最大面数
MAX_FACES = 180000
# ============================================


def simplify_one(stl_path):
    """简化单个 STL 文件。"""
    if not os.path.exists(stl_path):
        print(f"文件不存在: {stl_path}")
        return

    mesh = trimesh.load(stl_path)
    current_faces = len(mesh.faces)
    print(f"/n{stl_path}")
    print(f"  当前面数: {current_faces}")

    if current_faces <= MAX_FACES:
        print(f"  面数在限制 {MAX_FACES} 以内，无需简化")
        return

    target = int(MAX_FACES * 0.9)
    print(f"  超过 {MAX_FACES}，简化到 {target} 面...")

    bak = stl_path + ".bak1"
    os.rename(stl_path, bak)
    print(f"  备份: {bak}")

    simplified = mesh.simplify_quadric_decimation(face_count=target)
    result_faces = len(simplified.faces)
    print(f"  简化完成: {result_faces} 面")

    simplified.export(stl_path)
    print(f"  保存: {stl_path}")


def simplify_folder(folder):
    """遍历文件夹，简化所有 .STL 文件。"""
    if not os.path.exists(folder):
        print(f"文件夹不存在: {folder}")
        return

    files = [f for f in os.listdir(folder) if f.upper().endswith('.STL')]
    if not files:
        print(f"未找到 STL 文件: {folder}")
        return

    print(f"找到 {len(files)} 个 STL 文件:")
    for f in files:
        print(f"  {f}")

    for f in files:
        simplify_one(os.path.join(folder, f))


if __name__ == "__main__":
    if MODE == 1:
        simplify_one(STL_FILE)
    elif MODE == 2:
        simplify_folder(STL_DIR)
    else:
        print(f"未知模式: {MODE}，请设置 MODE = 1 或 2")
