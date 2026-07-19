


import mujoco
import os

# URDF 路径（改成你自己的）
urdf_path = r"source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/AnLi/unitree_rl_gym-main/resources/robots/Bennett_test2/meshes/Bennett_test3.urdf"

# 输出 MJCF 路径（同目录）
mjcf_path = urdf_path.replace(".urdf", ".xml")

# 加载 URDF 模型
m = mujoco.MjModel.from_xml_path(urdf_path)
print(f"模型加载成功: {m.nbody} 个物体, {m.njnt} 个关节")

# 另存为 MJCF
# mj_saveLastXML 把编译后的模型以 MJCF 格式写回文件
try:
    mujoco.mj_saveLastXML(mjcf_path, m)
    print(f"MJCF 文件已保存到: {mjcf_path}")
    print("现在在 MuJoCo GUI 中打开这个 .xml 文件即可使用 Save XML")
except Exception as e:
    print(f"保存失败: {e}")
    print("尝试手动生成 MJCF 包裹文件...")
    mjcf_content = f"""<mujoco>
  <compiler meshdir="." />
  <include file="{os.path.basename(urdf_path)}" />
</mujoco>"""
    with open(mjcf_path, "w") as f:
        f.write(mjcf_content)
    print(f"MJCF 包裹文件已保存到: {mjcf_path}")
