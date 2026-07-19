
"""URDF →  USD 自动转换脚本

用法：
  ./isaaclab.sh -p A_GongJu/urdf_to_clean_usd.py
  ./isaaclab.sh -p A_GongJu/urdf_to_clean_usd.py --urdf robot.urdf
  ./isaaclab.sh -p A_GongJu/urdf_to_clean_usd.py --usd-only robot.usd
"""

import argparse
from isaaclab.app import AppLauncher

DEFAULT_URDF_PATH = r"E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\assets\robots\Urdf_Bennett_4\urdf\Urdf_Bennett_3.urdf"
# DEFAULT_URDF_PATH = "E:/HuanCun/Desktop/gongsi/SIZU_Urdf_QianXiang1/urdf/SIZU_Urdf_QianXiang.urdf"


# 碰撞体模式: 1=convexHull | 2=convexDecomposition(推荐) | 3=none | 4=meshSimplification
COLLISION_MODE = 1

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--usd-only", type=str, default=None)
parser.add_argument("--fix-base", action="store_true", default=False)
parser.add_argument("--merge-fixed-joints", action="store_true", default=False)
parser.add_argument("--collider", type=str, default="convex_hull",
                    choices=["convex_hull", "convex_decomposition"])
parser.add_argument("--stiffness", type=float, default=100.0)
parser.add_argument("--damping", type=float, default=10.0)
parser.add_argument("--drive-type", type=str, default="force",
                    choices=["force", "acceleration"])

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, shutil, contextlib
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import carb
import omni.kit.app
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.utils import make_uninstanceable


# ============================================
#  辅助
# ============================================

def _find_robot_root(stage):
    p = stage.GetDefaultPrim()
    if p and p.IsValid():
        return p
    for p in stage.Traverse():
        if p.GetTypeName() == "Xform":
            if any(c.GetTypeName() in ("RevoluteJoint", "PrismaticJoint") for c in p.GetChildren()):
                return p
    for p in stage.GetPseudoRoot().GetChildren():
        if p.GetTypeName() == "Xform":
            return p
    return None


def _is_link(prim):
    if prim.GetTypeName() != "Xform":
        return False
    for c in prim.GetChildren():
        if c.GetName() in ("visuals", "collisions"):
            return True
    return False


def _apply_collision_api(prim):
    if not prim.IsValid() or prim.GetTypeName() != "Mesh":
        return
    try:
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        approx = {1: "convexHull", 2: "convexDecomposition", 3: "none", 4: "meshSimplification"}.get(COLLISION_MODE, "convexHull")
        mesh_api.CreateApproximationAttr(approx)
    except Exception as e:
        print(f"    ✗ CollisionAPI error: {e}")


def _flatten_usd(stage, usd_path):
    base, ext = os.path.splitext(usd_path)
    flat_path = f"{base}_flattened{ext}"
    stage.Export(flat_path)
    return flat_path, Usd.Stage.Open(flat_path)


# ============================================
#  复制 mesh → 删旧文件夹 → 设属性
# ============================================

# ============================================
#  主流程
# ============================================

def clean_usd_file(usd_path):
    print(f"  Clean USD...", end=" ", flush=True)
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("FAIL: cannot open")
        return False
    robot_prim = _find_robot_root(stage)
    if not robot_prim:
        print("FAIL: no robot root")
        return False
    stage.SetDefaultPrim(robot_prim)

    # 1) Flatten
    flat_path, stage = _flatten_usd(stage, usd_path)

    # 2) 展平文件上操作
    robot_prim = _find_robot_root(stage)
    if not robot_prim:
        return False

    for child in robot_prim.GetChildren():
        if not _is_link(child):
            continue
        p = str(child.GetPath())

        # 取消 instanceable
        for cat in ("visuals", "collisions"):
            try:
                make_uninstanceable(f"{p}/{cat}", stage)
            except Exception:
                pass

        # 复制 mesh 到顶层
        for cat in ("visuals", "collisions"):
            cat_prim = stage.GetPrimAtPath(f"{p}/{cat}")
            if not cat_prim.IsValid():
                continue
            mesh_list = []
            def _find(p):
                for c in p.GetChildren():
                    if c.GetTypeName() == "Mesh":
                        mesh_list.append(c)
                    _find(c)
            _find(cat_prim)
            for i, src in enumerate(mesh_list):
                dst = f"{p}/{cat}/{'mesh' if i == 0 else f'mesh_{i}'}"
                try:
                    dst_mesh = UsdGeom.Mesh.Define(stage, dst)
                    for attr_name in ["points", "faceVertexCounts", "faceVertexIndices",
                                      "extent", "normals", "primvars:st", "doubleSided"]:
                        attr = src.GetAttribute(attr_name)
                        if attr.HasValue():
                            dst_prim = dst_mesh.GetPrim()
                            dst_prim.CreateAttribute(attr_name, attr.GetTypeName()).Set(attr.Get())
                    rel = src.GetRelationship("material:binding")
                    if rel and rel.GetTargets():
                        dst_rel = dst_mesh.GetPrim().CreateRelationship("material:binding")
                        for t in rel.GetTargets():
                            dst_rel.AddTarget(t)
                except Exception:
                    pass

        # Deactivate 旧嵌套文件夹
        for cat in ("visuals", "collisions"):
            cat_prim = stage.GetPrimAtPath(f"{p}/{cat}")
            if not cat_prim.IsValid():
                continue
            for c in cat_prim.GetChildren():
                if c.GetName() == "Mesh" or c.GetName().startswith("mesh"):
                    continue
                try:
                    c.SetActive(False)
                except Exception:
                    pass

        # purpose + CollisionAPI
        cat_prim = stage.GetPrimAtPath(f"{p}/visuals")
        if cat_prim.IsValid():
            for m in Usd.PrimRange(cat_prim):
                if m.GetTypeName() == "Mesh":
                    UsdGeom.Imageable(m).GetPurposeAttr().Clear()

        cat_prim = stage.GetPrimAtPath(f"{p}/collisions")
        if cat_prim.IsValid():
            for m in Usd.PrimRange(cat_prim):
                if m.GetTypeName() == "Mesh":
                    UsdGeom.Imageable(m).CreatePurposeAttr(UsdGeom.Tokens.proxy)
                    _apply_collision_api(m)

    # 关节限位
    for prim in Usd.PrimRange(robot_prim):
        if prim.GetTypeName() not in ("RevoluteJoint", "PhysicsRevoluteJoint"):
            continue
        for prop in ("Physics:lower", "Physics:upper"):
            attr = prim.CreateAttribute(prop, Sdf.ValueTypeNames.Float)
            attr.Set(-1e10 if "lower" in prop else 1e10)

    # 保存
    stage.GetRootLayer().Save()
    if flat_path != os.path.abspath(usd_path):
        shutil.copy2(flat_path, usd_path)
    print("✓")
    return True


def export_and_clean(urdf_path, output_usd_path):
    print(f"URDF → Clean USD")
    print(f"  URDF: {urdf_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_usd_path)), exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(os.path.abspath(output_usd_path)),
        usd_file_name=os.path.basename(output_usd_path),
        force_usd_conversion=True,
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_fixed_joints,
        collider_type="convex_decomposition" if COLLISION_MODE == 2 else "convex_hull",
        collision_from_visuals=False,
        make_instanceable=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type=args_cli.drive_type,
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.stiffness, damping=args_cli.damping
            ),
        ),
    )
    try:
        converter = UrdfConverter(cfg)
        print(f"  ✓ Export: {converter.usd_path}")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        return False

    clean_usd_file(converter.usd_path)
    if converter.usd_path != os.path.abspath(output_usd_path):
        shutil.copy2(converter.usd_path, output_usd_path)

    # 清理中间文件
    out_dir = os.path.dirname(os.path.abspath(output_usd_path))
    for f in [".asset_hash", "config.yaml"]:
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)
    flat = os.path.splitext(os.path.abspath(output_usd_path))[0] + "_flattened.usd"
    if os.path.exists(flat) and flat != os.path.abspath(output_usd_path):
        os.remove(flat)

    print(f"  ✓ Done: {output_usd_path}")
    return True


def main():
    urdf = args_cli.urdf or DEFAULT_URDF_PATH
    if args_cli.usd_only:
        if os.path.exists(args_cli.usd_only):
            clean_usd_file(args_cli.usd_only)
            output_usd_path = args_cli.usd_only
    else:
        if not os.path.exists(urdf):
            print(f"ERROR: {urdf} not found/n请修改 DEFAULT_URDF_PATH")
            return
        out = args_cli.output or os.path.splitext(urdf)[0] + ".usd"
        export_and_clean(urdf, out)
        output_usd_path = out

    # 转换完成后，如果有 GUI 则打开 USD 并保持 Isaac Sim 运行
    carb_settings_iface = carb.settings.get_settings()
    local_gui = carb_settings_iface.get("/app/window/enabled")
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    if local_gui or livestream_gui:
        print(f"/n打开 USD 文件: {output_usd_path}")
        sim_utils.open_stage(output_usd_path)
        app = omni.kit.app.get_app_interface()
        print("Isaac Sim 保持运行中，关闭窗口退出")
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
