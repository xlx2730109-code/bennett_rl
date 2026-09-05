"""bennett_1.xml FINAL generator -- faithful Urdf_Bennett_3 pantograph (keeps _3).

The Bennett leg is a parallelogram (pantograph) closed chain:
    thigh -> calf -> _3   AND   thigh -> _1 -> [_2, foot(fixed on _1)]
The physical closure is the _3 rod's ankle tip meeting the _2 rod's ankle tip at
the shared ankle joint, so the loop is closed with  <connect _3 <->_2>  anchored
at the ankle (the point where the _3 mesh meets the _2 mesh at the design pose).

NOTES (REGRESSION HISTORY):
- v1 deleted the _3 coupler and (wrongly) welded calf<->_2 at the calf<->_3 pivot
  -> the lower leg broke open and looked disconnected.  v2/final keeps _3 and
  closes at the real ankle -> the leg renders connected.
- The foot-end (ground contact) is the _1<->_foot joint origin, i.e. the fixed
  `*_foot` body on `_1` -- that is what sim2sim tracks, NOT `*_2`.
- The `imu` site is a sensor marker that fakes a grey ball floating on the base;
  it is set to group 5 so it renders hidden while MuJoCo sensors still read it.
"""

import xml.etree.ElementTree as ET
import pathlib
import numpy as np
import mujoco as mj
import math

URDF = r"E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\assets\robots\Urdf_Bennett_3\urdf\Urdf_Bennett_3.urdf"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "bennett_1.xml"

RANGE = {"thigh": (-0.80, 0.80), "calf": (-0.90, 0.55)}
PASSIVE_RANGE = (-3.14, 3.14)


def fmt(x): return f"{float(x):.6g}"
def vec3(s): return [float(v) for v in s.split()]


def rpy_to_quat(rpy):
    r, p, y = [float(v) for v in rpy.split()]
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


def eig_inertia(ixx, ixy, ixz, iyy, iyz, izz):
    D = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    w, V = np.linalg.eigh(D)
    R = V
    m00, m01, m02 = R[0,0], R[0,1], R[0,2]
    m10, m11, m12 = R[1,0], R[1,1], R[1,2]
    m20, m21, m22 = R[2,0], R[2,1], R[2,2]
    tr = m00+m11+m22
    if tr > 0:
        s = np.sqrt(tr+1.0)*2; qw=0.25*s; qx=(m21-m12)/s; qy=(m02-m20)/s; qz=(m10-m01)/s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0+m00-m11-m22)*2; qw=(m21-m12)/s; qx=0.25*s; qy=(m01+m10)/s; qz=(m02+m20)/s
    elif m11 > m22:
        s = np.sqrt(1.0+m11-m00-m22)*2; qw=(m02-m20)/s; qx=(m01+m10)/s; qy=0.25*s; qz=(m12+m21)/s
    else:
        s = np.sqrt(1.0+m22-m00-m11)*2; qw=(m10-m01)/s; qx=(m02+m20)/s; qy=(m12+m21)/s; qz=0.25*s
    if qw < 0: qx, qy, qz, qw = -qx, -qy, -qz, -qw
    return w, [qw, qx, qy, qz]


def geom_for_link(link):
    vis = link.find("visual")
    if vis is not None:
        g = vis.find("geometry")
        if g is not None and g.find("mesh") is not None:
            return g.find("mesh").get("filename").split("/")[-1].replace(".STL", "")
    return None


def inertial_for_link(link):
    inert = link.find("inertial")
    if inert is None: return None
    mass = float(inert.find("mass").get("value"))
    o = inert.find("origin")
    pos = vec3(o.get("xyz")) if o is not None else [0,0,0]
    rpy = o.get("rpy") if o is not None else "0 0 0"
    q = rpy_to_quat(rpy)
    i = inert.find("inertia")
    if i is None:
        w, qq = np.array([1e-4,1e-4,1e-4]), [1,0,0,0]
    else:
        it = dict(i.attrib)
        w, qq = eig_inertia(float(it["ixx"]),float(it["ixy"]),float(it["ixz"]),
                            float(it["iyy"]),float(it["iyz"]),float(it["izz"]))
        w = np.sort(w)
    return mass, pos, qq, w


tree = ET.parse(URDF).getroot()
links = {l.get("name"): l for l in tree.iter("link")}
joints, children = {}, {}
for j in tree.iter("joint"):
    p = j.find("parent").get("link"); c = j.find("child").get("link")
    ty = j.get("type"); o = j.find("origin"); ax = j.find("axis")
    joints[c] = (p, ty,
                 vec3(o.get("xyz")) if o is not None else [0,0,0],
                 vec3(ax.get("xyz")) if ax is not None else [1,0,0])
    children.setdefault(p, []).append(c)


def joint_range(name):
    if name.startswith(("FL_", "FR_", "RL_", "RR_")):
        part = name.split("_")[1]
        if part == "foot": return None
        if part in ("thigh", "calf"): return RANGE[part]
    return PASSIVE_RANGE


def emit_body(c, indent):
    p, ty, origin, axis = joints[c]
    link = links[c]
    attrs = [f'name="{c}"',
             'pos="%s %s %s"' % (fmt(origin[0]), fmt(origin[1]), fmt(origin[2])),
             'quat="%s"' % (" ".join(fmt(q) for q in rpy_to_quat("0 0 0")))]
    pad = "  " * indent
    lines = [f"{pad}<body {' '.join(attrs)}>"]
    inert = inertial_for_link(link)
    if inert:
        mass, pos, qq, w = inert
        lines.append(f'{pad}  <inertial pos="{fmt(pos[0])} {fmt(pos[1])} {fmt(pos[2])}" '
                     f'quat="{" ".join(fmt(q) for q in qq)}" mass="{fmt(mass)}" '
                     f'diaginertia="{fmt(w[0])} {fmt(w[1])} {fmt(w[2])}"/>')
    mesh = geom_for_link(link)
    if mesh:
        lines.append(f'{pad}  <geom type="mesh" contype="2" conaffinity="1" mesh="{mesh}"/>')
    rng = joint_range(c)
    if ty != "fixed" and rng is not None:
        ax = " ".join(fmt(a) for a in joints[c][3])
        lines.append(f'{pad}  <joint name="{c}" pos="0 0 0" axis="{ax}" '
                     f'range="{fmt(rng[0])} {fmt(rng[1])}"/>')
    for ch in children.get(c, []):
        lines.extend(emit_body(ch, indent + 1))   # keep _3 (pantograph coupler)
    lines.append(f"{pad}</body>")
    return lines


blink = links["base"]
base_inert = inertial_for_link(blink)
bmass, bpos, bq, bw = base_inert

_meshes = ["base"] + [f"{l}_{p}" for l in ["FL","FR","RL","RR"]
                      for p in ["thigh","calf","1","2","3","foot"]]   # keep _3
_mesh_decls = "\n".join(f'    <mesh name="{m}" file="{m}.STL"/>' for m in _meshes)

body_lines = []
for ch in children.get("base", []):
    body_lines.extend(emit_body(ch, 3))

xml = f'''<?xml version="1.0"?>
<mujoco model="bennett_1">
  <compiler angle="radian" meshdir="meshes/"/>
  <option timestep="0.002" iterations="50" solver="Newton" tolerance="1e-10">
    <flag energy="disable" contact="enable"/>
  </option>
  <default>
    <joint damping="0.001" armature="0.01" frictionloss="0.05" limited="true"/>
    <geom contype="2" conaffinity="1" condim="3" solref="0.005 1" friction="0.9 0.2 0.2" group="1"/>
    <motor ctrllimited="true"/>
  </default>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="1 1" reflectance="0.2"/>
{_mesh_decls}
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.25" type="plane" material="groundplane" contype="1" conaffinity="2"/>
    <body name="base" pos="0 0 0.30" quat="1 0 0 0">
      <inertial pos="{fmt(bpos[0])} {fmt(bpos[1])} {fmt(bpos[2])}"
        quat="{" ".join(fmt(q) for q in bq)}" mass="{fmt(bmass)}"
        diaginertia="{fmt(bw[0])} {fmt(bw[1])} {fmt(bw[2])}"/>
      <joint name="floating_base_joint" type="free"/>
      <site name="imu" size="0.01" pos="0 0 0" quat="1 0 0 0" group="5"/>
      <geom type="mesh" contype="2" conaffinity="1" mesh="base"/>
{chr(10).join(body_lines)}
    </body>
  </worldbody>
  <actuator>
    <position name="FL_thigh" joint="FL_thigh" gear="1" kp="28" kv="2" forcerange="-8 8" ctrllimited="true" ctrlrange="-0.80 0.80"/>
    <position name="FL_calf"  joint="FL_calf"  gear="1" kp="28" kv="2" forcerange="-8 8" ctrllimited="true" ctrlrange="-0.90 0.55"/>
    <position name="FR_thigh" joint="FR_thigh" gear="1" kp="28" kv="2" forcerange="-8 8" ctrllimited="true" ctrlrange="-0.80 0.80"/>
    <position name="FR_calf"  joint="FR_calf"  gear="1" kp="28" kv="2" forcerange="-8 8" ctrllimited="true" ctrlrange="-0.90 0.55"/>
    <position name="RL_thigh" joint="RL_thigh" gear="1" kp="28" kv="2" forcerange="-8 8" ctrllimited="true" ctrlrange="-0.80 0.80"/>
    <position name="RL_calf"  joint="RL_calf"  gear="1" kp="28" kv="2" forcerange="-8 8" ctrllimited="true" ctrlrange="-0.90 0.55"/>
    <position name="RR_thigh" joint="RR_thigh" gear="1" kp="28" kv="2" forcerange="-8 8" ctrllimited="true" ctrlrange="-0.80 0.80"/>
    <position name="RR_calf"  joint="RR_calf"  gear="1" kp="28" kv="2" forcerange="-8 8" ctrllimited="true" ctrlrange="-0.90 0.55"/>
  </actuator>
</mujoco>
'''
OUT.write_text(xml, encoding="utf-8")
print("wrote", OUT)


# ---- close the pantograph: connect _3 <->_2 at the ankle ----
def add_weld():
    m = mj.MjModel.from_xml_path(str(OUT))
    d = mj.MjData(m)
    d.qpos[:] = 0.0; d.qpos[2] = 0.30; d.qpos[3] = 1.0
    mj.mj_forward(m, d)

    def Rof(b):
        R = np.zeros(9); mj.mju_quat2Mat(R, np.asarray(d.xquat[b], float)); return R.reshape(3, 3)
    def scale(a): return " ".join(f"{x:.6g}" for x in a)

    def verts(name):
        mid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_MESH, name)
        a0, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        return m.mesh_vert[a0:a0+n]

    eq = ["  <equality>"]
    for leg in ["FL", "FR", "RL", "RR"]:
        b3 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{leg}_3")
        b2 = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{leg}_2")
        # ankle = closest point pair between the _3 and _2 meshes at the design pose
        W3 = Rof(b3) @ verts(f"{leg}_3").T + d.xpos[b3][:, None]
        W2 = Rof(b2) @ verts(f"{leg}_2").T + d.xpos[b2][:, None]
        d2 = ((W3[:, :, None] - W2[:, None, :]) ** 2).sum(axis=0)
        i, j = np.unravel_index(np.argmin(d2), d2.shape)
        ankle_world = 0.5 * (W3[:, i] + W2[:, j])
        dist = np.sqrt(d2[i, j])
        # express the ankle in the _3 body local frame -> the connect anchor (body1=_3)
        anchor = Rof(b3).T @ (ankle_world - d.xpos[b3])
        eq.append(f'    <connect body1="{leg}_3" body2="{leg}_2" anchor="{scale(anchor)}" '
                  f'solref="0.01 1"/>')
        print(f"{leg}: ankle {np.round(ankle_world,4)}  _3<_2 closest dist@design={dist:.4f}  anchor(_3-local)={scale(anchor)}")
    eq.append("  </equality>")
    x = OUT.read_text(encoding="utf-8")
    x = x.rstrip().rstrip("</mujoco>").rstrip() + "\n\n" + "\n".join(eq) + "\n</mujoco>\n"
    OUT.write_text(x, encoding="utf-8")
    print("injected <connect> (_3 <->_2 ankle) ->", OUT)

add_weld()
