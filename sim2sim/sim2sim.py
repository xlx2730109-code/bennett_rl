# 共享引擎(加载MJCF/映射关节/步态时钟/构造obs/施加速度)





"""
Shared MuJoCo sim2sim engine for Bennett quadruped tasks.

Replays a trained TorchScript policy (exported from IsaacLab) on a MuJoCo
model that reproduces the *same* observation -> action interface.  Run with:

    python sim2sim.py --task quad_leg_trot1

The per-task contract (obs order, gait clock, joint mapping, model/policy
paths, action scale/clip, default pose) lives in configs/<task>.py ; the
engine is topology-agnostic and just follows the config.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys

from pynput import keyboard as pk
import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import TaskConfig


# ---------------------------------------------------------------------------- #
#  Geometry / orientation helpers
# ---------------------------------------------------------------------------- #
def _quat_conj_rotate(q, v):
    """Rotate vector v into the frame of a body whose world quaternion is q."""
    w, x, y, z = q
    qv = np.array([-x, -y, -z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _norm(v):
    return float(np.sqrt(np.sum(np.asarray(v) ** 2)))


# ---------------------------------------------------------------------------- #
#  MuJoCo runner
# ---------------------------------------------------------------------------- #
class MujocoRunner:
    def __init__(self, cfg: TaskConfig, headless: bool):
        self.cfg = cfg
        self.headless = headless
        self.model = mj.MjModel.from_xml_path(cfg.model)
        self.data = mj.MjData(self.model)

        # joint ids (for qpos) + dof adrs (for qvel/force), in training order
        self._act_jid = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jn)
                         for jn in cfg.actuated_joints]
        self._act_dof = np.asarray([self.model.jnt_dofadr[j] for j in self._act_jid],
                                   dtype=np.int64)
        # actuator ids, in training order (matches MJCF order)
        self._act_ids = np.asarray([
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, jn)
            for jn in cfg.actuated_joints], dtype=np.int64)

        self.base_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base")
        self.imu_site = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "imu")
        # floating (free) base => walking locomotion; welded base => fixed trace
        self._has_free_base = (mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_JOINT, "floating_base_joint") != -1)

        # gait clock state (single env, continuous)
        self.phase = 0.0
        self.step = 0
        self.last_step = 0
        self.was_moving = False
        self.command = np.zeros(3, dtype=np.float32)

        # previous raw action (obs term 'actions')
        self._last_action = np.zeros(cfg.num_actions, dtype=np.float32)

        # --- trace (fixed-base reference tracking) ---
        self.is_trace = (cfg.obs_mode == "trace")
        if self.is_trace:
            tr = cfg.trace
            amp = float(tr["amplitude_rad"])
            max_spd = float(tr["max_speed_rad_s"])
            freq_hz = max_spd / max(2.0 * math.pi * amp, 1.0e-6)
            self._trace_amp = amp
            self._trace_omega = 2.0 * math.pi * freq_hz
            self._trace_calf_phase = float(tr.get("calf_phase_rad", math.pi / 2.0))
            self._trace_scale = float(tr["scale"])
            self._trace_max_delta = float(tr["max_joint_speed"]) * cfg.step_dt
            self._trace_applied = np.zeros(2, dtype=np.float32)
            self._last_raw_action = np.zeros(2, dtype=np.float32)
            self._default_by_name = dict(zip(cfg.actuated_joints, cfg.default_joint_pos))
            tbl = tr["table"]
            n = len(tbl)
            self._trace_act = np.zeros(n, dtype=np.int64)    # actuator ids
            self._trace_jid = np.zeros(n, dtype=np.int64)    # joint ids
            self._trace_dof = np.zeros(n, dtype=np.int64)    # dof adrs
            self._trace_kind = np.zeros(n, dtype=np.int64)   # 0=thigh,1=calf
            self._trace_sign = np.zeros(n, dtype=np.float32)
            self._trace_default = np.zeros(n, dtype=np.float32)
            for i, (jname, kind, sign) in enumerate(tbl):
                jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
                self._trace_act[i] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, jname)
                self._trace_jid[i] = jid
                self._trace_dof[i] = self.model.jnt_dofadr[jid]
                self._trace_kind[i] = int(kind)
                self._trace_sign[i] = float(sign)
                self._trace_default[i] = float(self._default_by_name[jname])
            self._ctrl_defaults = np.asarray(cfg.default_joint_pos, dtype=np.float32).copy()

        # endpoint (foot) trail overlay -- enabled via enable_endpoint_trail()
        self._endpoint_ids = []
        self._endpoint_trails = []
        self._endpoint_points = 0
        self._endpoint_stride = 1
        self._marker_size = None
        self._trail_size = None
        self._mat_id = None
        self._endpoint_colours = ()

        self.reset()
        self._viewer = None
        if not headless:
            self._viewer = mjviewer.launch_passive(self.model, self.data)
            self._viewer.cam.distance = 1.6
            self._viewer.cam.lookat = np.array([0.0, 0.0, 0.25])

    # ---------------- state setup ---------------- #
    def reset(self):
        d, m = self.data, self.model
        self.phase = 0.0
        self.step = 0
        self.last_step = 0
        self.was_moving = False
        self._last_action[:] = 0.0
        if self.is_trace:
            self._trace_applied[:] = 0.0
            self._last_raw_action[:] = 0.0
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        if self._has_free_base:
            # floating base: set its pose explicitly (walking)
            b = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "base")
            d.xpos[b] = self.cfg.init_base_pos  # local hint, qpos set explicitly below
            d.qpos[:3] = self.cfg.init_base_pos
            d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        # driven joints to default pose, passive linkage left free
        for jid, q in zip(self._act_jid, self.cfg.default_joint_pos):
            d.qpos[m.jnt_qposadr[jid]] = float(q)
        d.qvel[:] = 0.0
        if self._endpoint_ids:
            for tr in self._endpoint_trails:
                tr.clear()
        mj.mj_forward(m, d)
        mj.mj_step(m, d)

    def set_command(self, vx: float, vy: float, wz: float):
        self.command[:] = (vx, vy, wz)

    # ---------------- physics / control loop (one control period) ---------------- #
    def _advance_gait(self):
        """Port of the task's gait clock to a single env.

        Handles both the speed-conditioned diagonal trot (trot1/slope4) and the
        fixed-frequency crawl (go2).  Both produce the same 17-D observation
        block; only the schedule math differs.  Returns
        (global_phase, leg_phase, desired_contact, freq, duty, height).
        """
        g = self.cfg.gait
        cmd = self.command
        moving = _norm(cmd[:3]) >= g["command_deadband"]

        if self.cfg.obs_mode == "crawl":
            # Fixed low-speed crawl: no speed/duty blending.
            freq = float(g["frequency_hz"])
            duty = float(g["duty_factor"])
            height = float(g["swing_height"])
            start_phase = 0.0
            swing_frac = float(1.0 - duty)
        else:
            # Speed-conditioned trot: blend freq/duty toward the command.
            equiv = _norm(cmd[:2]) + g["yaw_equivalent_radius"] * abs(cmd[2])
            blend = float(np.clip(
                (equiv - g["min_equivalent_speed"]) /
                max(g["max_equivalent_speed"] - g["min_equivalent_speed"], 1.0e-6), 0.0, 1.0))
            freq = g["min_frequency_hz"] + blend * (g["max_frequency_hz"] - g["min_frequency_hz"])
            duty = g["low_speed_duty_factor"] + blend * (g["high_speed_duty_factor"] - g["low_speed_duty_factor"])
            height = g["swing_height"]
            start_phase = float(np.clip(1.0 - duty, 0.0, 0.5))
            swing_frac = float(np.clip(1.0 - duty, 0.0, 0.5))

        if not moving:
            freq, duty, height = 0.0, 1.0, 0.0

        delta = float(self.step - self.last_step)
        phase = float(np.mod(self.phase + delta * self.cfg.step_dt * freq, 1.0))
        just_started = moving and not self.was_moving
        if just_started:
            phase = start_phase
        self.phase, self.last_step, self.was_moving = phase, self.step, moving

        # schedule
        offsets = g["phase_offsets"]
        leg_phase = np.array([np.mod(phase - o, 1.0) for o in offsets], dtype=np.float32)
        desired_swing = (leg_phase < swing_frac) & moving
        desired_contact = ~desired_swing
        global_phase = phase if moving else 0.0
        if not moving:
            leg_phase[:] = 0.0
        return global_phase, leg_phase, desired_contact.astype(np.float32), freq, duty, height

    def _obs(self):
        d, m = self.data, self.model
        if self.is_trace:
            t = self.step * self.cfg.step_dt
            phase = self._trace_omega * t
            sinp, cosp = float(np.sin(phase)), float(np.cos(phase))
            n = len(self._trace_act)
            ref = np.zeros(n, dtype=np.float32)
            jpos = np.zeros(n, dtype=np.float32)
            jvel = np.zeros(n, dtype=np.float32)
            for i in range(n):
                joint_phase = phase + (self._trace_calf_phase if self._trace_kind[i] == 1 else 0.0)
                ref[i] = self._trace_sign[i] * self._trace_amp * float(np.sin(joint_phase))
                jpos[i] = d.qpos[m.jnt_qposadr[self._trace_jid[i]]] - self._trace_default[i]
                jvel[i] = d.qvel[self._trace_dof[i]]
            err = jpos - ref
            return np.concatenate([
                np.array([sinp, cosp], dtype=np.float32),  # 2  phase sin/cos
                ref,             # n  reference offsets
                err,             # n  tracking error
                jpos,            # n  joint_pos_rel
                jvel,            # n  joint_vel_rel
                self._last_raw_action,  # 2  previous raw action
            ])

        base_q = np.array([d.qpos[3], d.qpos[4], d.qpos[5], d.qpos[6]], dtype=np.float64)  # w,x,y,z
        ang_w = np.array([d.qvel[3], d.qvel[4], d.qvel[5]])  # world angular vel
        base_ang_vel = _quat_conj_rotate(base_q, ang_w)

        grav = _quat_conj_rotate(base_q, np.array([0.0, 0.0, -1.0]))  # projected gravity in base frame

        cmd = self.command.astype(np.float32)

        joint_pos = np.array([d.qpos[m.jnt_qposadr[j]] for j in self._act_jid], dtype=np.float32)
        joint_vel = np.array([d.qvel[a] for a in self._act_dof], dtype=np.float32)
        joint_pos_rel = joint_pos - self.cfg.default_joint_pos

        base_obs = [
            base_ang_vel.astype(np.float32),   # 3  world->base ang vel
            grav.astype(np.float32),            # 3  projected gravity
            cmd,                                # 3  velocity command
            joint_pos_rel,                      # 8
            joint_vel,                          # 8
            self._last_action,                  # 8  previous action
        ]

        # Stair is gait-free (33-D): base_lin_vel and height_scan are unset and
        # no clock terms are appended.  Trot/crawl append the same 17-D block.
        if self.cfg.obs_mode == "plain":
            return np.concatenate(base_obs)

        gp, lp, dc, freq, duty, height = self._advance_gait()
        phase_sin_cos = np.array(
            [np.sin(2 * np.pi * gp), np.cos(2 * np.pi * gp)], dtype=np.float32)
        leg_phase_sin_cos = np.concatenate(
            [np.sin(2 * np.pi * lp), np.cos(2 * np.pi * lp)]).astype(np.float32)
        gait_params = np.array([freq, duty, height], dtype=np.float32)

        return np.concatenate([
            *base_obs,          # 33
            phase_sin_cos,      # 2
            leg_phase_sin_cos,  # 8
            dc,                 # 4
            gait_params,        # 3
        ])

    def _apply_action(self, raw: np.ndarray):
        """Isaac JointPositionAction: target = clamp(raw*scale + default, clip)."""
        target = self.cfg.default_joint_pos + raw * self.cfg.action_scale
        target = np.clip(target, self.cfg.clip_low, self.cfg.clip_high)
        self.data.ctrl[self._act_ids] = target
        self._last_action[:] = raw

    def _apply_trace_action(self, raw: np.ndarray):
        """Rate-limited 2-D base action expanded to per-joint position targets.

        Mirrors mdp.SingleLegPositionAction / QuadLegPositionAction: ``raw`` is
        clamped to [-1,1], scaled by ``trace.scale``, and rate-limited against the
        previous applied offsets by ``trace.max_joint_speed``.  The two base
        offsets (thigh, calf) are then expanded to each controlled (trace) joint
        via its (kind, sign); the held (non-trace) joints stay pinned at default.
        """
        raw2 = np.clip(np.asarray(raw, dtype=np.float32).reshape(-1)[:2], -1.0, 1.0)
        desired = raw2 * self._trace_scale
        self._trace_applied += np.clip(
            desired - self._trace_applied, -self._trace_max_delta, self._trace_max_delta)
        self._last_raw_action[:] = np.asarray(raw, dtype=np.float32).reshape(-1)[:2]

        self.data.ctrl[self._act_ids] = self._ctrl_defaults   # held joints at default
        for i in range(len(self._trace_act)):
            base_off = self._trace_applied[self._trace_kind[i]]
            self.data.ctrl[self._trace_act[i]] = (
                self._trace_default[i] + self._trace_sign[i] * base_off)

    def step_control(self, raw: np.ndarray):
        if self.is_trace:
            self._apply_trace_action(raw)
        else:
            self._apply_action(raw)
        for _ in range(int(self.cfg.phys_dt and round(self.cfg.step_dt / self.cfg.phys_dt))):
            mj.mj_step(self.model, self.data)
        self.step += 1
        if self._endpoint_ids and self.step % self._endpoint_stride == 0:
            for k, body_id in enumerate(self._endpoint_ids):
                tr = self._endpoint_trails[k]
                tr.append(np.array(self.data.xpos[body_id], dtype=np.float64))
                if len(tr) > self._endpoint_points:
                    del tr[: len(tr) - self._endpoint_points]
        return self._obs()

    # ---------------- viewer ---------------- #
    def render(self):
        if self._viewer is not None:
            if self._viewer.is_running():
                if self._endpoint_ids:
                    self._draw_endpoint_trail()
                self._viewer.sync()
                return True
            return False
        return True

    # ---------------- endpoint trail overlay ---------------- #
    def enable_endpoint_trail(self, bodies, points, stride, marker_r, trail_r):
        """Draw each endpoint body (e.g. foot `*_foot`, the fixed child of `_1`)
        as a coloured marker plus a
        position trail sampled every ``stride`` control steps.  This is the MuJoCo
        viewer equivalent of ``play.py --endpoint_trail``: coloured lifecycle
        markers written into the passive viewer's ``user_scn`` on top of the sim.

        Colours are fixed per body index (green, blue, orange, magenta) matching
        Isaac's ``trail_colours`` palette.
        """
        self._endpoint_ids = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, b) for b in bodies]
        missing = [b for b, i in zip(bodies, self._endpoint_ids) if i == -1]
        if missing:
            raise RuntimeError(f"endpoint bodies not found in model: {missing}")
        self._endpoint_trails = [[] for _ in bodies]
        self._endpoint_points = int(points)
        self._endpoint_stride = max(1, int(stride))
        self._marker_size = np.array([marker_r] * 3, dtype=float)
        self._trail_size = np.array([trail_r] * 3, dtype=float)
        self._mat_id = np.eye(3, dtype=float).ravel()
        self._endpoint_colours = (
            (0.0, 1.0, 0.0),    # green
            (0.0, 0.55, 1.0),   # blue
            (1.0, 0.75, 0.0),   # orange
            (1.0, 0.0, 0.65),   # magenta
        )
        print(f"[sim2sim] endpoint trail ON  bodies={list(bodies)} "
              f"points={points} stride={self._endpoint_stride}.")

    def _draw_endpoint_trail(self):
        scn = self._viewer.user_scn
        with self._viewer.lock():
            idx = 0
            n_col = len(self._endpoint_colours)
            for k, body_id in enumerate(self._endpoint_ids):
                c = self._endpoint_colours[k % n_col]
                rgba = np.array((c[0], c[1], c[2], 1.0), dtype=float)
                # sampled trail (small spheres)
                for p in self._endpoint_trails[k]:
                    mj.mjv_initGeom(scn.geoms[idx], mj.mjtGeom.mjGEOM_SPHERE,
                                    self._trail_size, p, self._mat_id, rgba)
                    idx += 1
                # current body marker (large sphere)
                cur = self.data.xpos[body_id]
                mj.mjv_initGeom(scn.geoms[idx], mj.mjtGeom.mjGEOM_SPHERE,
                                self._marker_size, cur, self._mat_id, rgba)
                idx += 1
            scn.ngeom = idx

    def info_report(self):
        b = self.data.xpos[self.base_id]
        foot = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, n)
                for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
        fz = [self.data.xpos[i][2] for i in foot]
        return {"base": (round(b[0], 3), round(b[1], 3), round(b[2], 3)),
                "feet_z": [round(z, 3) for z in fz]}


# ---------------------------------------------------------------------------- #
#  Config loader
# ---------------------------------------------------------------------------- #
def load_config(task: str):
    path = pathlib.Path(__file__).parent / "configs" / f"{task}.py"
    spec = importlib.util.spec_from_file_location(task, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


# ---------------------------------------------------------------------------- #
#  Keyboard
# ---------------------------------------------------------------------------- #
# Hold-to-move is driven by an OS-level pynput key listener (runs on its own
# thread), NOT by the MuJoCo viewer: the viewer window is created on a
# background thread (launch_passive), so GLFW key state is not reachable from
# the main control loop (get_key polled from here always reads 0 -> "holding
# does nothing").  pynput is thread-agnostic and gives both on_press and
# on_release, so the command follows the keys exactly: hold to move, release to
# stop.
_MAG = {"fwd": 0.30, "back": 0.30, "left": 0.10, "right": 0.10,
        "turn_l": 0.40, "turn_r": 0.40}
_AXIS = {"fwd": (0, +1.0), "back": (0, -1.0),
         "left": (1, +1.0), "right": (1, -1.0),
         "turn_l": (2, +1.0), "turn_r": (2, -1.0)}
_DIGIT = {"6": "turn_l", "7": "turn_r"}


class Keyboard:
    """Hold-to-move driver.  read() -> (vx, vy, wz).

    ``start()`` spawns the pynput listener; ``read()`` returns the command this
    step from whichever keys are currently held; ``stop()`` tears it down.
    """

    def __init__(self):
        self._held = set()
        self._listener = None

    def start(self):
        if self._listener is not None:
            return

        def on_press(key):
            name = self._resolve(key)
            if name is not None:
                self._held.add(name)

        def on_release(key):
            name = self._resolve(key)
            if name is not None:
                self._held.discard(name)

        self._listener = pk.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @staticmethod
    def _resolve(key):
        if key == pk.Key.up:
            return "fwd"
        if key == pk.Key.down:
            return "back"
        if key == pk.Key.left:
            return "left"
        if key == pk.Key.right:
            return "right"
        try:
            c = key.char
        except AttributeError:
            c = None
        return _DIGIT.get(c)

    def read(self):
        if not self._held:
            return 0.0, 0.0, 0.0
        v = [0.0, 0.0, 0.0]
        for name in self._held:
            axis, sign = _AXIS[name]
            v[axis] += sign * _MAG[name]
        return v[0], v[1], v[2]


class CommandReplaySampler:
    """Reproduce Isaac Lab ``mdp.UniformVelocityCommandCfg`` for a scripted replay.

    Every ``resampling_time_range`` seconds (uniform) a fresh (vx, vy, wz) command
    is drawn uniformly from the training command ranges; with probability
    ``standing_frac`` it is instead the zero (stand-still) command.  ``heading``
    is False, so the 3rd component is a raw yaw-RATE -- exactly the command block
    the bridge sends, so the replay feeds the policy the same velocity_command obs
    it saw in training.

    This is what makes a ``--cmd_play`` replay move in ALL directions, including the
    lateral ``lin_vel_y`` that free_gait3 trained on (vy=+-0.25), instead of the
    fixed forward-only (0.30, 0, 0) that the old headless smoke test used.
    """

    def __init__(self, spec, step_dt, rng=None):
        spec = spec or {}
        self.rng = rng or np.random.default_rng()
        self.step_dt = float(step_dt)
        self.standing = float(spec.get("standing_frac", spec.get("standing", 0.0)))
        self.resample_range = tuple(spec.get("resampling_time_range", (1.0, 1.0)))
        self.heading = bool(spec.get("heading", False))
        # accept either a flat dict of ranges or a nested {"ranges": {...}}
        self.ranges = spec.get("ranges", spec)
        self._next = 0
        self._cmd = (0.0, 0.0, 0.0)

    def _draw(self, step):
        r = self.ranges
        if self.rng.random() < self.standing:
            self._cmd = (0.0, 0.0, 0.0)
        else:
            vx = self.rng.uniform(*r.get("lin_vel_x", (-0.30, 0.30)))
            vy = self.rng.uniform(*r.get("lin_vel_y", (0.0, 0.0)))
            wz = self.rng.uniform(*r.get("ang_vel_z", (0.0, 0.0)))
            self._cmd = (float(vx), float(vy), float(wz))
        t = float(self.rng.uniform(*self.resample_range))
        self._next = step + max(1, int(round(t / self.step_dt)))
        return self._cmd

    def step(self, step):
        if step >= self._next:
            return self._draw(step)
        return self._cmd


# ---------------------------------------------------------------------------- #
#  Main
# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="config name in configs/")
    ap.add_argument("--headless", action="store_true", help="no viewer, run --steps and exit")
    ap.add_argument("--steps", type=int, default=None, help="headless run length (control steps)")
    ap.add_argument("--cmd_play", action="store_true", default=False,
                    help="Scripted replay: resample (vx,vy,wz) from the training "
                         "UniformVelocityCommand ranges every `resampling_time_range` s "
                         "(needs cfg.command, e.g. free_gait3), so the robot moves in "
                         "ALL directions exactly like the training play video instead of "
                         "the fixed forward-only smoke test.")
    ap.add_argument("--record_video", type=str, default=None,
                    help="Offscreen-record the replay to an mp4 (uses mujoco.Renderer). "
                         "Adds the foot-end trail like play.py --endpoint_trail.")
    ap.add_argument("--cmd_seconds", type=float, default=None,
                    help="--cmd_play / --record_video run length in seconds "
                         "(default: cfg.ep_len_s).")
    # foot-trajectory overlay (mirrors play.py --endpoint_trail)
    ap.add_argument("--endpoint_trail", action="store_true", default=False,
                    help="Draw each endpoint body as a coloured marker with a trail.")
    ap.add_argument("--endpoint_bodies", type=str, nargs="+", default=None,
                    help="Bodies to trail (default FL_foot FR_foot RL_foot RR_foot).")
    ap.add_argument("--endpoint_trail_points", type=int, default=600,
                    help="Maximum saved trail points per body.")
    ap.add_argument("--endpoint_trail_stride", type=int, default=None,
                    help="Sample a trail point every N control steps. "
                         "Default = max(1, round(0.016/step_dt)).")
    ap.add_argument("--endpoint_radius", type=float, default=0.016,
                    help="Radius (m) of the current-body marker.")
    ap.add_argument("--endpoint_trail_radius", type=float, default=0.002,
                    help="Radius (m) of each trail marker.")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        torch = None

    cfg = load_config(args.task)
    cfg.task = args.task
    runner = MujocoRunner(cfg, headless=args.headless)

    if args.record_video and not args.endpoint_trail:
        args.endpoint_trail = True   # recorded replays get the foot-end trail by default
    if args.endpoint_trail:
        bodies = args.endpoint_bodies or ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        stride = args.endpoint_trail_stride
        if stride is None:
            stride = max(1, int(round(0.016 / cfg.step_dt)))
        runner.enable_endpoint_trail(bodies, args.endpoint_trail_points,
                                     stride, args.endpoint_radius,
                                     args.endpoint_trail_radius)

    policy_path = cfg.policy
    if torch is None:
        raise SystemExit("PyTorch is required to load the policy (run in env_isaaclab).")
    policy = torch.jit.load(policy_path)
    policy.eval()

    # initial obs
    obs = runner._obs()

    use_cmd_play = args.cmd_play or args.record_video is not None
    sampler = (CommandReplaySampler(cfg.command, cfg.step_dt) if use_cmd_play else None)
    if use_cmd_play and cfg.command is None:
        raise SystemExit("--cmd_play/--record_video needs a `command` spec in the task config "
                         "(e.g. quad_leg_free_gait3.py), which mirrors the training "
                         "UniformVelocityCommandCfg ranges.")
    run_s = (args.cmd_seconds or cfg.ep_len_s)

    if args.headless:
        n = args.steps or int(run_s / cfg.step_dt)
        if use_cmd_play:
            run_cmd_replay(runner, policy, obs, n, sampler,
                           record_path=args.record_video)
        else:
            # fixed forward command during headless smoke test
            run_headless(runner, policy, obs, n)
        return

    # interactive viewer loop -------------------------------------------------
    if runner.is_trace:
        print(f"[sim2sim] task={cfg.task}  model={cfg.model}\n"
              f"          control {1.0/cfg.step_dt:.0f} Hz / phys {1.0/cfg.phys_dt:.0f} Hz  "
              f"obs={cfg.num_obs} act={cfg.num_actions}\n"
              "fixed-base reference trace (deterministic, no joystick).  Ctrl+C quit.")
    else:
        print(f"[sim2sim] task={cfg.task}  model={cfg.model}\n"
              f"          control {1.0/cfg.step_dt:.0f} Hz / phys {1.0/cfg.phys_dt:.0f} Hz  "
              f"obs={cfg.num_obs} act={cfg.num_actions}\n"
              "keys (hold): arrows move, 6/7 turn.  Ctrl+C quit.")
    kbd = Keyboard()
    if not runner.is_trace and not use_cmd_play:
        kbd.start()
    frame = 0
    try:
        while True:
            vx = vy = wz = 0.0
            if not runner.is_trace:
                if use_cmd_play:
                    vx, vy, wz = sampler.step(frame)
                    runner.set_command(vx, vy, wz)
                else:
                    vx, vy, wz = kbd.read()
                    runner.set_command(vx, vy, wz)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().reshape(1, -1)
                raw = policy(obs_t).cpu().numpy().reshape(-1)

            obs = runner.step_control(raw)

            # allow debug base/feet print every 2s
            if frame % 100 == 0:
                if runner.is_trace:
                    print(f"t={frame*cfg.step_dt:5.1f}s  {runner.info_report()}")
                else:
                    print(f"t={frame*cfg.step_dt:5.1f}s cmd=({vx:.2f},{vy:.2f},{wz:.2f}) "
                          f"{runner.info_report()}")
            frame += 1

            if not runner.render():
                break
    finally:
        kbd.stop()


def run_headless(runner, policy, obs, n):
    import torch
    cfg = runner.cfg
    # coarse sway to exercise the scheduler: 1 s forward, 1 s turn, repeat
    t_on = int(1.0 / cfg.step_dt)
    for k in range(n):
        cyc = (k % (2 * t_on)) < t_on
        if not runner.is_trace:
            runner.set_command(0.30, 0.0, 0.0 if cyc else 0.4)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().reshape(1, -1)
            raw = policy(obs_t).cpu().numpy().reshape(-1)
        obs = runner.step_control(raw)
        if k % 50 == 0:
            print(f"step {k}/{n}  {runner.info_report()}")
    print("[sim2sim] headless done. final:", runner.info_report())


def _render_offscreen_scene(runner, renderer, cam):
    """Update the offscreen scene for renderer and inject the endpoint (foot) trail.
    Mirrors ``_draw_endpoint_trail`` but writing into ``renderer.scene`` (offscreen
    Renderer has no ``user_scn`` like the GUI viewer).  Falls back to the plain scene
    if the scene has no room for the extra geoms."""
    import mujoco as mj
    d = runner.data
    cam.lookat = d.xpos[runner.base_id].astype(np.float64)   # keep the robot in frame
    renderer.update_scene(d, camera=cam)
    if runner._endpoint_ids:
        scn = renderer.scene
        avail = scn.maxgeom - scn.ngeom
        if avail <= 0:
            return renderer.render()
        idx = scn.ngeom
        added = 0
        n_col = len(runner._endpoint_colours)
        for k, body_id in enumerate(runner._endpoint_ids):
            c = runner._endpoint_colours[k % n_col]
            rgba = np.array((c[0], c[1], c[2], 1.0), dtype=float)
            # oldest-first truncation if the scene cannot hold every trail point
            tr = runner._endpoint_trails[k][-max(0, (avail - added) - 1):]
            for p in tr:
                if added >= avail:
                    break
                mj.mjv_initGeom(scn.geoms[idx], mj.mjtGeom.mjGEOM_SPHERE,
                                runner._trail_size, p, runner._mat_id, rgba)
                idx += 1
                added += 1
            if added < avail:   # current-body marker
                mj.mjv_initGeom(scn.geoms[idx], mj.mjtGeom.mjGEOM_SPHERE,
                                runner._marker_size, d.xpos[body_id], runner._mat_id, rgba)
                idx += 1
                added += 1
            if added >= avail:
                break
        scn.ngeom = idx
    return renderer.render()


def run_cmd_replay(runner, policy, obs, n, sampler, record_path=None):
    """Scripted training-matched replay: resample the velocity command like Isaac Lab
    ``UniformVelocityCommandCfg`` every few seconds, so the robot moves in ALL
    directions (incl. lateral vy) exactly like the training play video.  Optionally
    records an offscreen mp4 with the foot-end trail (``record_path``)."""
    import torch
    cfg = runner.cfg
    renderer, cam, writer = None, None, None
    if record_path:
        import mujoco as mj
        import imageio
        renderer = mj.Renderer(runner.model, width=640, height=480)
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = 135.0, -20.0, 1.7
        writer = imageio.get_writer(record_path, fps=1.0 / cfg.step_dt,
                                    macro_block_size=None, format="FFMPEG")
    for k in range(n):
        if not runner.is_trace:
            vx, vy, wz = sampler.step(k)
            runner.set_command(vx, vy, wz)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().reshape(1, -1)
            raw = policy(obs_t).cpu().numpy().reshape(-1)
        obs = runner.step_control(raw)
        if record_path:
            writer.append_data(_render_offscreen_scene(runner, renderer, cam))
        if k % 50 == 0:
            b = runner.data.xpos[runner.base_id]
            print(f"step {k}/{n}  base=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})  {runner.info_report()}")
    if writer is not None:
        writer.close()
    b = runner.data.xpos[runner.base_id]
    print(f"[sim2sim] cmd replay done. final base=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})")
    if record_path:
        print(f"[sim2sim] video -> {record_path}")


if __name__ == "__main__":
    main()
