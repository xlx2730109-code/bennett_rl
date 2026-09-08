"""Gait-agnostic foot-clearance rewards for the higher-clearance free-gait task.

Two terms, both motivated by the same sim2real symptom (the previously-trained
sim lift transferred to the real robot as a foot that dragged / shuffled) and
redesigned twice against observed training failures:

``swing_foot_clearance`` -- nudges each swinging foot to a generous,
speed-scaled apex height. Design history:
  * v1 charged the shortfall through the whole airborne arc -> a parabolic
    swing spends most of its arc below the apex target, ideal behavior still
    ate ~60% of the max penalty, and the term locked in a tug-of-war against
    the smoothness penalties (3x jerk, 6x below-height terminations).
  * v2 (mid-swing weighting, speed-scaled target, canonical 0.1 gate) fixed
    the arc tax but introduced a DIVISION-OF-LABOR loophole: the penalty was
    the MEAN over swinging feet, and a foot that never leaves the ground pays
    nothing. The trained policy made ONE leg the designated high-stepper
    (137 mm apexes, 62% airborne) while the other three glued to the floor
    (92.7% contact, 30 mm scuffs) -- gluing was free and a second low swing
    leg would have raised the mean.
  * v3 (current) closes the loophole:
      - normalized by the FIXED foot count (sum/4): every foot's swing
        quality counts independently, one performer cannot average the rest
        away;
      - adds a mild OVERSHOOT tax above a wide free band (h >
        over_free_mult * target): the saturating one-sided form made
        over-lifting free, and ``feet_air_time`` even paid for marathon
        swings -- the RL leg farmed both into violent 137 mm flails.

``feet_stance_time`` -- the anti-glue mirror of the official
``feet_air_time``: at the LIFT-OFF instant of each foot, penalize
``(last_contact_time - threshold)``. Its threshold scales with the commanded
speed (fast walk 0.25 s, crawl 0.6 s), the same duty-factor-with-speed
coupling the reference paper builds into its phase generator. Without this
term a never-swinging foot costs zero clearance, which is exactly the free
gluing v2 exploited.

Both terms read foot world-Z, contact forces and body velocities only -- pure
sim-terrain constructs. Neither adds anything to the frozen 33-dim observation
(no clock, no phase, no swing flags), so the obs contract and the deployment
bridge are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CANONICAL_VELOCITY_GATE = 0.1  # official feet_air_time's no-reward-for-zero-command gate


def _command_planar_speed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Planar commanded speed |(vx, vy)| per env (m/s)."""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, :2], dim=1)


def _foot_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(force, dim=-1).amax(dim=1) > float(threshold)


def swing_foot_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    clearance_low: float,
    clearance_high: float,
    speed_ref: float,
    v_foot_ref: float,
    saturating_k: float,
    over_free_mult: float,
    over_norm: float,
    command_name: str,
) -> torch.Tensor:
    """Mid-swing-weighted penalty for swinging feet that stay too low or flail.

    Returns a *positive* penalty magnitude in [0, 1] per env (summed over the
    feet, normalized by the FIXED foot count); attach it with a NEGATIVE weight
    (e.g. ``weight=-0.40``). A foot is taxed only while it is airborne AND
    translating (``|v_foot_xy| -> v_foot_ref``, the phase-free mid-swing
    weight): below the speed-scaled target through a saturating shortfall,
    above ``over_free_mult * target`` through a linear overshoot tax.
    """
    contact = _foot_contacts(env, sensor_cfg, threshold)
    contact_float = contact.to(torch.float32)
    in_swing = (~contact).to(torch.float32)

    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # world foot Z
    stance_count = contact_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    stance_height = (foot_height * contact_float).sum(dim=1, keepdim=True) / stance_count
    rel_height = torch.clamp(foot_height - stance_height, min=0.0)

    # target scales with the commanded planar speed: h_low at creep, h_high at nominal
    cmd_speed = _command_planar_speed(env, command_name)
    target = float(clearance_low) + (float(clearance_high) - float(clearance_low)) \
        * torch.clamp(cmd_speed / float(speed_ref), 0.0, 1.0)[:, None]

    shortfall = torch.clamp(target - rel_height, min=0.0)
    saturating = 1.0 - torch.exp(-float(saturating_k) * shortfall)
    # overshoot tax above a wide free band: over-lifting must cost something
    over = torch.clamp(rel_height - float(over_free_mult) * target, min=0.0) / float(over_norm)
    over = torch.clamp(over, max=1.0)

    # mid-swing weight: only a translating foot scuffs or flails (paper's w(s), no clock)
    foot_vxy = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    translating = torch.clamp(foot_vxy / float(v_foot_ref), 0.0, 1.0)

    penalty = (saturating + over) * in_swing * translating
    # fixed-foot-count normalization: one designated stepper cannot average
    # the glued feet's account away (v2's division-of-labor loophole)
    n_feet = float(contact.shape[1])

    has_support = (contact_float.sum(dim=1) > 0.0).to(torch.float32)
    moving = (cmd_speed > CANONICAL_VELOCITY_GATE).to(torch.float32)
    return moving * has_support * penalty.sum(dim=1) / n_feet


def feet_stance_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    stance_thr_fast: float,
    stance_thr_slow: float,
    speed_ref: float,
) -> torch.Tensor:
    """Anti-glue mirror of the official ``feet_air_time``: tax long STANCES.

    At each foot's lift-off instant, returns ``last_contact_time - threshold``
    (positive = planted too long), summed over the feet. The threshold scales
    with the commanded planar speed -- ``stance_thr_fast`` at ``speed_ref`` down
    to ``stance_thr_slow`` at the crawl gate -- mirroring the duty-factor/speed
    coupling of the reference paper, so legitimate slow walking is not taxed.
    Attach with a NEGATIVE weight (e.g. ``weight=-0.25``). Inactive below the
    canonical 0.1 m/s command gate, like the official ``feet_air_time``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd_speed = _command_planar_speed(env, command_name)

    # threshold: fast at nominal speed, relaxed toward the crawl gate
    thr = float(stance_thr_fast) + (float(stance_thr_slow) - float(stance_thr_fast)) \
        * torch.clamp(1.0 - cmd_speed / float(speed_ref), 0.0, 1.0)

    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    penalty = torch.sum((last_contact_time - thr[:, None]) * first_air, dim=1)
    # no penalty for (near-)zero command: standing is legal
    penalty *= cmd_speed > CANONICAL_VELOCITY_GATE
    return penalty
