"""Directional ramp terrain used by the Bennett slope task."""

from __future__ import annotations

import math

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return 3.0 * u**2 - 2.0 * u**3


def _smoothstep_integral(u: np.ndarray) -> np.ndarray:
    return u**3 - 0.5 * u**4


def _make_profiled_lane_mesh(
    x_positions: np.ndarray, z_heights: np.ndarray, y0: float, y1: float, depth: float,
) -> trimesh.Trimesh:
    vertices: list[list[float]] = []
    for x, z in zip(x_positions, z_heights):
        vertices.extend([[x, y0, z], [x, y1, z], [x, y0, -depth], [x, y1, -depth]])
    faces: list[list[int]] = []
    for i in range(len(x_positions) - 1):
        a, b = 4 * i, 4 * (i + 1)
        faces.extend([[a, b, b + 1], [a, b + 1, a + 1]])
        faces.extend([[a + 2, b + 3, b + 2], [a + 2, a + 3, b + 3]])
        faces.extend([[a, a + 2, b + 2], [a, b + 2, b]])
        faces.extend([[a + 1, b + 1, b + 3], [a + 1, b + 3, a + 3]])
    last = 4 * (len(x_positions) - 1)
    faces.extend([[0, 1, 3], [0, 3, 2]])
    faces.extend([[last, last + 2, last + 3], [last, last + 3, last + 1]])
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def directional_slope_terrain(
    difficulty: float, cfg: DirectionalSlopeTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    slope_angle = cfg.slope_angle_range[0] + difficulty * (cfg.slope_angle_range[1] - cfg.slope_angle_range[0])
    slope_ratio = math.tan(slope_angle)
    total_length = cfg.size[0]
    width = cfg.size[1]
    center_y = width / 2.0
    lane_width = width if cfg.lane_width is None else cfg.lane_width
    ramp_start_x = cfg.approach_length
    ramp_end_x = total_length - cfg.top_platform_length
    ramp_length = ramp_end_x - ramp_start_x
    transition_length = min(cfg.transition_length, ramp_length * 0.25)
    transition_segments = max(int(cfg.transition_segments), 2)
    top_height = slope_ratio * ramp_length
    effective_slope = top_height / max(ramp_length - transition_length, 1.0e-6)

    def height_at_x(x_values: np.ndarray) -> np.ndarray:
        heights = np.zeros_like(x_values)
        if transition_length <= 1.0e-6:
            return slope_ratio * np.clip(x_values - ramp_start_x, 0.0, ramp_length)
        bottom_end_x = ramp_start_x + transition_length
        top_start_x = ramp_end_x - transition_length
        bottom_height = 0.5 * effective_slope * transition_length
        bm = (x_values >= ramp_start_x) & (x_values < bottom_end_x)
        u_b = (x_values[bm] - ramp_start_x) / transition_length
        heights[bm] = effective_slope * transition_length * (u_b**3 - 0.5 * u_b**4)
        mm = (x_values >= bottom_end_x) & (x_values < top_start_x)
        heights[mm] = bottom_height + effective_slope * (x_values[mm] - bottom_end_x)
        tm = (x_values >= top_start_x) & (x_values < ramp_end_x)
        u_t = (x_values[tm] - top_start_x) / transition_length
        middle_height = bottom_height + effective_slope * max(top_start_x - bottom_end_x, 0.0)
        heights[tm] = middle_height + effective_slope * transition_length * (u_t - (u_t**3 - 0.5 * u_t**4))
        heights[x_values >= ramp_end_x] = top_height
        return heights

    x_samples = [
        0.0, ramp_start_x,
        *np.linspace(ramp_start_x, ramp_start_x + transition_length, transition_segments + 1)[1:].tolist(),
        ramp_end_x - transition_length,
        *np.linspace(ramp_end_x - transition_length, ramp_end_x, transition_segments + 1)[1:].tolist(),
        total_length,
    ]
    x_positions = np.asarray(sorted(set(round(float(x), 6) for x in x_samples)), dtype=np.float64)
    z_heights = height_at_x(x_positions)
    y0 = center_y - lane_width / 2.0
    y1 = center_y + lane_width / 2.0
    mesh = _make_profiled_lane_mesh(x_positions, z_heights, y0, y1, cfg.terrain_depth)
    return [mesh], np.array([cfg.spawn_x, center_y, 0.0])


@configclass
class DirectionalSlopeTerrainCfg(SubTerrainBaseCfg):
    function = directional_slope_terrain
    slope_angle_range: tuple[float, float] = (0.0, math.radians(6.0))
    approach_length: float = 1.20
    top_platform_length: float = 1.00
    spawn_x: float = 0.65
    terrain_depth: float = 0.50
    lane_width: float | None = None
    transition_length: float = 0.25
    transition_segments: int = 8
