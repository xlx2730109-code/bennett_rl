"""Straight, forward-only staircase terrain for Bennett Stair1."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def ascending_stairs_terrain(
    difficulty: float, cfg: "AscendingStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat approach, ascending treads, and a top platform.

    Difficulty is quantized onto ``height_levels``. This makes the ten terrain
    rows physically exact rather than allowing Isaac Lab's within-row jitter
    to turn a nominal 1 cm row into a mixture of heights.
    """

    if not cfg.height_levels:
        raise ValueError("height_levels must not be empty.")
    if cfg.step_depth <= 0.0 or cfg.num_steps <= 0:
        raise ValueError("step_depth and num_steps must be positive.")
    if cfg.base_thickness <= 0.0:
        raise ValueError("base_thickness must be positive.")

    level = min(int(np.floor(float(difficulty) * len(cfg.height_levels))), len(cfg.height_levels) - 1)
    level = max(level, 0)
    step_height = float(cfg.height_levels[level])
    first_riser_x = float(cfg.spawn_x + cfg.approach_distance)
    stair_end_x = first_riser_x + cfg.num_steps * float(cfg.step_depth)
    if first_riser_x <= 0.0 or stair_end_x >= cfg.size[0]:
        raise ValueError(
            "Stair geometry does not fit inside the terrain: "
            f"first_riser={first_riser_x:.3f}, stair_end={stair_end_x:.3f}, size_x={cfg.size[0]:.3f}."
        )

    meshes: list[trimesh.Trimesh] = []
    base_dims = (cfg.size[0], cfg.size[1], cfg.base_thickness)
    base_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.base_thickness)
    meshes.append(
        trimesh.creation.box(base_dims, trimesh.transformations.translation_matrix(base_pos))
    )

    for step_index in range(cfg.num_steps):
        x_start = first_riser_x + step_index * cfg.step_depth
        top_height = (step_index + 1) * step_height
        box_height = top_height + cfg.base_thickness
        box_dims = (cfg.step_depth, cfg.size[1], box_height)
        box_pos = (
            x_start + 0.5 * cfg.step_depth,
            0.5 * cfg.size[1],
            0.5 * (top_height - cfg.base_thickness),
        )
        meshes.append(
            trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        )

    top_height = cfg.num_steps * step_height
    top_length = cfg.size[0] - stair_end_x
    top_dims = (top_length, cfg.size[1], top_height + cfg.base_thickness)
    top_pos = (
        stair_end_x + 0.5 * top_length,
        0.5 * cfg.size[1],
        0.5 * (top_height - cfg.base_thickness),
    )
    meshes.append(
        trimesh.creation.box(top_dims, trimesh.transformations.translation_matrix(top_pos))
    )

    # The robot starts here at ground height, facing world +X. The first riser
    # is ``approach_distance`` ahead of the origin.
    origin = np.array([cfg.spawn_x, 0.5 * cfg.size[1], 0.0], dtype=np.float64)
    return meshes, origin


@configclass
class AscendingStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for one straight ascending staircase lane."""

    function = ascending_stairs_terrain

    height_levels: tuple[float, ...] = tuple(0.01 * (index + 1) for index in range(10))
    step_depth: float = 0.30
    num_steps: int = 6
    spawn_x: float = 0.75
    approach_distance: float = 0.60
    base_thickness: float = 0.20

