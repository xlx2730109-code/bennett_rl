"""Terrain-coordinate MDP terms for the FreeGait2 slope task."""

from .curriculums import uphill_terrain_levels
from .rewards import (
    base_clearance_above_feet_below_minimum,
    base_height_above_feet_l2,
)
