"""Task-local MDP terms for the higher-clearance free-gait Bennett task.

Everything else is the canonical Isaac Lab velocity-locomotion namespace, plus one
task-local reward: the one-sided saturating swing-foot clearance penalty that is
the whole point of free_gait4 (higher swing = survives the sim->real lift drop).
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .rewards import swing_foot_clearance

__all__ = ["swing_foot_clearance"]
