"""Task-local MDP terms for the higher-clearance free-gait Bennett task.

Everything else is the canonical Isaac Lab velocity-locomotion namespace, plus
two task-local rewards that together are the whole point of free_gait4 (higher
swing = survives the sim->real lift drop, without letting any leg glue to the
floor): ``swing_foot_clearance`` and the anti-glue ``feet_stance_time``.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .rewards import feet_stance_time, swing_foot_clearance

__all__ = ["swing_foot_clearance", "feet_stance_time"]
