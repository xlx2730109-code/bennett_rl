"""Task-local MDP terms for the canonical emergent-gait Bennett task.

The free-gait env uses the open-source Isaac Lab velocity-locomotion terms only
(canonical ``UniformVelocityCommandCfg`` command + the reward functions inherited
from ``LocomotionVelocityRoughEnvCfg``). No Bennett-specific command generator or
reward remains -- the gait is driven purely by the canonical ``feet_air_time``
bonus and the heading command, so nothing needs re-exporting here beyond the
upstream ``isaaclab.envs.mdp`` namespace.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
