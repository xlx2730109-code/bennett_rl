# Copyright (c) 2026, Bennett. All rights reserved.

"""Bennett on the official go2 rough-terrain template.

Environment recipe mirrors ``go2/rough_env_cfg.py`` (the official Isaac Lab
rough-terrain template: pyramid stairs up/down, box grid, random rough, and
pyramid slopes, trained in parallel with terrain-level curriculum).  The only
structural change is the robot swap: Unitree Go2 -> Bennett.

This task is a motor-selection data source (plan.md step 4), not a deployment
target: torque limits are opened to +-20 N-m so the recorded joint torque can
exceed the 8 N-m continuous rating and reveal the terrain's true demand.

Bennett deltas from the go2 template (everything else stays the template):
  * effort/saturation opened to +-20 N-m (motor-selection decision).
  * command ranges scaled to Bennett's speed envelope (go2's +-1.0 m/s is
    unreachable); heading control from the template is kept.
  * feet_air_time threshold 0.5 -> 0.25 s (Bennett stays airborne less).
  * tracking kernels narrowed (std 0.5 -> 0.15 lin / 0.25 ang) plus fine
    companion kernels and a stronger air-time term.  This is the one
    Bennett-scale delta the template cannot absorb: with Bennett's +-0.35 m/s
    commands the template's std=0.5 kernel pays ~78% of the tracking reward
    for standing still, so stance is near-optimal and the policy never walks
    (quad_leg_go2-13 solved the identical problem the same way).
  * one velocity-solver iteration (V5 default 0 under-solves joint damping
    in PhysX; value proven in quad_leg_go2-13).

Second round (run 2026-09-06_00-03-58 showed tracking learning but the
terrain curriculum frozen at level 0: the robot walked 3-3.6 m per 20 s
episode, inside the 2.5-4 m no-promotion band of terrain_levels_vel):
  * lin_vel_x range -0.35 -> -0.50 m/s (raise the distance per episode past
    the 4 m move-up line),
  * feet_air_time threshold 0.25 -> 0.15 s (Bennett's normal swing time is
    0.15-0.2 s; the 0.25 threshold kept the term negative while stepping),
  * reset_base offset zeroed (episodes start on the patch origin; the
    curriculum still moves envs across patches).

Third round (2026-09-06): robot asset switched V5 -> V1 (Urdf_Bennett_1 --
the build whose training behaviour the template was originally proven on).
Asset-sync deltas this swap brings:
  * the foot-tip bodies on this USD are the _1 links (FL_1/FR_1/RL_1/RR_1,
    per quad_leg_go2-7), not the *_foot links of Urdf_Bennett_3 -- the
    feet_air_time sensor_cfg is renamed accordingly,
  * gains revert to the V1 asset values (kp 40 / kd 1.5 / +-20 effort
    override kept for the motor-selection mission).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V1


@configclass
class BennettGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Bennett running the official go2 rough-terrain recipe."""

    def __post_init__(self):
        # parent post-init: terrain curriculum on, height scanner, contact
        # sensor, canonical reward set, push robot, etc.
        super().__post_init__()

        # ------------------------------------------------------------------
        # Robot swap: the only structural change vs the go2 template
        # ------------------------------------------------------------------
        robot_cfg = BENNETT_CFG_V1.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Motor-selection run: open the torque ceiling to +-20 N-m (the
        # DM8006 peak) so stairs/slopes can demand more than the 8 N-m
        # continuous rating and the data shows the true requirement.
        robot_cfg.actuators["base_legs"].effort_limit = 20.0
        robot_cfg.actuators["base_legs"].saturation_effort = 20.0
        # one velocity-solver iteration: the V5 default 0 under-solves joint
        # damping in PhysX (value proven in quad_leg_go2-13).
        robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
        self.scene.robot = robot_cfg

        # Bennett's USD carries 16 joints: the 8 actuated thigh/calf joints
        # plus 8 passive parallel-four-bar chain joints.  The template's
        # default joint_names=".*" (fine for the fully-actuated go2) would
        # turn the passive chain joints into policy-driven targets, fighting
        # the closed-chain constraint.  Restrict actions and joint
        # observations to the 8 actuated joints, matching the actuator
        # definition above.
        self.actions.joint_pos.joint_names = [".*_thigh", ".*_calf"]
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_thigh", ".*_calf"]
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_thigh", ".*_calf"]
        )

        # Height scanner attaches to the Bennett base body (go2 template does
        # the same for its base link).
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # ------------------------------------------------------------------
        # Terrain: go2's small-robot scaling of the shared preset
        # ------------------------------------------------------------------
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------------------------------------------
        # Actions (go2 template scale + Bennett joint-range clamp)
        # ------------------------------------------------------------------
        self.actions.joint_pos.scale = 0.25
        # Bennett's joints have a much smaller range than go2's; clamp the
        # offset targets so an unbounded policy output cannot command the
        # joints past their limits.
        self.actions.joint_pos.clip = {
            ".*_thigh": (-0.80, 0.80),
            ".*_calf": (-0.90, 0.55),
        }

        # ------------------------------------------------------------------
        # Commands: template heading control, ranges scaled to Bennett
        # ------------------------------------------------------------------
        # +-0.5 m/s (second run 2026-09-06): the +-0.35 envelope left the
        # terrain curriculum frozen -- the robot tracked at ~0.15-0.2 m/s,
        # covering 3-3.6 m per 20 s episode, stuck between the move_down
        # line (0.5*|cmd|*20s = 2.5-4 m) and the move-up line (4 m).  The
        # proven bennett_test4_go2 recipe kept the template's +-1.0 and let
        # the wide kernel force locomotion; +-0.5 keeps Bennett's envelope
        # while restoring a walk-or-starve gradient.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.50, 0.50)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.60, 0.60)

        # ------------------------------------------------------------------
        # Events (go2 template)
        # ------------------------------------------------------------------
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_base.params = {
            # zero reset offset (bennett_test4_go2 proven): every episode
            # starts exactly on the patch origin so early learning is not
            # wasted on recovering from random spawns; the terrain curriculum
            # still moves envs across patches through env_origins.
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_com = None

        # ------------------------------------------------------------------
        # Rewards (go2 template weights, Bennett-scale tracking kernels)
        # ------------------------------------------------------------------
        # Urdf_Bennett_1 carries no separate foot links: the foot tip body is
        # the _1 link (FL_1/FR_1/RL_1/RR_1, as in quad_leg_go2-7).  The
        # template's ".*_foot" (and V5's "*_foot") match nothing on this USD.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [
            "FL_1",
            "FR_1",
            "RL_1",
            "RR_1",
        ]
        # Bennett has no hip abduction; the template's 0.01 gives the feet no
        # lift incentive and the policy drags them across rough terrain.
        self.rewards.feet_air_time.weight = 0.05
        # 0.15 s (second run 2026-09-06, was 0.25): Bennett's normal step
        # frequency puts swing time around 0.15-0.2 s, so the 0.25 threshold
        # kept the term negative even while stepping.
        self.rewards.feet_air_time.params["threshold"] = 0.15
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # The template's kernels (std = sqrt(0.25) = 0.5) pay ~78% of the
        # tracking reward for standing still at Bennett's +-0.35 m/s command
        # envelope, so stance is near-optimal and the first run (2026-09-05)
        # converged to standing: feet_air_time ~ 0, terrain curriculum
        # collapsed to level 0.  Narrow the kernels so only real walking
        # earns them, and add the narrow "fine" companions that break the
        # standing-vs-tracking tie (both proven on Bennett in quad_leg_go2-13).
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.15
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.track_lin_vel_xy_fine_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=0.8,
            params={"std": 0.05, "command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_fine_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.6,
            params={"std": 0.10, "command_name": "base_velocity"},
        )

        # ------------------------------------------------------------------
        # Terminations (go2 template body name)
        # ------------------------------------------------------------------
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class BennettGo2RoughEnvCfg_PLAY(BennettGo2RoughEnvCfg):
    """Play variant of the Bennett rough-terrain environment (go2 template)."""

    def __post_init__(self):
        # play mode: 50 envs in a 5x5 grid without terrain curriculum,
        # deterministic observations, and no domain randomization.
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
