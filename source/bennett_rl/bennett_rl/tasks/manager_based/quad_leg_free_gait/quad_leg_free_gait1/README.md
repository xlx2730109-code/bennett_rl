# QuadLeg FreeGait1

`FreeGait1` is an isolated flat-ground Bennett locomotion experiment. It asks
the policy to discover its own contact sequence while covering the commands
used by deployment controls.

## What is intentionally absent

- no phase clock;
- no desired-contact observation;
- no prescribed diagonal pair or leg order;
- no fixed gait frequency or duty factor;
- no trot contact-matching reward.

The policy observation has 33 hardware-observable values:

`base_ang_vel(3) + projected_gravity(3) + command(3) + joint_pos(8) + joint_vel(8) + last_action(8)`.

## What remains constrained

- Motor B contract: 8 Nm continuous effort, 20 Nm transient saturation and
  19.896753 rad/s velocity limit.
- At least two feet should support the body while moving. The reward does not
  specify which two feet.
- Feet in actual contact are discouraged from sliding.
- Whichever feet the policy chooses to swing receive a mild 0.035 m clearance
  target.
- Target tracking, body stability, action smoothness, touchdown impact and
  standstill behavior remain active.

## Command distribution

Commands are sampled as explicit modes rather than independent uniform axes:

| Mode | Probability |
|---|---:|
| stand | 0.10 |
| forward | 0.16 |
| backward | 0.16 |
| lateral left | 0.11 |
| lateral right | 0.11 |
| yaw in place | 0.12 |
| forward + yaw | 0.10 |
| backward + yaw | 0.08 |
| lateral + yaw | 0.06 |

Ranges are `vx=[-0.35, 0.35] m/s`, `vy=[-0.20, 0.20] m/s`, and
`yaw=[-0.60, 0.60] rad/s`.

## Training

```powershell
D:\Conda\envs\env_isaaclab\python.exe scripts\rsl_rl\train.py `
  --task Isaac-BennettRL-Flat-QuadLeg-FreeGait1-v0 `
  --headless
```

For the first assessment, stop at 300 iterations and compare:

1. episode length and termination causes;
2. forward, backward, lateral and yaw command errors separately;
3. base tilt and roll/pitch angular velocity;
4. target-step and action second-difference percentiles;
5. touchdown force, stance slip and number of supporting feet;
6. the learned footfall pattern, without assuming that trot must win.

Continue to 600 iterations only if locomotion and command coverage are still
improving. A lower aggregate reward is not by itself evidence that a
non-prescribed gait is worse than Trot1.
