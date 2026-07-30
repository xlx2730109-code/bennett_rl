# QuadLeg Slope4

Slope4 is a standalone Bennett training task intended to learn from random
weights. It does not inherit any Bennett Trot, FreeGait, or Slope environment
or PPO configuration.

- Eight-joint Bennett V6 robot configuration is defined locally.
- Motor contract is `8 Nm` continuous, `20 Nm` peak, `19.896753 rad/s`,
  `Kp=28`, `Kd=2`.
- Policy uses 50 reproducible values and 8 position-residual actions. The
  additional 17 values are global/leg trot phase, desired foot contacts, and
  speed-conditioned gait parameters.
- Commands always request world-+X uphill progress with heading correction.
- Terrain has ten global levels from `0.0` to `13.5 deg` in `1.5 deg`
  increments.
- A level passes only when at least 75% of a validation batch of at least 512
  completed episodes holds all four feet on the upper platform for `0.25 s`.
  Failed validation batches repeat the same level.
- TensorBoard reports discrete curriculum state under
  `Curriculum/level_progress/*`; `validated_pass_level` is the highest level
  actually passed and starts at `-1`.
- A task-local soft diagonal-trot schedule pairs `FL+RR` and `FR+RL`.
  Frequency varies with speed from `0.65` to `1.05 Hz`; the uphill duty factor
  varies from `0.68` to `0.60`, and swing height is `0.045 m`.
- At the `0.20 m/s` command midpoint the schedule is `0.85 Hz`, about 15%
  slower and 18% longer per cycle than the previous `1.00 Hz` schedule.
- A body-frame front/rear stance-width penalty allows `0.40/0.39 m` (neutral
  is about `0.36 m`) and keeps a live gradient across larger unsafe spreads.
  It does not prescribe individual footholds.
- A worst-swing-foot clearance shortfall term requires every scheduled foot
  to reach at least 75% of the existing smooth `0.045 m` profile. This closes
  the mean-reward loophole where one diagonal foot could remain planted while
  its partner lifted correctly.
- Swing height is measured above the exact local smooth-ramp profile. Merely
  moving a foot uphill no longer earns false foot-clearance reward.
- Compared with Slope3, action-rate, action-curvature and joint-acceleration
  penalties are moderately stronger; PPO starts at `log_std=log(0.6)` with
  entropy coefficient `0.003`.
- The schedule shapes contact and foot clearance but never terminates an
  episode for a phase mismatch, so the policy may extend support on the ramp.
- No old policy or checkpoint is required.

Train from scratch:

```powershell
cd E:\Project\Isaaclab\bennett_rl

D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\rsl_rl\train.py `
  --task Isaac-BennettRL-QuadLeg-Slope4-v0 `
  --headless
```

Flat smoke test:

```powershell
D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\random_agent.py `
  --task Isaac-BennettRL-QuadLeg-Slope4-Flat-Play-v0
```

Play a trained slope checkpoint:

```powershell
D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\rsl_rl\play.py `
  --task Isaac-BennettRL-QuadLeg-Slope4-Play-v0 `
  --checkpoint <absolute-model-path>
```
