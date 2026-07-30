# QuadLeg Slope3

Slope3 is a standalone Bennett training task intended to learn from random
weights. It does not inherit any Bennett Trot, FreeGait, or Slope environment
or PPO configuration.

- Eight-joint Bennett V6 robot configuration is defined locally.
- Motor contract is `8 Nm` continuous, `20 Nm` peak, `19.896753 rad/s`,
  `Kp=28`, `Kd=2`.
- Policy uses 50 reproducible values and 8 position-residual actions. The
  additional 17 values are global/leg trot phase, desired foot contacts, and
  speed-conditioned gait parameters.
- Commands always request world-+X uphill progress with heading correction.
- Terrain starts at the zero-degree row and advances from 0 to 6 degrees only
  after sufficient signed uphill distance.
- A task-local soft diagonal-trot schedule pairs `FL+RR` and `FR+RL`.
  Frequency varies with speed from `0.75` to `1.25 Hz`; the uphill duty factor
  varies from `0.68` to `0.60`, and swing height is `0.035 m`.
- The schedule shapes contact and foot clearance but never terminates an
  episode for a phase mismatch, so the policy may extend support on the ramp.
- No old policy or checkpoint is required.

Train from scratch:

```powershell
cd E:\Project\Isaaclab\bennett_rl

D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\rsl_rl\train.py `
  --task Isaac-BennettRL-QuadLeg-Slope3-v0 `
  --headless
```

Flat smoke test:

```powershell
D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\random_agent.py `
  --task Isaac-BennettRL-QuadLeg-Slope3-Flat-Play-v0
```

Play a trained slope checkpoint:

```powershell
D:\Conda\envs\env_isaaclab\python.exe -B .\scripts\rsl_rl\play.py `
  --task Isaac-BennettRL-QuadLeg-Slope3-Play-v0 `
  --checkpoint <absolute-model-path>
```
