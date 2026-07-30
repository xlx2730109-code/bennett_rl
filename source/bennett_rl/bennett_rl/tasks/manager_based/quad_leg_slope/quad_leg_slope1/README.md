# QuadLeg Slope1

This task is the directional-slope counterpart of `quad_leg_free_gait2`.

- Robot, motor B (`8/20/19.896753`), actions, 33 policy observations,
  omnidirectional command sampler, FreeGait rewards, events, and PPO are
  inherited from FreeGait2.
- The smooth 0–6 degree directional lanes are reused from the older Slope1
  terrain.
- Base-height reward and collapse termination use base clearance above the
  four feet so climbing elevation is not mistaken for height error.
- No crawl clock, prescribed foot order, desired-contact observation, or
  slope height scan is introduced.

Training:

```powershell
D:\Conda\envs\env_isaaclab\python.exe D:\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py `
  --task Isaac-BennettRL-QuadLeg-Slope1-v0 `
  --headless
```

The slope policy should not be started from random weights.  Create a
fresh-optimizer bootstrap checkpoint from the mature FreeGait2 policy:

```powershell
D:\Conda\envs\env_isaaclab\python.exe `
  .\source\bennett_rl\bennett_rl\tasks\manager_based\quad_leg_slope\quad_leg_slope1\analysis\bootstrap_from_free_gait2.py `
  --headless
```

Then start a new Slope1 run from that checkpoint:

```powershell
D:\Conda\envs\env_isaaclab\python.exe .\scripts\rsl_rl\train.py `
  --task Isaac-BennettRL-QuadLeg-Slope1-v0 `
  --resume `
  --load_run bootstrap_free_gait2_model2100 `
  --checkpoint model_0.pt `
  --headless
```

The bootstrap copies actor/critic weights, converts FreeGait2's `std` to
Slope1's `log_std`, starts at iteration zero, and intentionally uses a fresh
Slope1 optimizer.

Play:

```powershell
D:\Conda\envs\env_isaaclab\python.exe D:\IsaacLab\scripts\reinforcement_learning\rsl_rl\play.py `
  --task Isaac-BennettRL-QuadLeg-Slope1-Play-v0
```

The `...-Flat-v0` and `...-Flat-Play-v0` registrations provide an isolated
flat-terrain smoke test with the same policy contract.
