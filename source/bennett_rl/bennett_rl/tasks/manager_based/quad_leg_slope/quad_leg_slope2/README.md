# QuadLeg Slope2

This task is the directional-slope counterpart of `quad_leg_free_gait2`.

- Robot, motor B (`8/20/19.896753`), actions, 33 policy observations,
  omnidirectional command sampler, FreeGait rewards, events, and PPO are
  inherited from FreeGait2.
- The smooth 0–6.5° directional lanes are reused from the older Slope1
  terrain, but with **finer low-end gradation** (0.25° steps) and a
  **longer flat approach + top platform**.
- Base-height reward and collapse termination use base clearance above the
  four feet so climbing elevation is not mistaken for height error.
- No crawl clock, prescribed foot order, desired-contact observation, or
  slope height scan is introduced.

## Changes from Slope1

| Item | Slope1 | Slope2 | Rationale |
|------|--------|--------|-----------|
| Slope ladder | 0.0, 0.7, 1.4, …° | 0.0, 0.25, 0.50, …° | Finer steps so level 1 is nearly flat |
| Approach length | 1.20 m | 1.80 m | More room to stabilise before ramp |
| Top platform | 1.00 m | 1.50 m | More room to trigger upgrade |
| Terrain length | 5.0 m | 6.0 m | Accommodates longer approach + top |
| Spawn X | 0.65 m | 1.20 m | Further forward on approach |
| Min clearance | 0.20 m | 0.15 m | More forgiving on slope |
| Speed range | 0.16–0.24 m/s | 0.10–0.20 m/s | Slower = easier start |
| base_height_l2 weight | –8.0 | –6.0 | Less pressure on height |
| leg_lift_starvation weight | –0.40 | –0.25 | More room for balance |
| Curriculum threshold | size[0]/2 | size[0]/3 | Easier to upgrade |
| PPO entropy | 0.01 | 0.02 | More exploration |
| PPO learning rate | 3e-4 | 2e-4 | More stable updates |
| PPO epochs | 5 | 4 | Less overfit to flat data |

## Training

```powershell
D:\Conda\envs\env_isaaclab\python.exe D:\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py `
  --task Isaac-BennettRL-QuadLeg-Slope2-v0 `
  --headless
```

## Play

```powershell
D:\Conda\envs\env_isaaclab\python.exe D:\IsaacLab\scripts\reinforcement_learning\rsl_rl\play.py `
  --task Isaac-BennettRL-QuadLeg-Slope2-Play-v0
```

The `...-Flat-v0` and `...-Flat-Play-v0` registrations provide an isolated
flat-terrain smoke test with the same policy contract.
