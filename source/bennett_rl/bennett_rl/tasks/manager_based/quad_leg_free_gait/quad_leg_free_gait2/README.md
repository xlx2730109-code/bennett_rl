# QuadLeg FreeGait2

FreeGait2 is a controlled follow-up to FreeGait1. It preserves the complete
FreeGait1 plant, commands, observations, actions, PPO settings, randomization,
and existing rewards.

The only training change is `leg_lift_starvation`.

## Why

FreeGait1 can obtain full instantaneous clearance credit by repeatedly lifting
only one foot because its clearance score is averaged over whichever feet
happen to be airborne. The other three feet can remain in contact and the
two-foot minimum-support term remains inactive.

## Per-leg participation rule

For each of `FL`, `FR`, `RL`, and `RR`, the environment tracks the time since
that foot last:

1. lost contact;
2. reached at least 0.020 m above the current supporting-foot plane.

While moving, a leg is penalized only after its individual timer exceeds:

- 1.20 s at low equivalent command speed;
- 0.65 s at high equivalent command speed;
- a continuous interpolation between those limits.

The term is disabled while standing. It does not define foot order, diagonal
pairs, phase, duty factor, or a fixed gait frequency.

## First screen

Train for 100 to 150 iterations before committing to a long run. A checkpoint
passes the first screen only if:

- all four legs produce valid lift events in deterministic forward, backward,
  lateral, and yaw tests;
- no leg remains continuously planted throughout a moving segment;
- velocity tracking remains useful;
- episode length is not collapsing.

Use `analysis/collect_leg_diagnostics.py` after a checkpoint is available.

Run it from the Bennett RL project root:

```powershell
Set-Location E:\Project\Isaaclab\bennett_rl

D:\Conda\envs\env_isaaclab\python.exe -B `
  .\source\bennett_rl\bennett_rl\tasks\manager_based\quad_leg_free_gait\quad_leg_free_gait2\analysis\collect_leg_diagnostics.py `
  --checkpoint "<absolute-checkpoint-path>" `
  --output_csv "<new-output-path>.csv" `
  --headless
```

The collector refuses to overwrite an existing CSV. It writes both raw
policy-rate samples and a sibling `_summary.csv`.
