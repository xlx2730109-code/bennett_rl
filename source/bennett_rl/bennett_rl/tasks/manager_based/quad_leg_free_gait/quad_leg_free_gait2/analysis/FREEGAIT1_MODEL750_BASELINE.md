# FreeGait1 model_750 per-leg failure baseline

Source run:

`logs/rsl_rl/quad_leg_free_gait/quad_leg_free_gait1/flat/2026-07-29_02-39-24/model_750.pt`

Deterministic setup:

- seed 42;
- 0.3 s settle, 2.0 s command and 0.2 s recovery per scenario;
- forward/backward at 0.18 m/s;
- left/right lateral at 0.12 m/s;
- left/right yaw at 0.35 rad/s;
- valid lift threshold 0.020 m.

Aggregate moving-scenario result:

| Leg | Valid lift events | Maximum relative lift | Mean contact ratio | Worst stance-slide P95 |
|---|---:|---:|---:|---:|
| FL | 0 | 0.76 cm | 90.8% | 0.489 m/s |
| FR | 0 | 0.27 cm | 95.2% | 0.916 m/s |
| RL | 13 | 4.01 cm | 19.0% | 0.153 m/s |
| RR | 1 | 4.18 cm | 83.7% | 0.043 m/s |

The single RR event occurred only in yaw-left. In forward, backward, and both
lateral scenarios, RL was the only leg to cross the 2 cm valid-lift threshold.
This confirms the one-leg clearance exploit and provides the numerical
baseline for the FreeGait2 100–150 iteration screen.
