# DM-J8006 Trot Motor A/B Protocol

## Fixed variables

Both variants inherit `quad_leg_trot1`. Do not change rewards, observations,
commands, gait scheduling, action scaling, PD gains, events, terrain,
terminations, PPO settings, environment count, or seed between paired runs.

## Single changed category

| Variant | effort_limit | saturation_effort | velocity_limit |
|---|---:|---:|---:|
| A: current/older | 7 Nm | 12 Nm | 20 rad/s |
| B: datasheet candidate | 8 Nm | 20 Nm | 19.8967534727 rad/s |

The standard Isaac Lab `DCMotor` hard-clips applied torque at `effort_limit`.
The `saturation_effort` and `velocity_limit` values define its linear
torque-speed envelope. This experiment does not model temporary 20 Nm peak
duration or thermal derating.

## Evaluation stages

1. Create both environments with one environment and step finite zero/random
   actions. Reject either variant if imports, entity resolution, spawn,
   contact sensors, observations, rewards, or terminations fail.
2. Run a short paired training screen with the same seed and iteration count.
   Reject checkpoints that do not move in deterministic simulation, saturate
   actions excessively, or collapse/explode exploration.
3. Train paired full runs. If the result is close or seed-sensitive, run at
   least three matched seeds before claiming one model is better.
4. Evaluate deterministic policies with the same forward, backward, yaw, and
   stand command sequence. Compare tracking, falls, torque clipping, joint
   acceleration, touchdown impact, foot slide, and base angular motion.
5. Only after both policies pass simulation gates, perform conservative
   hardware tests. Use the same deployment contract and collect synchronized
   command, target/actual joint state, torque, IMU, temperature, voltage, and
   timing data. The comparison is incomplete if only one variant is tested on
   hardware.

## Decision rule

Training reward alone is not a winner criterion. Prefer the variant that:

- remains stable across matched seeds;
- tracks commands in deterministic simulation;
- reduces sim-to-real joint/base response error;
- does not increase torque spikes, acceleration, impact, or temperature;
- preserves the exact policy/action/observation/control-rate contract.
