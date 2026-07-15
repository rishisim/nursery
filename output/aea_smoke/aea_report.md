# Nursery AEA real-data phase report

**Status:** infrastructure_smoke_test_not_a_real_data_finding

Data represented in this run: 96 windows from 12 recordings across 3 locations.

AEA is an adult, partly scripted sensor-format analogue. It is not developmental evidence and is not represented as BabyView-matched.

## Primary question

Does correctly synchronized six-axis head IMU during training improve motor-withheld video-language action grounding relative to split-local, whole-episode-shuffled IMU?

Primary synchronized − episode-shuffled estimate: **+1.04 pp (95% CI +0.00, +2.08)**.

Claim gate: **not passed**.

## Protocol safeguards

- Accelerometer and gyroscope are resampled together on a fixed grid in SI units.
- Whole windows stay with their source sequence; concurrent views are grouped or purged.
- Shuffled IMU donors are split-local, come from a different performed-event group (including concurrent partner recordings), and form a permutation.
- Seeds, initialization, batch order, optimizer schedule, and test inputs are paired across arms.
- Locked training configuration match: False (expected for this smoke configuration).
- The primary evaluator never loads IMU or calls the motor encoder.

## Evaluation families

| Family | Splits run | Synchronized − shuffled |
|---|---:|---:|
| held_out_location | 2 | +1.04 pp (95% CI +0.00, +2.08) |

## Interpretation limits

Action labels are noisy ASR lexical anchors, object labels are nearby ASR lexical items, and the wearer split uses a release-visible location+script+recording proxy rather than persistent identity. Smoke-test numbers validate plumbing only. Real-data estimates are findings only when the report status explicitly says so and all audit gates pass.
