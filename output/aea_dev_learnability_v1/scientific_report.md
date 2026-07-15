# AEA development learnability decision

## Technical summary: REVISE, with no confirmation or locked run authorized

The frozen v1 development protocol yields **REVISE**. The current ASR-anchored task failed its visual-validity gate, the video-text learner failed both the capacity control and every held-out grounding gate, and synchronized head IMU did not predict held-out action labels above permutation-calibrated chance. In addition, the preregistered paired synchronized-versus-shuffled IMU test is invalid because several fixed held-out folds cannot form the required whole-window, different-event-group donor bijection.

Under the frozen rule, that integrity failure forces REVISE rather than a scientific STOP. This recommendation does **not** authorize inspecting the 55-window prospective reserve, running the locked five-fold/seven-seed/eight-epoch experiment, or contacting Professor Frank. It authorizes only a prospectively frozen v2 on development data that replaces noisy ASR anchors with human wearer-action annotations, repairs donor-feasible group folding, and demonstrates a passing capacity control.

AEA is an adult, partly scripted sensor-format analogue. These results are not developmental evidence and are not represented as BabyView-like.

## Scope and partition remained development-only

The source contains 247 accepted windows. Before any new audit labels or model outcomes, five performed-event groups—55 windows and six recordings, with concurrent views kept together—were frozen as prospective confirmation. The other 192 windows in 18 performed-event groups formed development. The earlier reduced infrastructure smoke had aggregated all records, so this reserve is prospective only from this iteration onward, not pristine in an absolute sense.

The development materialization matched the partition exactly. Confirmation ID overlap, confirmation event-group overlap, fold event-group overlap, and confirmation file accesses were all zero. See `preregistered_protocol.json`, `partition_manifest.json`, `development_examples.receipt.json`, and `commands_and_provenance.json`.

## ASR anchors were usually not visible wearer actions

The deterministic audit sampled 48 development windows across 25 actions, all 18 development event groups, and all five locations. All eight frames spanning each six-second window were reviewed.

| Audit category | Count | Rate | 95% Wilson interval |
|---|---:|---:|---:|
| Clear match | 7 | 14.58% | 7.25%–27.17% |
| Plausible or ambiguous | 7 | 14.58% | — |
| Clear or plausible | 14 | 29.17% | 18.24%–43.18% |
| Mismatch | 34 | 70.83% | 56.82%–81.76% |
| Not visually judgeable | 0 | 0% | — |

The preregistered audit gate required the lower interval bound for clear-or-plausible to be at least 60% and the upper mismatch bound to be at most 25%; both failed by a wide margin. Common failure modes in `audit_labels.json` include speech describing another person, figurative uses such as “read my mind,” nouns such as “cutting board,” and action words from phone/video narration. The predeclared semantic high-motion set was not rescued: only 5 of 18 audited windows were clear or plausible, with 13 mismatches.

## Video-language grounding did not pass learnability or capacity checks

The video-text learner used motor-free training and evaluation, three frozen seeds, five event-group-safe folds, 30 epochs, and chance-calibrated action-balanced 2AFC. The corrected coarse endpoint includes all five predeclared categories and all 192 development windows.

| Endpoint | Labels / windows | Train 2AFC | Held-out 2AFC | Event-group 95% CI | Permutation p | Gate |
|---|---:|---:|---:|---:|---:|---:|
| Fine actions | 7 / 132 | 57.14% | 56.69% | 48.14%–62.69% | 0.1429 | Fail |
| Coarse categories | 5 / 192 | 57.14% | 46.53% | 36.89%–61.94% | 0.7466 | Fail |
| Semantic high-motion fine | 3 / 67 | 49.78% | 45.08% | 34.49%–54.40% | 0.8211 | Fail |

The easier 64-window/120-epoch capacity control reduced video-text loss by 39.20% but reached only 61.85% training 2AFC, below the frozen 90% gate. This means v1 cannot cleanly distinguish an intrinsically unlearnable visual task from insufficient learner capacity/optimization. The raw per-action, fold, seed, score, loss, bootstrap, and permutation records are retained in `aea_dev_results.json`.

## Head IMU showed overfit, not held-out action information

The fixed 129-dimensional six-axis feature pipeline and train-fold-standardized multinomial logistic regression used the same group-safe folds. Synchronized IMU versus chance is descriptive because the paired shuffled endpoint was invalid before IMU loading.

| Endpoint | Train balanced accuracy | Held-out balanced accuracy | Event-group 95% CI | Permutation p |
|---|---:|---:|---:|---:|
| Fine actions | 98.37% | 12.03% | 6.01%–18.62% | 0.6932 |
| Coarse categories | 93.24% | 15.20% | 10.05%–19.98% | 0.9365 |
| Semantic high-motion fine | 100.00% | 29.80% | 16.84%–41.67% | 0.5657 |

The large train/held-out gaps show event-specific overfit. None of the held-out synchronized estimates exceeded its fold-preserving permutation null. These descriptive nulls do not substitute for the paired estimand.

For every endpoint, at least one fixed held-out side could not form a bijective shuffled donor map with no self or same-performed-event-group donor: fine folds 0, 1, and 4; coarse folds 0, 2, and 3; and semantic-high-motion folds 0, 1, and 3. No donor reuse, fold change, or endpoint relaxation was allowed. Therefore synchronized-minus-shuffled lift, confidence interval, and randomization p-value are intentionally absent and every paired IMU gate is invalid. The full feasibility matrix and synchronized predictions are in `imu_results.json`.

## Leakage and manipulation checks

Passed checks include exact source/partition membership, zero reserve overlap, every endpoint-eligible row held out exactly once, zero train/test event-group overlap, motor-free grounding with zero IMU file accesses, finite fixed-dimension IMU features, and training-fold-only feature standardization. The hard aggregate fails only because the required shuffled donor bijections are not feasible; donor checks are marked unavailable/failed rather than fabricated.

The first combined runner output is preserved as `raw_results_attempt1_invalid_coarse_support.json` and excluded from conclusions because it incorrectly inherited fine-action support for coarse categories. A regression test now requires all five coarse categories and 192 coarse rows. `raw_results_v2.json` and its identical canonical copy `aea_dev_results.json` are the valid result.

## Rule-based recommendation and next authorized work

**REVISE.** The frozen rule gives hard integrity and failed capacity checks precedence. Independently, the audit provides strong evidence that current ASR lexical anchors are not acceptable wearer-action labels; no coarse or semantic high-motion rescue passed.

A v2 development protocol may be drafted only if it prospectively specifies:

1. human annotations of visible wearer action, agency, and temporal alignment, with a reliability check;
2. group folds constrained for donor feasibility or another fully group-safe manipulation defined before outcomes;
3. an easier visual/action capacity control that passes before held-out claims;
4. the same confirmation reserve exclusion until a revised protocol passes development gates.

If human wearer-action annotation and a valid paired manipulation are not feasible, the appropriate next decision is STOP for the current AEA ASR-anchor task. The present evidence does not support the expensive locked run or external outreach.

## Limitations

- The audit used one auditor and eight sparse frames rather than continuous video; ambiguous cases were retained explicitly.
- The prospective reserve was included in an earlier infrastructure smoke, though it was untouched in this iteration.
- The paired IMU estimand is invalid under v1, so no synchronized-minus-shuffled scientific estimate exists.
- The corrected MPS run is reproducible from saved code, seeds, raw predictions, and provenance, but Apple MPS training is not guaranteed bitwise deterministic; only the corrected v2 output is used.
- ASR lexical anchors are noisy labels, not human action annotations.

