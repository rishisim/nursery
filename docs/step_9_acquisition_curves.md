# Step 9: Acquisition curves by word type

Status: **draft protocol, awaiting review. No runs launched.**

## Question

How do gaze gain and body gain change with the amount of experience, for
nouns and for actions, under the fixed shared learner? Step 6 measured nouns
only, with static video and no body gain. Step 7 showed at 630 clips that the
body adds +11.4 pp on actions and nothing on nouns. Step 9 asks whether that
action gain holds, grows or shrinks across amounts, and whether `V` on actions
shows signs of a plateau that `V+G+B` lifts.

## Decisions already made

- **Learner:** Step 7 temporal video learner with per-word zero-init temporal
  weight (`temporal_weight_init="per_word"`), the default since 21 Nov 2026.
- **Step 8 skipped.** Revisit wordless pretraining only if `V+G+B` lags
  `V+G` on actions at low data (63–158 clips).
- **Rungs:** nouns and actions. Heavy/light stays a completed Step 7 proof of
  concept and is not scored in the curves (its panel may be logged for
  reference but does not enter selection).

## Arms

| Arm | Inputs | Clip counts |
|---|---|---|
| `V` | temporal video | all |
| `V+G` | `V` + steering gaze | all |
| `V+G+B` | `V+G` + current body, instance mass | all |
| `V+G+B` shuffled | body shuffled across clips, real gaze kept | 158 and 630 only |

**Shuffled body: key points only (recommended).** In Step 7 the shuffled arm
added −0.66 to 0.00 pp on every rung and dev turned its weight nearly off
(0.05–0.20). It is an alignment check, not a curve. 630 reproduces the Step 7
control under the fixed learner; 158 checks that any low-data body gain also
needs aligned body. 63 is skipped because its intervals will be widest. Add
more points only if real body shows a gain at a point where shuffled was not
run.

## Clip counts and seeds

- Step 6 ladder: nested seed-0 prefixes of the 630 training clips, **63, 158,
  315, 473, 630**; split seed 0; same test and s8 panels as Step 7.
- **Seeds 0–4 at every point**, so the 630 point is directly the clean
  fixed-learner Step 7 table (same 5 seeds, same panels, same arms), and low
  points get the same seed count as the top one.
- Per-amount fitting as in Step 6: vocabulary (min count 5) and body
  normalization use only that amount's clips.

**Pre-launch check:** count train utterances for the five action words (pass,
lift, inspect, offhand, drink) at each amount. If a word falls below min count
at 63 clips, it is absent from that model's vocabulary; report it as a miss
(chance on its trials), not drop its trials, so every point scores the same
panel.

## Dev selection and epoch cap

- **Selection score:** equal-weight mean of noun dev (`clips_dev`) and action
  dev accuracy. Used for both best-epoch checkpoint and Step 4's body-weight
  rule. Heavy/light is dropped from the score. Selected weights reported.
- **Epoch cap: 60 at every point (recommended), best-dev checkpoint.**
  Each epoch passes over every example once, so 63 clips gets ~10× fewer
  updates per epoch than 630. The saved Step 6 noun curves confirm low data
  peaks late:

  | Clips | Dev peak epoch, 9 runs (V, V+G, V+G+B × 3 seeds) | Loss if capped at 30 |
  |---|---|---|
  | 63 | 31–57 | 0.4–2.7 pp |
  | 158 | 33–60 | 1.0–4.3 pp |
  | 315 | 25–58 | 0.6–3.3 pp |
  | 473 | 22–59 | 0–2.5 pp |

  At 630 clips Step 7 arms peaked by epoch 22, so a 30-epoch cap would be
  safe there, but one cap at every point keeps the ladder comparable and
  matches Step 6 and Step 7 (60). Best-dev checkpointing removes the cost of
  late overfitting. Check after the runs: if any 63-clip run peaks in the last
  5 epochs, flag that point as possibly under-trained.

## Outputs

Per rung (nouns, actions) and per clip count: `V`, `V+G`, `V+G+B` accuracy
(mean ± SD over seeds); gaze, body and total gain with joint 95% intervals over
seeds and test source clips; selected body weights; test and s8 halves; shuffled
body at 158 and 630; per-action-word body gain. The 630 row also stands as the
fixed-learner Step 7 table and is compared against the original Step 7 numbers.

Key reading: the body gain on actions across amounts, and whether `V` on
actions flattens while `V+G+B` keeps rising.

## Model count and runtime

| Part | Models |
|---|---|
| 3 arms × 5 clip counts × 5 seeds | 75 |
| Shuffled body × 2 clip counts × 5 seeds | 10 |
| **Total** | **85** |

Runtime from saved Step 7 histories at 630 clips: ~46 s per epoch on
H100/H200, up to ~140 s on A30; a 60-epoch 630-clip run took 23–154 min
(median ~48). Training time scales with clip count (the ladder sums to 2.6×
a 630-clip run per arm and seed).

- Training: ~36 GPU-hours at the median speed; ~25 on fast nodes, ~70 if
  most jobs land on A30.
- Plus ~5 min per model for load and evaluation: ~7 GPU-hours.
- **Estimate: about 40–75 GPU-hours. At 8 concurrent jobs on Juno (Step 7's
  array), about 6–10 hours wall clock.** Juno only; no rental spend.
- Slurm time limit: keep Step 7's 5.5 h per job (the longest 630-clip run was
  ~2.6 h).

## Data handling

Same inputs and Juno storage as Step 7 (`/work/dal503972/nursery-step7`). Before
launch, confirm that no Databrary or BabyView participant-level files enter the
run; if any do, flag them in the run plan per the repository rules.

## Build order after approval

1. Add the Step 9 protocol builder (ladder × arms × seeds, noun+action dev
   score) to the existing Step 7 runner; no new learner code.
2. Action-word count check at each amount.
3. One-epoch smoke per arm at 63 clips.
4. Freeze the protocol, then launch the 85-model array on Juno.
