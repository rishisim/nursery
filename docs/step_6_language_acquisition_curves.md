# Step 6: language acquisition with richer bodily experience

The main outcome in [PLAN.md](../PLAN.md), following
[representation work](step_3_body_representation.md), the
[controlled comparison](step_4_body_representation_comparison.md) and the
[training-recipe decision](step_5_body_training_recipe.md). Protocol
[frozen](#frozen-execution-protocol--16-november-2026); no runs have started.

## The idea

Give learners the same video and weak speech, with body inputs available to
the multimodal learner. Increase independent experience and ask whether body
helps acquire words sooner, sustains an advantage at larger amounts, or
eventually lifts a visual plateau. The desired shape remains a hypothesis.

The earlier diagnostics serve this goal. Better object classification from
body alone is supporting evidence, not a replacement for word-learning curves.

## The learners and historical reference

| Curve | Role |
|---|---|
| Recorded 4×4 video | Preserved historical reference; no automatic rerun |
| 16×16 video | Matched current visual baseline |
| 16×16 video + steering gaze | Intermediate embodied reference |
| 16×16 video + steering gaze + selected body | Representation and training recipe selected before this study |

Use steering for gaze and additive fusion for the other body channels unless
the preceding decision explicitly changes it. The current-body representation
remains a fixed-data control, not an automatic fourth live curve. If no body
candidate is supported, video versus video + gaze remains a valid curve study;
do not require a positive body result. No automatic
joint condition or per-point architecture search is included. If wordless
pretraining is selected, its experience and normalization are restricted to
each point's own selected training clips.

## What the learner experiences

Reuse the same gaze-driven weak speech and existing source experiences. Names may be
delayed, describe a nonheld object or lack a visible referent. Hidden identity
does not select ordinary training samples. Video and speech, crop, visual
history and source partitions are identical across arms.

Keep body representation, accessible history and calibration rules consistent
across points. Report sensory sample rate and body exposure alongside clip
counts. Higher-rate samples and replayed seconds do not count as new independent
clips. Do not blend new property-word/sentence tasks into the noun score or
claim new-instance generalization without genuinely disjoint instances.

## How training and testing work

- Use nested subsets of independent source clips. The previous fractions
  10%, 25%, 50%, 75% and 100% are a starting proposal; their old counts were
  63, 158, 315, 473 and 630. Freeze actual counts from the validated pool.
- Train fresh models with paired seeds at each point. Match visual initialization,
  example order, language-training opportunity and development selection.
  Fix the budget rule and disclose extra body pretraining computation.
- Fit vocabulary and all normalization on that point's selected training data.
  Keep panels independent of vocabulary, freeze unknown-word scoring and
  report vocabulary coverage separately.
- Split by source clip, keeping replays, histories and pretraining samples in
  their partition. Use independent test situations and report held-out-person
  results separately. Ten seeds on one panel do not create ten new panels.
- Use identical four-choice candidates for all learners; each body-enabled
  candidate receives its own history. Keep a complete-history shuffle control
  on the same selected models, trials and frozen donors, retaining real gaze
  when shuffling the other body channels.
- Freeze development selection, uncertainty reporting and runtime limits
  before launch. A null result completes the comparison; it does not queue
  extra data amounts, conditions or retries.

The existing runner, audited inputs and completed models are
[preserved](../archive/README.md#completed-study-worktrees). Reuse verified
mechanics where appropriate and record every protocol change. New results
require a fresh ignored run root and provenance; preserve the original runs.

## The output we want

Plot independent experience against held-out word accuracy, with the 25% chance
line and clearly identified uncertainty. Show recorded 4×4 video as historical
reference alongside the three planned live curves. Primary curves use each
learner's assigned inputs. Separately report all-extra-inputs-removed and
picture-only transfer; these are stronger transfer questions, not requirements
for every embodied benefit.

Report gaze gain (V+G minus V), additional body gain (V+G+B minus V+G),
and total embodied gain (V+G+B minus V). Keep gaze settings consistent across
gaze-enabled arms and data amounts, with one predeclared calibration rule.

| Independent clips | Recorded 4×4 video | 16×16 video | Video + steering gaze | Video + steering gaze + selected body |
|---:|---:|---:|---:|---:|
| 63 | 30.0% | Pending | Pending | Pending |
| 158 | 39.5% | Pending | Pending | Pending |
| 315 | 49.6% | Pending | Pending | Pending |
| 473 | 56.4% | Pending | Pending | Pending |
| 630 | 60.1% | Pending | Pending | Pending |

These rows illustrate old clip counts, not a frozen new pool. The historical
column is measured; pending cells await the matched curve study. The completed Step 2 full-data
V and V+G measurements may be reused only if protocol and selection settings
match; the combined gaze-plus-body condition is unmeasured. Full uncertainty and
seed values are in the [original record](../archive/language_learning/acquisition_curves.md#completed-results--27-october-2026).

Report 16×16-minus-4×4 only with matching status established as in
[Step 1](step_1_visual_baseline.md). If evaluation changes, keep historical context
without calling the raw score difference a resolution effect. Separate seed
variability from uncertainty across test clips; cluster dependent encounters
by source clip. Keep shuffled-body, held-out-person and picture-only transfer
results separate from the primary curve.

## What the result means for the build

A low-data advantage supports faster acquisition under the tested conditions.
An advantage across the range supports better measured acquisition. A higher
ceiling requires evidence that the visual curve levels off as independent data
increases; an epoch plateau or one favorable point is insufficient. Rising
curves leave the ceiling unresolved.

Representation, pretraining choice, data points, panels, seed count, budgets
and runtime cap remain to be specified after the preceding steps. The small
four-noun 16×16 result is not this broader task's ceiling or numerical target.

## Frozen execution protocol — 16 November 2026

Freeze script: `scripts/step6_acquisition_curves.py`. Protocol and runs:
`outputs/runs/step6_acquisition_curves/` (ignored). Step 5 is skipped: additive
fusion, no wordless pretraining. The selected body is the Step 4 current body
(153 dims); rich body is not run.

| Setting | Value |
|---|---|
| Arms | `video`, `video_gaze`, `video_gaze_body_current`, all 16×16 |
| Learner, training, steering, body, pairing, body-weight rule, evaluation | Exactly Step 4's [frozen protocol](step_4_body_representation_comparison.md#frozen-execution-protocol--10-november-2026) |
| Data amounts | Nested seed-0 prefixes of the 630 training source clips; split seed 0; same panels (`clips_test` 556, s8 793) |
| Budget | 60 epochs at every amount: each example gets the same passes, so larger amounts get more updates |
| Per-amount fitting | Vocabulary (min count 5) and body normalization fit only on that amount's selected clips |
| Seeds | 0, 1, 2 at every amount; 45 fresh models |
| Compute | CUDA, `scripts/step4_rental.py` queue, 4-hour runtime cap, $20 cost cap |

| Fraction | Independent clips | Usable utterances | Replay seconds | Vocabulary (incl. pad) |
|---:|---:|---:|---:|---:|
| 10% | 63 | 1,528 | 6,819 | 109 |
| 25% | 158 | 3,859 | 16,312 | 146 |
| 50% | 315 | 8,018 | 31,776 | 158 |
| 75% | 473 | 12,925 | 47,334 | 159 |
| 100% | 630 | 18,613 | 63,153 | 159 |

Preparation checks: the 100% subset and all panels are SHA-256 identical to
Step 4's, each subset strictly contains the previous one, and no dev, test or
s8 clip enters any subset. A one-epoch CPU smoke on the 10% subset passed for all
three arms. The 100% runs must reproduce Step 4 (video 63.67, gaze 75.60,
current body 76.26 on test) within seed variability.

Note: Step 4's dev rule turned the current body off at 100% (w = 0, 0.05, 0).
If it does so at every amount, the body curve measures the gaze pathway trained
alongside body, not body use; selected weights are reported per point.
