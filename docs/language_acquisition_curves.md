# Step 6: language acquisition with richer bodily experience

The main outcome in [PLAN.md](../PLAN.md), following
[representation work](body_representation.md), the
[controlled comparison](body_representation_comparison.md) and the
[training-recipe decision](body_training_recipe.md). The new protocol is not
frozen and no new runs have started.

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
| 16×16 video + body | Representation and training recipe selected before this study |

Use additive unless the preceding decision explicitly changes it. No automatic
joint condition or per-point architecture search is included. If wordless
pretraining is selected, its experience and normalization are restricted to
each point's own selected training clips.

## What the learner experiences

Reuse corrected weak speech and existing source experiences. Names may be
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
- Use identical four-choice candidates for both learners; each body-enabled
  candidate receives its own history. Keep a complete-history shuffle control
  on the same selected models, trials and frozen donors.
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
reference alongside the two new live curves.

| Independent clips | Recorded 4×4 video | 16×16 video | Selected 16×16 video + body | Body gain over 16×16 video |
|---:|---:|---:|---:|---:|
| 63 | 30.0% | Pending | Pending | Pending |
| 158 | 39.5% | Pending | Pending | Pending |
| 315 | 49.6% | Pending | Pending | Pending |
| 473 | 56.4% | Pending | Pending | Pending |
| 630 | 60.1% | Pending | Pending | Pending |

These rows illustrate old clip counts, not a frozen new pool. The historical
column is measured; all pending cells are unmeasured. Full uncertainty and
seed values are in the [original record](../archive/language_learning/acquisition_curves.md#completed-results--27-october-2026).

Report 16×16-minus-4×4 only with matching status established as in
[Step 1](visual_baseline.md). If evaluation changes, keep historical context
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
