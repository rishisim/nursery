# Step 3: compare current and richer body representations

Design reference in [PLAN.md](../PLAN.md), following the
[visual baseline](visual_baseline.md) and [body representation](body_representation.md).
No execution settings are frozen and no run has started.

## The idea

At one fixed amount of experience, ask whether richer bodily information
improves word learning under the same learning setup. Keep the comparison
interpretable before expanding to acquisition curves. Better use of body is
the hypothesis, not a criterion for accepting the measurements.

## The conditions

| Condition | Visual grid | Fusion | Wordless pretraining | Status |
|---|---|---|---|---|
| Recorded video reference | 4×4 | Video only | None | Existing historical result; not an automatic new training arm |
| Video only | 16×16 | Video only | None | Planned |
| Video + current body | 16×16 | Additive | None | Planned matched control |
| Video + richer body | 16×16 | Additive | None | Planned representation change |

Keep the recorded 4×4 row to show the visual improvement. Use its matched clip
count from the [original curves](../archive/language_learning/acquisition_curves.md#completed-results--27-october-2026).
[Step 1](visual_baseline.md#how-we-use-the-baseline) defines when its difference
from 16×16 is a matched comparison. The old 4×4 additive model cannot substitute
for a current-body control using 16×16 vision.

## What the learner experiences

Use one fixed subset of the broader natural weak-speech acquisition task,
not only the four-noun panel fine-token vision already solves at about 92%.
The three planned conditions share source clips, video, speech and evaluation
candidates. Names may be delayed, nonheld or absent from view; do not filter
ordinary training examples by hidden referent identity.

Body conditions differ in their specified representation. Keep all five modality
groups, accessible temporal span and body dropout consistent. Record additional
temporal samples, channel dimensions, required parameter changes and compute.
Keep encoder family and capacity comparable where possible rather than changing
the fusion architecture at the same time.

## How training and testing work

Use additive fusion for both body conditions, with no wordless pretraining,
staged body-first word training, joint interaction or auxiliary sensory losses.
Train fresh models with paired seeds and matched example order, common visual
initialization and declared optimization opportunity.

Apply the same predeclared development-only body-weight calibration rule to
both body conditions. Freeze the weight grid, tie rule and checkpoint/weight
selection order before launch; include zero body weight. Test results must not
choose the rule or tune the richer condition.

Keep source-clip partitions disjoint, including replays and body history. Use
identical four-choice situations, each with its own body where applicable.
Select checkpoints on development data. Report held-out-person and new-clip
results separately. Evaluate complete shuffled body histories on the same
selected models and candidates, moving channels and masks together.

Keep body-information diagnostics separate from word accuracy: a category
probe can reveal accessible distinctions but does not establish word learning.

## The output we want

| Condition at the chosen data amount | Held-out word accuracy | Difference from 16×16 video |
|---|---:|---:|
| Recorded 4×4 video reference | Existing result, with protocol label | Historical comparison |
| 16×16 video | Pending | — |
| 16×16 video + current body | Pending | Pending |
| 16×16 video + richer body | Pending | Pending |

Report richer-minus-current body, the matching status of 16×16-minus-4×4,
all seed values, shuffled-body scores, selected weights, independent clip
counts and compute. Separate seed variability from uncertainty across test
source clips. Reused panels remain exploratory.

## What this decides

Record what representation is supported for the next build step and why. Poor
from-scratch learning does not prove richer input contains no information;
dimensionality and optimization can matter. Follow-ups should address a specific
unresolved issue rather than start a broad search.

The fixed clip count, seed count, panels, budget, calibration rule and runtime
cap remain to be frozen. Use one descriptive ignored run root when execution
is requested. This step does not launch the
[training-recipe comparison](body_training_recipe.md) or [main curves](language_acquisition_curves.md).
