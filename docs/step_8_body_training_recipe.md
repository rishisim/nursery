# Step 8: settle how body input is learned

Design reference in [PLAN.md](../PLAN.md), following the
[property and action word tests](step_7_word_types.md). Formerly Step 5, which
was skipped before the [noun curves](step_6_language_acquisition_curves.md).
No new pretraining or architecture experiment is launched by this document.

## The idea

Decide whether learning the body signal's structure before word learning
makes the video-plus-body learner use the body. Keep the representation and
the Step 7 learner fixed while changing this training choice.

Wordless pretraining hides parts of body sequences and trains the encoder to
reconstruct them from surrounding bodily context, without words or category
targets. The encoder then enters word learning with video and body. This is
different from body-first word training, which learns word associations from
body before introducing video.

## Why consider it

In Steps 4 and 6, the body pathway trained from scratch with words is not used
at any data amount: development selection keeps its weight at 0–0.15, and at
the as-trained weight it lowers accuracy. The limit is how the body learns, not
only what it receives.

The exploratory diagnosis found masked-body pretraining followed by additive
word learning reached 81.56% against 76.88% video-only on the four-noun test,
a +4.69-point paired gain. It used 4×4 vision and additional pretraining
experience: 631 training clips across broader categories. The interval
described seed variability on a reused panel, not new-panel uncertainty.
Superiority to separately trained late fusion remained unresolved. The source
is identified under [diagnosis evidence](../archive/README.md#body-diagnosis-evidence).

This motivates a candidate, not a promised gain with 16×16, gaze or richer
body. Reconstruction can exploit easy temporal smoothness and cannot create
absent sensations.

## Keep the comparison controlled

| Condition | Body representation | Fusion | Wordless pretraining |
|---|---|---|---|
| Video reference | None | Video only | None |
| Video + steering gaze reference | None | Steering | None |
| Video + steering gaze + body | Fixed in Step 7 | Steering + additive body | None |
| Video + steering gaze + pretrained body | Same | Steering + additive body | Masked body reconstruction |

All conditions use the Step 7 learner, including its temporal video encoder,
and are scored on every rung: nouns, property words and action words. Reuse
Step 7 runs as references when data, model and selection settings match.

Gaze settings and gaze-driven speech remain matched across the relevant arms.
Wordless pretraining applies to the non-gaze body encoder in this comparison.
Additive remains the default for that body branch. Joint interaction and active
exploration stay deferred unless a later explicit question requires them. This
is not an open-ended fusion, loss or pretraining-method search.

## Count experience explicitly

For the curves, pretraining may access only the source clips selected for
that point. At 63 clips, pretraining and word learning both use those 63; the
remaining training pool and development/test clips stay outside pretraining
and normalization. New models at each point cannot inherit an encoder trained
at a larger point.

Extra passes add computation, not independent experience. Record pretraining
examples, steps, duration and compute separately from language-training updates.
All arms share DINO's visual prior; perception is not learned from scratch.
A gain supports the declared recipe. Isolating the pretraining objective's
value from extra compute would need an appropriate compute control.

Pretraining on all 631 clips before a smaller language subset would answer
language learning after broader prior bodily experience. That is a possible
study, but it is not the selected total-experience axis and must not be
silently substituted into the curves.

## How we choose and report

Keep vision, speech, splits, language budget and development-only checkpoint/
body-weight rules matched. Choose the recipe with predeclared criteria on
Step 7's development panels across all rungs, including treatment of compute;
test outcomes do not trigger retuning. Report ordinary and shuffled-body word
accuracy per rung, the selected body weights, useful body-only diagnostics, and
the additional pretraining budget.

Freeze the recipe before the Step 9 curves. Use it at every point, with a fixed
rule for scaling pretraining exposure and updates. Do not choose whichever
recipe wins each test point or rung. Retaining no pretraining is valid if the
comparison does not support adding it.

## The output we want

A compact ordinary-versus-pretrained table per rung, the selected recipe,
data/compute accounting and the reason for carrying it forward. Masking,
pretraining budget, fine-tuning, seeds, development selection and runtime cap
remain open. Keep representation quality, word-learning benefit and curve
results distinct.
