# Step 5: settle how the improved body input is learned

Design reference in [PLAN.md](../PLAN.md), following the
[representation comparison](body_representation_comparison.md). No new
pretraining or architecture experiment is launched by this document.

## The idea

After the sensory representation is understood, decide whether learning its
structure before word learning improves the video-plus-body learner. Keep the
representation fixed while changing this training choice.

Wordless pretraining hides parts of body sequences and trains the encoder to
reconstruct them from surrounding bodily context, without words or category
targets. The encoder then enters word learning with video and body. This is
different from body-first word training, which learns word associations from
body before introducing video.

## Why consider it

The exploratory diagnosis found masked-body pretraining followed by additive
word learning reached 81.56% against 76.88% video-only on the four-noun test,
a +4.69-point paired gain. It used 4×4 vision and additional pretraining
experience: 631 training clips across broader categories. The interval
described seed variability on a reused panel, not new-panel uncertainty.
Superiority to separately trained late fusion remained unresolved. The source
is identified under [diagnosis evidence](../archive/README.md#body-diagnosis-evidence).

This motivates a candidate, not a promised gain with 16×16 or richer body.
Reconstruction can exploit easy temporal smoothness and cannot create absent
sensations.

## Keep the comparison controlled

| Condition | Body representation | Fusion | Wordless pretraining |
|---|---|---|---|
| 16×16 video reference | None | Video only | None |
| Video + body | Fixed candidate from Step 4 | Additive | None |
| Video + pretrained body | Same candidate | Additive | Masked body reconstruction |

Reuse Step 4 references when data, model and selection settings match. If the
question instead becomes whether pretraining changes the relative value of
current versus richer body, apply it to both representations in a separately
specified comparison rather than mixing questions.

Additive remains the default. Joint interaction and active exploration stay
deferred unless a later explicit question requires them. This is not an
open-ended fusion, loss or pretraining-method search.

## Count experience explicitly

For the main curves, pretraining may access only the source clips selected for
that point. At 63 clips, pretraining and word learning both use those 63; the
remaining training pool and development/test clips stay outside pretraining
and normalization. New models at each point cannot inherit an encoder trained
at a larger point.

Extra passes add computation, not independent experience. Record pretraining
examples, steps, duration and compute separately from language-training updates.
Both arms share DINO's visual prior; perception is not learned from scratch.
A gain supports the declared recipe. Isolating the pretraining objective's
value from extra compute would need an appropriate compute control.

Pretraining on all 631 clips before a smaller language subset would answer
language learning after broader prior bodily experience. That is a possible
study, but it is not the selected total-experience axis and must not be
silently substituted into these curves.

## How we choose and report

Keep vision, speech, splits, language budget and development-only checkpoint/
body-weight rules matched. Choose the recipe with predeclared development
criteria, including treatment of compute; test outcomes do not trigger
retuning. Report ordinary and shuffled-body word accuracy, useful body-only
diagnostics, and the additional pretraining budget.

Freeze the recipe before the [main curves](language_acquisition_curves.md).
Use it at every point, with a fixed rule for scaling pretraining exposure and
updates. Do not choose whichever recipe wins each test point. Retaining no
pretraining is valid if the comparison does not support adding it.

## The output we want

A compact ordinary-versus-pretrained table, the selected recipe, data/compute
accounting and the reason for carrying it forward. Masking, pretraining budget,
fine-tuning, seeds, development selection and runtime cap remain open. Keep
representation quality, word-learning benefit and curve results distinct.
