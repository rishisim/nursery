# Past experiments

This is the explanatory guide to Nursery's completed and inactive work as of
27 October 2026. Start active work from [PLAN.md](../../PLAN.md). The canonical
detailed language protocols, frozen gates, commands and results remain in
[experiments.md](experiments.md); this guide explains what each intervention
changed and what its result does and does not establish. Historical proposals
are not an active queue or authorization to train.

The long-term question is whether bodily inputs improve language acquisition
over matched video and speech, especially across independent data amounts.
None of the results below guarantees a noun-learning benefit or establishes a
higher ceiling. Most noun pilots used one seed and fixed data; accuracy changes
across protocols are not acquisition curves. Old hard/soft scoring, speech and
candidate-rule changes make absolute scores across stages incomparable.

## How the learners differed

The cue studies weighted image/speech examples; the hand gates biased visual
region selection. Neither supplied the later information-bearing sensory
representation as a noun input. The additive learner instead encodes contact,
finger pose and arm position with group MLPs and a clip-reset bidirectional GRU,
then adds a learned body–word score to a visual–word score. Frozen DINO features
feed a trainable 512-d visual projection and trainable word vectors.

In ordinary additive runs, the visual projection, word vectors and body encoder
are trained **simultaneously from the first update**. The body encoder was not
first pretrained and then attached to a finished video learner. Body dropout
also trains the picture pathway. The optional vision-to-body loss is an
auxiliary objective during training, distinct from the later two-phase sensory
pretraining experiment. See [the implementation reference](../../docs/learner.md).
Frozen DINO contributes prior visual knowledge; these are not learners starting
vision from scratch.

## Cue weights and hand gates

The [90-run contact grid and independent 100-run confirmation](cue_experiments.md)
asked whether correctly timed contact-based weights improve word learning.
The primary grid contrast was inconclusive (+2.27 pp, interval −9.34 to +13.89).
Confirmation over 25 fresh seeds found contact − no cue +0.94 pp
[−0.95, +2.83], while contact beat shuffled and shifted weights. The verdict
was timing-only: corrupted weights are worse, but improvement over ordinary
no-cue learning was not established. This does not reject richer body inputs.
The original staging/render repair also means invalidated caches must stay
separate from corrected evidence.

The historical hard-pooling gate (`gatehard`, implementation
[`7ec9bd17`](https://github.com/rishisim/nursery/commit/7ec9bd17)) and soft-MIL
prior (`gatesoft`, [`b2ed6599`](https://github.com/rishisim/nursery/commit/b2ed6599))
used hand location to steer region selection. Learned trust collapsed toward
zero. This showed those gates were ineffective in their setup, not that hand
or contact content is useless. Their design also coarsened/noised legitimate
category-predictive contact and hand-shape information under an overly broad
leakage rule. Exclude hidden object identities from normal learner inputs;
do not remove sensory information merely because it predicts a category.

## Additive noun learning and diagnostic controls: 4.1–4.8

**Actual body input (4.1).** The 512-d additive learner trained on weak speech
with body dropout and visual-to-sensory prediction. It did not improve
picture-only rendered/Konkle performance or held-out body-present decisions.
The pathway was active; a collapsed or disconnected body branch did not explain
the result. This tested one fusion and supervision setup, not all multimodal
representation learning.

**Oracle, distillation and direct labels (4.2–4.4).** A privileged held-object
identity oracle reached 94.3% body-only accuracy on held-referent trials; fused
accuracy was 76.6%, versus 61.7% picture-only under that scoring setup. The
pathway could use strong information, though additive calibration was poor.
A frozen oracle teacher's distillation into a picture-only student did not
improve transfer. Direct visible-object labels with the same frozen features
and visual head reached 90.0% rendered / 84.3% Konkle versus 76.4% / 69.1%
for weak speech. Thus visual headroom existed; the earlier hypothesis that
exhausted frozen features explained the failed distillation was too strong.
Neither privileged oracle inputs nor direct labels are evidence of learning
from real body signals and naturally weak speech. Soft rescoring and removing
the concurrent prediction loss did not rescue the real-body comparison.

**Body signal and extra language loss (4.5–4.7).** The [contact audit](body_signal_audit.md)
found meaningful geometric contact and only limited source-grasp sharing.
The body contained some category information. Adding a body-only word-matching
loss on the same weak pairs increased in-sample body accuracy from 57.1% to
74.6%, while s8 remained near chance (26.2% to 27.9%); fused generalization
worsened. This separated ability to fit training examples from useful transfer.
It did not identify exactly which nuisance variables were memorized.

**New clips versus new people (4.8).** A source-clip-disjoint split held all
replays of a source clip together, with s8 additionally held out. Body failed
on new clips even from familiar people: fused 45.6% versus video 48.3%; s8
40.6% versus 43.7%. Dev never showed the proposed substantial early benefit.
The earlier interpretation that subject calibration alone was the problem was
therefore inadequate. Subject normalization and automatic shuffled/seed sweeps
were proposed conditionally and were not performed as follow-ups here.

Full settings, historical hypotheses and corrected interpretations are in
[the 4.1–4.8 record](experiments.md#order).

## Joint fusion and supervised sensor probes: 4.9–4.13

**Joint Transformer (4.9).** Visual-region tokens and encoded body states
shared a small Transformer; only contextualized visual tokens scored words.
The matched joint control scored 36.9% on familiar-person new clips, the body
arm 36.0%, versus 48.3% for the earlier linear control. The joint pilot also
reduced the common representation from 512 to 128 dimensions and changed
region interactions. Its regression and train/dev gap are facts; blaming
joint fusion alone or treating memorization as a uniquely proven cause is not
justified. Keep the working 512-d reference when researching a new architecture.

**Supervised beyond-video probe (4.10).** On labeled held frames, video plus
legacy body features gained only +1.0 pp [−2.5, +6.1] over video on new clips,
and lost 5.1 pp on s8. Tiny/hidden-object gains did not generalize to s8. These
are supervised category diagnostics, not weak-language acquisition results.
They do not impose a universal upper bound on assistance during learning.

**Modeled load and skin channels (4.11–4.12).** Analytic wrist load from
sampled mass and recorded acceleration supplied stronger category information.
The decisive per-clip control gained +2.7 pp on new clips and +6.6 on s8;
revised load-plus-skin channels gained +5.8/+10.8. Temperature alone failed;
skin channels also contain contact area, so the combined gain cannot be
attributed uniquely to thermal information. Authored category-conditioned
property distributions partly build in the signal. Resampling properties per
clip supplies new assignments, not new physical trajectories or meshes.

**Physical channels in language (4.13).** Adding load and skin to the full
body encoder did not produce a noun gain: fused 42.4% versus video 48.3%
on new clips, with no triggered shuffled follow-up. A supervised sensor-probe
gain did not translate into this weak-language objective. Load/skin-only noun
learning, area-only probes and noise-free-effusivity ceilings were suggestions,
not completed experiments in this record. See [the 4.9–4.13 record](experiments.md#order).

## Corrected matched noun comparison: 4.14

The [historical review](body_input_review.md) identified held-channel speech
whose supporting contact came from another hard-cut source clip. Weak/delayed
and nonheld naming were intentional; that construction mismatch was distinct.
The repair made speech associations clip-local and body-present scoring use
renderer-grounded, matched candidate identities. Both arms received identical
corrected weak speech and retained the 512-d additive architecture.

At seed 0, body-present new-clip accuracy was video **52.28%** versus full body
plus load/skin **47.54%** (−4.74 pp, paired-trial interval [−8.60, −0.88]).
s8 and picture-only outcomes also showed no gain. The independent audit passed.
This closes the bounded repair-and-comparison question; the trial interval
does not capture seed variability or full source-clip dependence. The repaired
candidate rules differ from 4.13, so their absolute scores are not a causal
before/after estimate of the speech repair. This fixed-data result is not a
ceiling test. See [the exact record](experiments.md#414-matched-corrected-exposure-rerun).

## Heavy/light positive control: 4.15

This was additive language learning with **load only**, matched video,
balanced heavy/light labels aligned to lifting episodes, and two test categories
excluded from diagnostic training. It was scaffolded property-word learning,
not ordinary ambiguous noun speech. Across three seeds, video + load scored
**92.42%**, video **37.88%**, and deranged-load evaluation **46.21%**. All frozen
capability gates and independent review passed. Body-trained picture-only
performance was 40.91%, also below chance.

The result shows that the additive route can learn a narrow physically
informative word distinction and depend on the aligned load signal. Video's
below-chance score does not prove weight cannot be inferred visually. Mass was
sampled per clip, did not change motion/RGB, and was not a persistent asset
property; load was analytic, not a contact-force solve. Three seeds and two
test categories do not establish general weak noun learning, visual transfer
or a higher language ceiling. See [the exact protocol and result](experiments.md#step-2-heavy-light-capability-test).

## Sensory-target pretraining and visual transfer: 4.16

V used visual temporal pretraining; A added aligned sensory prediction; S added
deranged sensory targets. The sensory values were **prediction targets**, not
an information-bearing body encoder's inputs to joint language learning.
Only the trainable 512-d visual projection transferred into fresh, otherwise
identical video-only noun learners. The body encoder was not trained during
pretraining, and the sensory predictors were removed before language training;
all downstream language readouts were body-absent. This was a
picture-only transfer test, not a benefit with body present that was lost
merely by removing sensors at evaluation.

Across three seeds, corrected new-clip accuracy was **V 52.63%, A 51.87%,
S 52.87%**. Neither A−V nor A−S passed the frozen transfer gate. Aligned sensor
prediction was learned descriptively (A loss 1.4768 versus S 2.1508 and the
training-mean reference 2.1026), without demonstrated weak-noun transfer.
The nine planned runs and independent audit completed. This does not establish
harm, settle joint body-input acquisition, or refute all sensory pretraining.
The previously consulted test set makes this exploratory. See [the frozen
protocol, uncertainties and exact results](experiments.md#step-3-transfer-focused-sensory-pretraining).

## Earlier real-data and simulator work

AEA used adult partly scripted video/transcript/IMU data as a development
analogue. Its terminal coarse-action diagnostic failed its annotation/language
support gates; later conditional split, capacity and IMU checks were not run.
The exact-window route stopped, without establishing transferable language
learning. This was not evidence from infants or a general rejection of weak
alignment. The [Git-preserved scientific report](https://github.com/rishisim/nursery/blob/e6acf51c/output/aea_coarse_action_v3/scientific_report.md)
and [terminal decision](https://github.com/rishisim/nursery/blob/e6acf51c/output/aea_coarse_action_v3/terminal_decision.json)
separate failed gates from unrun checks.

The older EgoBabyVLM CLIP+ pilot completed bounded preprocessing, DINO/BERT
rehearsals, a joint training cycle and evaluation on exactly two videos and one
engineering seed. Its own [P9 decision](https://github.com/rishisim/nursery/blob/92629eeb/reproductions/egobabyvlm_clip_plus/pilots/juno_sample/p9_decision.json)
classifies it as engineering-only, non-comparable and not a reproduction result;
scientific full-reproduction readiness remained blocked by exact usable-subset
evidence. Pilot weights and scores are not Nursery acquisition evidence.

Earlier synthetic-generator screens, procedural simulation, MIMo/MolmoSpaces
and Unity/MuJoCo gates are indexed in [the archive](../README.md). EmbodiedGen,
HOIDiNi conversion and InterMimic reached useful kinematic/format milestones,
but physical contact and floor validity failed; see [the motion record](../motion_pipeline.md).
Their architecture proposals remain historical proposals.

### Replay acceptance record

The s1/s9/s10, two-session-per-subject, seed-7 acceptance pilot rendered 720 frames
and 816 probes in 164.05 s with six workers. It realized 3/4/5 distractors in
24/23/29 clips, with no zero-fit clips, mean realized count 4.066 and mean visible
categories 4.551. Labelled-contact median and p90 changed by -0.0154 mm and
+0.0470 mm against the earlier pilot; the whole-hand median was unchanged.
Each session's preparation time (17.83–29.04 s) was below its Blender time
(30.06–45.34 s). The superseded global margin produced 1,155 zero-fit clips out
of 1,177 and 1.011 mean visible categories; those train sessions are discarded.
These are historical measurements, not fresh validation. Adult kinematics,
visibility and physical-coherence limits still matter.

## What was retained and why

Artifact paths are unchanged. The subsequent source cleanup moved completed
diagnostic runners and motion adapters into `archive/code/`, with their tests.
The simulator now uses the common output guard; active video-only and additive
models have explicit files and shared training/scoring utilities. The
[archive index](../README.md#archived-source) identifies the historical snapshots
and exact Git layout for reproduction. Presence of a retained diagnostic flag
does not recommend using it.

| Ignored location | Retained role |
|---|---|
| `outputs/renders/clean_856/` | Source renders/manifests and reusable data; not new independent experience per replay. |
| `outputs/runs/clean_body/` | Earlier saved speech/features, models and noun comparisons; preserve historical hashes and path references. |
| `outputs/runs/step1_matched/` | Corrected matched exposure, 4.14 runs and audits; also inputs reused by 4.16. |
| `outputs/runs/step2_heavy_light/` | Narrow property-word diagnostic, six runs and verification. |
| `outputs/runs/step3_pretrain/` | Frozen protocol, nine completed runs, comparison and audit records. |
| `outputs/legacy/` | Earlier cue/gate work and superseded or invalidated artifacts; never pool invalid evidence with clean runs. |
| `.external/` | Environments, upstream checkouts and licensed/external dependencies. |

These roots remain ignored and were not uploaded, moved or deleted for this
reset. Curated small records are versioned in the archive; local artifacts may
be absent from a fresh clone. The [historical literature/audit review](body_input_review.md)
retains its dated recommendations as evidence, superseded as an active plan by
[PLAN.md](../../PLAN.md). New joint fusion and optional pretraining remain
research candidates; no historical “next” clause chooses or authorizes them.
