# Main experiment: language acquisition with video and body

Canonical design and execution protocol. Preparation and implementation are
verified; scientific training has not started. The
[smaller study](cross_situational_learning.md) uses the same learner.

## The idea

Give learners the same video and speech, but give some learners bodily
experience too. Increase the amount of independent experience and measure how
well each learner understands the words on unfamiliar clips.

The hypothesis is the curve we have been aiming for: video plus body learns
more effectively and eventually reaches a higher level than video alone.
That shape is what we want to test, not a result we already have.

## What the learner experiences

Use the existing corrected video/speech data and cached visual features.
Both conditions hear the same weakly aligned speech: names can be delayed,
concern another object, or have no visible referent. Hidden referent identity
must not select ordinary training examples.

The full-body conditions receive all five existing streams: contact, finger
configuration, arm position/movement, modeled load, and modeled skin
temperature/contact area. Load and skin are synthetic measurements; the
recorded motion does not adapt to their assigned physical properties.

Keep the existing visual window and clip-bounded ±5-second body window, with
individual body time steps available for interaction. This is offline learning
using context before and after speech, with no persistent memory across clips.

## The three learners

| Learner | How it uses the experience |
|---|---|
| Video only | Learns visual–word associations. |
| Additive video + body | Learns visual–word and body–word evidence, then combines their scores. |
| Joint video + body | Keeps those direct pathways and adds a learned visual–body interaction that also contributes to word matching. |

The joint design is deliberately an extension of the additive learner:

```text
Visual evidence ─────────────────┐
Body evidence ───────────────────┼── Match the experience to language
Visual–body interaction ─────────┘
```

Body information can support a word directly, even when it says something the
video cannot. The interaction handles relationships that depend on both
modalities. It must retain access to both visual and bodily content; visual
tokens must not be the only route to the language decision.

Preserve frozen DINO features, the working 512-dimensional visual/word
representation, and the existing body encoder. Add one small interaction
module, initialized with zero contribution so that disabling it recovers the
additive model. Train the visual projection, word vectors, body encoder and
interaction through the same language-matching loss. No sensory pretraining,
extra sensory-prediction loss, object tracker, or intention model is included.

## How training and testing work

- Use nested amounts of training experience: proposed points are 10%, 25%,
  50%, 75% and 100% of eligible training source clips, with three paired seeds.
  All replays of a source clip stay in the same partition. Replayed frames do
  not count as new independent experience.
- At each point, train fresh learners on the same video/speech subset. Match
  common initialization, example order and training opportunity. Keep body
  exposure/dropout identical between additive and joint conditions.
- Fit normalization and vocabulary on that subset only. Keep a fixed test panel,
  report vocabulary coverage, and define scoring for unseen words before launch.
- Select checkpoints on development data. Report source clips, utterances and
  exposure duration alongside each point.
- Evaluate with the repaired four-choice situation-matching protocol: given a
  word, select its matching situation from four candidates. Each body-enabled
  candidate includes its own sensory history. Candidates are identical across
  learners. New-source-clip testing is primary; report held-out-person results
  separately.

The first main curve measures the existing grounded noun task. Broader property
or sentence benchmarks are not silently added or blended into that score.

## The output we want

A plot with **independent training experience on the horizontal axis** and
**held-out word-understanding accuracy on the vertical axis**. Show all three
learners, uncertainty across paired seeds, and the 25% chance line.

The desired shape, shown schematically rather than as measured numbers:

```text
Accuracy
  |                      ______  desired video + body
  |                 ____/
  |            ____/
  |        ___/      __________  video only
  |     __/     ____/
  |____/_______/
  +--------------------------------> independent experience
```

The additive reference's position is measured, not assumed. Add a table of
accuracy and paired differences at each data amount, plus a fixed shuffled-body
evaluation control. Report picture-only transfer in a separate secondary panel.

## Starting alongside the smaller study

Before launch, freeze the interaction details, splits, subsets, trials, training
settings and runtime limits. Verify baseline equivalence, gradients, masks and
train/evaluation scoring. Existing test panels have informed earlier work;
results on reused panels remain exploratory.

Both studies use the same frozen implementation, separate models and separate
ignored run directories. Neither supplies pretrained weights or mid-run design
changes to the other. Preserve the data and
[past experiments](../archive/language_learning/past_experiments.md).

## Frozen execution protocol

Use the corrected `outputs/runs/step1_matched/` speech and frozen feature caches,
`outputs/renders/clean_856/` episodes and their `physical_by_clip/` signals in
the original checkout. New artifacts live only under this worktree's ignored
`outputs/runs/acquisition_curves/`. `protocol.json`, `signal_hashes.json` and
`audit.json` record the exact input hashes, preparation and checks. No rendering,
speech regeneration, additional benchmark or smaller-study run is included.

Keep source-clip split seed 0, 15% development and 15% test among the training
subjects, and s8 held out. Use one seed-0 ordering of eligible training clips,
nested prefixes rounded upward, and fresh training seeds 0, 1 and 2 at every
point. All three learners share the ordering and examples. Exclude training
examples whose accessible visual context reaches an unselected clip or another
partition; preserve the natural speech without referent-based selection.

| Fraction | Independent clips selected/exposed | Usable utterances | Available replay seconds |
|---|---:|---:|---:|
| 10% | 63 | 1,528 | 6,819 |
| 25% | 158 | 3,859 | 16,312 |
| 50% | 315 | 8,018 | 31,776 |
| 75% | 473 | 12,925 | 47,334 |
| 100% | 630 | 18,613 | 63,153 |

Replay seconds describe available selected-clip exposure, not independent
physical duration or epoch-repeated exposure. Independent source duration is
not inferred from those repetitions. Build each vocabulary with minimum count
5 on that subset's speech and fit normalization only on its selected clips.
The preparation audit reconstructs the subsets, vocabularies, normalization,
panels and donors from the retained inputs.

Train 45 models: 5 data amounts × 3 seeds × 3 learners. Each gets 60 epochs,
batch size 256, AdamW with learning rate 0.0003 and weight decay 0.1, soft pooling
temperature 0.1, ±2-second visual and clip-bounded ±5-second body windows.
Both body learners receive all five groups and 50% body dropout; every auxiliary
loss is off. Independent random generators preserve paired initialization,
example order and body-dropout masks. Larger datasets receive more updates;
each example receives the same number of passes.

The joint interaction uses a single 64-dimensional body-query/visual-key/value
attention operation. Multiply attended visual values by the projected body
state, average over observed time steps in the contiguous centre clip, and map
to 512 dimensions using a bias-free zero-initialized output projection. Its
dot product with the word vector adds to the unchanged direct scores. Invalid
visual tokens are masked; absent body histories or body dropout disable both
body-dependent terms. Do not normalize the initially zero interaction output.
The output head gets gradients first, followed by its upstream parameters once
the head moves away from zero. The old `--arch joint` remains historical.

Freeze noun-only query panels, candidates and correct slots at trial seed 0
before vocabulary filtering. Training retains full weak utterances. An unknown
query receives 25% expected accuracy with the denominator unchanged; report
trial-weighted coverage and known-query accuracy separately. Use each learner's
best clips-dev accuracy for its primary checkpoint, earliest epoch on a tie.
Use a separate picture-dev-selected checkpoint for picture-only rendered and
Konkle transfer. Test scores do not select epochs.

Shuffled-body evaluation uses a fixed seed-0 donor for each situation, sampled
with replacement from the same subject and evaluation panel but a different
source clip. Move all channels and validity masks together. The same mapping
is used for both body learners at every size and seed. Missing valid donors
are a preflight failure; no candidate rule is relaxed.

Run a one-epoch, full-data timing check for each learner, discard those weights,
then start every scientific model fresh. Execute sequentially on local MPS with
no paid compute and a 24-hour cap starting at the timing check. Stop before the
matrix if its conservative timing projection exceeds that cap. Preserve and
report incomplete work if the cap is reached; never shrink the matrix to fit.

Report mean ± sample standard deviation and all three seed values, with paired
differences computed within seed. The main curve is new-clip body-present noun
understanding; s8, shuffled-body controls and picture-only transfer stay separate.
Reused panels are exploratory. A null result still completes the comparison;
an unsaturated curve does not establish a language-learning ceiling.

The smaller study's common video/additive/body/loss/window implementation traces
to `bdcb1c7f5ffcabf99631136659a288502fe83d6b`. This work extracts the same scoring
arithmetic for reuse and adds the joint extension; it does not borrow the smaller
study's models or measurements. Shared baseline equivalence is tested before
the acquisition implementation is frozen for training.

## Commands and validation

The existing learner CLI exposes `acquisition prepare`, `timing`, `run` and
`report`. Preparation refuses a nonempty destination; timing binds the source
revision and source/input hashes. Runs reject changed inputs or incomplete
existing model directories rather than silently overwriting them.

```bash
# Preparation already completed in this worktree. For a fresh ignored root:
.external/cue-env/bin/python -m nursery.learner acquisition prepare \
  --root outputs/runs/acquisition_curves \
  --inputs /Users/rishisim/Documents/research/nursery/outputs/runs/step1_matched \
  --episodes /Users/rishisim/Documents/research/nursery/outputs/renders/clean_856
.external/cue-env/bin/python -m nursery.learner acquisition timing
.external/cue-env/bin/python -m nursery.learner acquisition run
.external/cue-env/bin/python -m nursery.learner acquisition report \
  --plot-python .external/acquisition-plot-env/bin/python
```

The plotting interpreter is an ignored isolated environment with Matplotlib;
it does not modify the shared training environment. Timing uses development
panels only, including shuffled development histories, and estimates final
evaluation cost conservatively. Timing weights are removed. The cap includes
timing and reserves 25% headroom in the matrix projection.

Focused tests exercise actual models, data batches, fixed-panel scoring and
backpropagation in a tiny CPU run. They cover baseline equivalence, all sensory
group gradients, zero-output initialization, clip/padding/missing-data masks,
paired initialization/order/dropout, unseen queries, complete shuffled histories,
development-only selection, timing-only panel access, frozen input/source
rejection, and complete-matrix report calculations. The historical learner
tests remain regression checks; archived joint tests do not validate the new
joint interaction by themselves.
