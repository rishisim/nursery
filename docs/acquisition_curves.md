# Main experiment: language acquisition with video and body

Design reference; no implementation or training has started. The
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
