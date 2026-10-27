# Current Nursery plan

## Active objective

Measure direct language-acquisition curves for video-only (V) versus the same
video plus bodily inputs (V+B), across amounts of independent training
experience. The long-term hypothesis is a higher language-learning ceiling;
the existing experiments neither establish that benefit nor rule it out.

## Current step and next direction

Prepare a matched acquisition-curve comparison. Count independent source clips,
not replayed frames, as the training-data axis; use nested training subsets and
give both arms exactly the same video, temporal history and weak speech at each
size. Preserve natural weak/delayed/nonheld naming rather than selecting normal
learning examples with hidden referent identity.

Before new training, fix the split and subset membership, paired training seeds,
evaluation trials, training budget at each size, and dev-only checkpoint
selection rule. Keep test experience outside every training subset. If
pretraining is used, confine it to that size's training subset and match the
visual experience and declared optimization budget across arms. Specify how
pretraining cost enters that budget before comparing curves.

The primary outcome is understanding with body available: V+B versus the
separately trained V learner on identical held-out situations and candidates.
Picture-only transfer after learning is secondary and reported separately.
Distinguish faster learning or better performance at limited data from a higher
eventual ceiling. A fixed-data epoch plateau or one favorable data size cannot
establish the latter; insufficient saturation leaves it unresolved.

The working 512-dimensional additive learner over frozen DINO features is the
reference baseline. DINO supplies prior visual knowledge; these experiments do
not start visual learning from scratch. Joint visual/body fusion with optional
sensory pretraining is under research, not a chosen implementation or an
approved training run. The old 128-dimensional joint pilot changed several
factors and does not settle that choice.

## Preserved starting points

- Keep `outputs/renders/clean_856/`, saved speech/features and past run roots in
  place. They are retained inputs/evidence, not an active experiment queue.
- Read [Past experiments](archive/language_learning/past_experiments.md) before
  designing a new comparison; [experiments.md](archive/language_learning/experiments.md)
  remains the detailed historical protocol/result record.
- Use [the learner reference](docs/learner.md), [speech evidence](docs/speech_coupling.md)
  and [design intent](archive/language_learning/design_intent.md) for existing
  implementation and weak-alignment constraints. The [archive](archive/README.md)
  also holds earlier simulator work and the historical literature review.
