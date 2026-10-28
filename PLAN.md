# Current Nursery plan

## Active objective

Measure direct language-acquisition curves for video-only (V) versus the same
video plus bodily inputs (V+B), across amounts of independent training
experience. The long-term hypothesis is a higher language-learning ceiling;
the existing experiments neither establish that benefit nor rule it out.

## Current step and next direction

The matched 45-run acquisition comparison is complete. At full data, video-only,
additive and joint learners scored 60.1%, 54.4% and 54.3% on new-clip noun
understanding. The joint interaction did not deliver the desired gain, and no
acquisition ceiling was established. The [canonical result record](docs/acquisition_curves.md#completed-results--27-october-2026)
contains the full curves, controls, uncertainties and implications for the next
build decision. No further experiment is queued by that record.

The comparison counts independent source clips, not replayed frames, as the
training-data axis. It uses nested training subsets and gives all three learners
exactly the same video, temporal history and weak speech at each size. It
preserves natural weak/delayed/nonheld naming rather than selecting normal
learning examples with hidden referent identity.

Before new training, fix the split and subset membership, paired training seeds,
evaluation trials, training budget at each size, and dev-only checkpoint
selection rule. Keep test experience outside every training subset and match
the visual experience and declared optimization opportunity across arms.
These first two studies use direct language training without sensory pretraining.

The primary outcome is understanding with body available: V+B versus the
separately trained V learner on identical held-out situations and candidates.
Picture-only transfer after learning is secondary and reported separately.
Distinguish faster learning or better performance at limited data from a higher
eventual ceiling. A fixed-data epoch plateau or one favorable data size cannot
establish the latter; insufficient saturation leaves it unresolved.

The working 512-dimensional additive learner over frozen DINO features is the
reference baseline. The implemented joint learner preserves direct visual–word and
body–word paths and adds a learned interaction. DINO supplies prior visual
knowledge; these experiments do not start visual learning from scratch. The
first comparison excludes sensory pretraining. The old 128-dimensional joint
pilot changed several factors and does not settle this design.

The two design references are the [main acquisition comparison](docs/acquisition_curves.md)
and [small cross-situational study](docs/cross_situational_learning.md). They can
run in parallel after shared implementation verification, with separate models
and artifacts. The small study compares video-only with additive video + body
at one fixed data amount, reporting a two-row accuracy table and a supporting
shuffled-body evaluation check; it has no joint arm or acquisition curves.
The main study retains its three learners and acquisition curves.
The main document now includes completed results. Dataset feasibility, frozen
run settings and runtime limits remain prerequisites for any subsequent study.

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
