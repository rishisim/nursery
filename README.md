# Synthetic Video

This branch is the clean workspace for the Nursery/BabyWorld synthetic-video
research program.

## Scientific question

Can prompt- and plan-conditioned child-view video with synchronized language
audio become a useful, scalable complement to natural developmental
egocentric video?

The initial claim is deliberately narrow: synthetic augmentation may reduce
the amount of real data needed to reach a fixed downstream learning result.
It is not assumed to replace real experience.

## Proposed method

A frozen structured episode plan will control scene content, persistent object
identity, egocentric camera behavior, actions, utterance timing, and visible
referents. Video and audio may be produced by separate components driven by
the same timeline. Simulator-derived geometry and sensor streams are not
required for the primary study.

The two primary evaluation families are:

1. Blinded real-versus-synthetic discrimination and supporting distributional
   diagnostics.
2. Matched-protocol downstream learning comparisons evaluated on held-out real
   data, including real-only, synthetic-only, and real-plus-synthetic
   conditions.

## Empirical boundary

ChildLens is the sole empirical child-data source for the current prototype.
It contains naturalistic child-centered recordings from ages 3–5 and is not
infant ground truth. BabyView is unavailable and must not be accessed or
claimed as an empirical target until its access and governance requirements
are satisfied.

Raw or restricted ChildLens material must remain local. It may not be uploaded
to cloud services, model APIs, or unauthorized annotators. Only permitted
aggregates may leave the governed boundary.

## Repository state

The branch was intentionally reset after commit `7571c0e`. Legacy simulator,
AEA, grounding, and experiment implementations remain recoverable through Git
history but are not part of the active synthetic-video architecture. See
[`docs/prior_work.md`](docs/prior_work.md) for the small set of prior findings
that may inform future work.

No generator, learner, evaluation protocol, or model dependency is currently
selected. Those choices should be frozen prospectively before an experiment is
run.
