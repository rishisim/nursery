# Synthetic Video

This branch is the clean workspace for the Nursery/BabyWorld synthetic-video
research program.

## Scientific question

Can \(H\) hours of prompt- and plan-conditioned child-view video with
synchronized language audio produce lexical-grounding performance equivalent
to \(H\) hours of natural developmental egocentric video, while costing less to
produce than comparable new real-data collection?

The initial claim is deliberately narrow: **equal video hours, equivalent
lexical-grounding performance, and lower measured cost**. The primary
comparison is synthetic-only versus real-only at the same \(H\)-hour learner
budget. Mixed real-plus-synthetic and reduced-real analyses are secondary; the
study does not primarily hypothesize that synthetic video reduces how much
ChildLens data the learner needs, and it does not claim synthetic experience is
the same as real linguistic acquisition.

## Proposed method

A frozen structured episode plan will control scene content, persistent object
identity, egocentric camera behavior, actions, utterance timing, and visible
referents. Video and audio may be produced by separate components driven by
the same timeline. Simulator-derived geometry and sensor streams are not
required for the primary study.

The claim is evaluated with:

1. one common generated-image Machine-DevBench Lexical asset for the primary
   standardized equivalence test;
2. one child/session-disjoint held-out-real ChildLens temporal-transfer
   safeguard, reported separately because Machine-DevBench images are
   generated; and
3. a prospective like-for-like real-versus-synthetic cost ledger, with blinded
   fidelity and distributional diagnostics reported separately.

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

The architecture review now freezes EgoBabyVLM CLIP+ in `triple` mode as the
learner; Machine-DevBench Lexical as the common standardized primary endpoint;
an above-chance, positive real-only learning-curve readiness gate; and a
separate held-out-real ChildLens temporal-transfer safeguard. It retains the
modular video and German-TTS architecture, Wan 2.2 TI2V-5B and LTX-2 as the
only public-pilot generator candidates, and a prospective no-substitution pilot
protocol. See
[`docs/synthetic_video_architecture_review.md`](docs/synthetic_video_architecture_review.md).
No generator pilot has been executed and no generator dependency has been
installed. The public/dummy single-L4 EgoBabyVLM compatibility preflight is
complete and passed as engineering evidence only; DDP and restricted-data
readiness remain unproven. Phase 3 safe governance/preregistration decisions
are now frozen, but its status is **INFRASTRUCTURE/PERMISSION GATE**, not PASS.
MPI accepted the signed project-specific request and granted non-commercial
ChildLens model calibration/evaluation and aggregate reporting through July
2027; third-party access remains prohibited. Permission is established. The
current ChildLens sparsebundle is unencrypted, so encrypted-storage remediation
and governed CUDA qualification remain infrastructure gates.
Exact unblock actions are in
[`docs/synthetic_video_preregistration.md`](docs/synthetic_video_preregistration.md).
Common assets, LTX/Wan, TTS, ChildLens preprocessing, generation, and learner
training remain unauthorized.
