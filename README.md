# Synthetic Video

Phase 4 uses the prospectively frozen public-only language gate in
`configs/synthetic_video_language_gate.json` and
`docs/synthetic_video_phase4.md`. Stage A selected one immutable offline
pipeline; governed Stage B common-asset construction remains the only current
authorization involving ChildLens-derived assets.

A separate public-only qualitative generator preview is complete. It produced
all eight frozen Wan/LTX clips with the public Piper track, retained the
qualitative failures, and used no ChildLens/BabyView input or derivative. See
[`docs/synthetic_video_public_pilot.md`](docs/synthetic_video_public_pilot.md).

The scientifically distinct, public-only LTX-2.3 versus Seedance 2.0
quality-ceiling comparison is complete. Seedance was visibly cleaner but did
not clear the prospectively frozen material-improvement threshold, so LTX
remains the reproducible baseline. Seedance training use remains blocked
pending written provider and institutional clearance. See
[`docs/synthetic_video_quality_ceiling.md`](docs/synthetic_video_quality_ceiling.md).

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

Raw or restricted ChildLens material must remain in the encrypted
applicant-only local store or applicant-private UTD-managed Juno paths. It may
not be sent to cloud services, model APIs, hosted GPU services, Git, telemetry,
or unauthorized annotators. Only permitted aggregates may leave the governed
boundary.

## Repository state

The branch was intentionally reset after commit `7571c0e`. Legacy simulator,
AEA, grounding, and experiment implementations remain recoverable through Git
history but are not part of the active synthetic-video architecture. See
[`docs/prior_work.md`](docs/prior_work.md) for the small set of prior findings
that may inform future work.

The architecture review freezes EgoBabyVLM CLIP+ in `triple` mode as the
learner; Machine-DevBench Lexical as the common standardized primary endpoint;
an above-chance, positive real-only learning-curve readiness gate; and a
separate held-out-real ChildLens temporal-transfer safeguard. It retains the
modular video and German-TTS architecture, Wan 2.2 TI2V-5B and LTX-2 as the
reproducible public-pilot generator candidates, and a prospective
no-substitution pilot protocol. Seedance is evaluated separately as a hosted
quality ceiling whose training use requires an additional terms/governance
gate. See
[`docs/synthetic_video_architecture_review.md`](docs/synthetic_video_architecture_review.md).
The public/dummy single-L4 EgoBabyVLM compatibility preflight is complete and
passed as engineering evidence only; DDP remains untested.
Phase 3 governance/preregistration is **PASS**.
MPI accepted the signed project-specific request and granted non-commercial
ChildLens model calibration/evaluation and aggregate reporting through July
2027; third-party access remains prohibited. Permission is established. The
local ChildLens corpus is now in a checksum-verified AES-256 sparsebundle.
Inventory fixes 18 prior development children / 14.374241 source hours as
\(C\), leaving 40 untouched catalog children / 134 recordings / 40.362056
source hours for confirmatory allocation. The Juno account and SSH key are
active, with applicant-only home/work/scratch permissions and visible
A30/H100/H200 capacity. Official Juno self-study orientation is reviewed. Juno
is qualified for applicant-only ChildLens processing under the proportionate
UTD policy controls frozen in the preregistration. Public egress is available
for pinned dependency ingress, but ChildLens and derived restricted artifacts
may not be sent to APIs, hosted services, Git, cloud storage, telemetry, or
other third parties. The missing `yding` SLURM association is a non-blocking
fair-share/accounting correction. Post-pass gates are in
[`docs/synthetic_video_preregistration.md`](docs/synthetic_video_preregistration.md).
A separately scoped Phase 4 common-asset task may begin. The bounded
public-only ASR/translation selection is complete. LTX/Wan and TTS execution
are authorized only for the self-authored qualitative preview described above;
governed or ChildLens-derived generator work, real-only training, and
scientific evaluation remain unauthorized.
