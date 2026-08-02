# Synthetic Video

Phase 4 uses the prospectively frozen public-only language gate in
`configs/synthetic_video_language_gate.json` and
`docs/synthetic_video_phase4.md`. Stage A selected one immutable offline
pipeline. The first Stage B build is preserved as provisional engineering
evidence but is scientifically superseded pending validity repair.
Confirmatory Phase 5 is not authorized.

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
or unauthorized annotators. Only compact non-identifying aggregates that pass
the frozen disclosure review may leave the governed boundary; those aggregates
may condition a public episode-plan distribution without exposing ChildLens
media, text, vocabulary, identifiers, row-level values, embeddings, or
reconstructive combinations.

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
modular video and German-TTS architecture. A completed four-family public-only
bakeoff ranked Gemini first, but the latest prospective user amendment selects
MiniMax H3 for the lean pilot. MiniMax scored 25/28 versus Gemini's 27/28 and
must not be reported as the bakeoff winner. See
[`docs/synthetic_video_architecture_review.md`](docs/synthetic_video_architecture_review.md).
A public generator pilot and bakeoff have been executed without ChildLens or
BabyView inputs. The corrected governed common assets passed on the final
two-H100 DDP topology. The original 570-step Real-1h/Real-3h pilot remains
sealed as stopped, but it used fewer than one contrastive-equivalent pass over
Real-1h. A separate prospective redesign now freezes three matched seeds and
778 complete 4:1:1 cycles (4,668 steps; 3,112 contrastive updates), equivalent
to about 5.003 passes over each 1,244-record one-hour learner manifest.
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
The corrected Phase 4 common-asset record is
[`results/synthetic_video_phase4.json`](results/synthetic_video_phase4.json).
Its deviation ledger is
[`results/synthetic_video_phase4_deviations.json`](results/synthetic_video_phase4_deviations.json).
The latest prospective user choice selects locally governed LTX-2.3 for the
redesigned pilot. This is a practical execution choice, not a quality claim:
Gemini led the blinded screen at 27/28, followed by MiniMax at 25/28, Seedance
at 24/28, and LTX at 19/28. No generation may begin unless the redesigned
multi-pass Real-1h positive control passes, followed by governed C calibration
and a bounded final-topology LTX preflight. Synthetic data remains capped at
exactly one accepted credited hour.
The three redesigned Real-1h seeds completed. Mean realistic macro rose from
`0.51836` at initialization to `0.53602`; two seeds improved and every temporal
safeguard was non-catastrophic. The mean gain of `0.01766` nevertheless missed
the frozen `0.02` positive-signal threshold, so that exploratory stop rule
fired and remains sealed. After that result, the user prospectively authorized
a scientifically weaker descriptive Synthetic-1h LTX extension. It does not
reinterpret the failed gate. Its next conditional stage is bounded governed
C calibration and a public episode-plan commitment, followed by public/dummy
LTX topology sizing. Any later corpus remains capped at exactly one accepted
credited hour and cannot support directional-competitiveness, equivalence,
noninferiority, same-quality, or confirmatory claims.
Restricted execution was paused after a schema-inspection command
unintentionally printed 94 opaque asset-level SHA-256 keys to the Codex task
output. No media, text, filenames, paths, child/session keys, direct identifiers,
embeddings, prompts, or outcomes were exposed, and the keys were not copied into
Git. The authorized applicant/data-steward reviewed the incident, required no
additional reporting or remediation, and authorized processing to resume. An
aggregate-only containment check found zero restricted-key matches in tracked
Git content, and future governed inspection uses a flat whitelisted reporter.
The resumed bounded C calibration then stopped the descriptive extension. Five
of eight axes exceeded the frozen 20% missingness ceiling; the critical
audiovisual-grounding axis had 41.25% missingness, and only three axes were
measured within the ceiling. Its governed target and provisional plan
commitments are retained, but the plan is non-executable. The frozen threshold
was not relaxed and the estimator was not replaced. LTX topology preflight,
generation, and Synthetic-1h learner training were not run.
After that no-go was sealed, the user prospectively authorized one extractor
repair. The original result and governed files remain preserved. A single
candidate—PE-Core prompt ensembling plus pinned Apache-2.0 OWLv2 object/hand
detection—is frozen before new C outcomes. It must first pass an eight-image
public-only qualification with no restricted mount. That gate returned a
no-go: activity was 7/8, expected objects 8/8, hand positives 4/5, all eight
proxy rows were complete, and boxes were valid, but only 1/3 hand-negative
fixtures stayed negative (2/3 required). No thresholds or labels were changed,
no second detector was tried, and C was not reopened. Governed C repair, LTX
preflight, generation, and synthetic learner training remain stopped.
Confirmatory Phase 5, multi-seed equivalence inference, and full-corpus
generation remain unauthorized.
