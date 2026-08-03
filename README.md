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
bakeoff ranked Gemini first. The sole selected generator for this pilot is
locally governed LTX-2.3. A later MiniMax-H3 open-weight selection remains
preserved in Git and in the canonical protocol but is shelved without any
weight download or inference and is now explicitly out of scope. LTX scored
19/28, MiniMax 25/28, and Gemini
27/28; neither local choice may be reported as the bakeoff winner. See
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
The generator-selection history remains explicit: Gemini led the blinded
screen at 27/28, followed by MiniMax at 25/28, Seedance at 24/28, and LTX at
19/28. LTX was selected as a practical local route, then MiniMax H3 was
prospectively selected but remained blocked and unrun, and the latest user
instruction makes LTX the sole local generator. None of these decisions
changes the bakeoff ranking, and no generator output has been produced. The
original coverage-redesign rule required the
multi-pass Real-1h positive control to pass before generation; that rule fired
and remains sealed. It is not the active authorization path for the later
descriptive extension. Synthetic data remains capped at exactly one accepted
credited hour.
The three redesigned Real-1h seeds completed. Mean realistic macro rose from
`0.51836` at initialization to `0.53602`; two seeds improved and every temporal
safeguard was non-catastrophic. The mean gain of `0.01766` nevertheless missed
the frozen `0.02` positive-signal threshold, so that exploratory stop rule
fired and remains sealed. After that result, the user prospectively authorized
a scientifically weaker descriptive Synthetic-1h LTX extension. It does not
reinterpret the failed gate. Its next conditional stages are the corrected
complete public source/module combined gate, the governed C transfer audit and
combined calibration gate, and an aggregate-conditioned public episode-plan
commitment, followed by public/dummy LTX topology sizing. Any later corpus
remains capped at exactly one accepted credited hour and cannot support
directional-competitiveness, equivalence, noninferiority, same-quality, or
confirmatory claims.
The corrected complete public source gate has now run and is a frozen no-go at
commitment `5f4aeff2…13b37`: all other independent source families passed, but
the first-person Charades-Ego order-action family yielded 44 rather than 48
items in each partition, with frozen `turn_on`/`turn_off` deficits. No model
qualification, governed C run, LTX generation, synthetic hour, or synthetic
learner result was opened. That separately blocking source-feasibility route
therefore remains stopped; the later construct-aligned amendment below does
not reinterpret it.

A prospective learner-effective amendment at `d907d247…e2855d` preserves
that no-go while prospectively treating the existing 44/44 order-action set as
a supporting, nonblocking diagnostic. The active gate still requires every one
of the five critical learner-effective axes and at least six of seven axes;
broad activity and global visual similarity are descriptive. If public and
governed C gates pass, the intended route is feature-matched episode planning,
exactly 3,600 accepted synthetic seconds, and the same three 4,668-step learner
seeds. The schema-16 amendment at `842d5a16…81a39` retains that
construct-aligned public-gate hierarchy. The current schema-17 amendment at
`cb4a7cd2…19c62` makes pinned LTX-2.3 the sole conditional
applicant-governed generator and puts H3 out of scope. The exact 44/44 action fixtures are
reported diagnostically on development and holdout but cannot gate or rescue
the seven learner-effective axes; valid action-performance failure is
nonblocking, while action integrity, privacy, provenance, and external-call
failures remain blocking. The canonical runner now implements that exact source
reuse and diagnostic role and passes focused fail-closed regression tests,
without opening a new public model outcome or overwriting the prior no-go. The
next gate is the blind applicant review and lineage-bound seal of the public
no-hand nominees; fixture preparation and model inference remain fail-closed
until that seal exists. LTX work remains conditional on combined public and
governed-C passes, episode-plan sealing, and final-topology resource preflight.
The first public-only review-preparation attempt stopped before inference when
one frame archive was silently truncated; the canonical downloader now verifies
declared response length and retries under the unchanged ceiling. This is an
engineering retry, not a scientific no-go or protocol change.
The retry uses four bounded archive workers within the already frozen four-CPU
topology; exact source selection and deterministic manifest sealing are
unchanged.
Preparation is now complete and sealed for review: 384 public frames, 48
contact sheets, zero decode failures, no restricted mount, and no model
inference. All four engineering attempts together used 6,924 wall seconds and
13.246703 GiB, within the frozen two-hour and 200-GiB ceilings. The authorized
applicant's blind visual labels and attestations are now the sole next gate.
During that still-unsealed review, the applicant directed a binary-label
semantics correction: of 195 codes already entered, 193 yes/no values were
swapped and two abstentions were retained. Development changed from 6/184/2
yes/no/abstain to 184/6/2, and holdout from 0/3/0 to 3/0/0. The queue remains
fixed at `5ba8ae3e…f2b7`; the compact label record changed from
`7dd640e1…e4fb5` to `2ffae4d1…eae8`. Review remains in progress. No review
seal, model inference, public outcome, fixture, threshold, or scientific rule
changed.

Before any material generation, one deterministic structured compiler must
map each sealed public-word episode plan into an LTX prompt. Its fixed schema
encodes child-height first-person camera behavior, scene complexity and
continuity, public noun/adjective contrasts, hand/contact/action phases,
ordered temporal beats, referent dominance or null cases, recurrence
assignments, modular German-TTS timing, and fixed negative constraints. Manual
or attempt-specific prompt improvement is prohibited. Public/dummy validation
must seal compiler/template/source hashes, LTX and text-encoder provenance,
decoding and camera/action controls, retry vocabulary, acceptance rules, exact
resource ceilings, and prompt commitments before the 3,600-second run.
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

The user then authorized one prospective domain-appropriate calibration
extractor redesign without changing either prior no-go. The single frozen
stack was EgoVLPv2 + EgoHOS + Grounding DINO/SAM 2.1 + DINOv2, retaining the
working deterministic and shared-language modules. It stopped at the pre-model
feasibility gate: the official EgoVLPv2 zero-shot Charades-Ego checkpoint URL
returned HTTP 403, so checkpoint bytes and SHA-256 could not be resolved, and
the official record did not separately state checkpoint-weight terms. No model
inference, public development/holdout, C reopening, LTX run, generation, or
synthetic training occurred.

That third failure is also preserved. The user has now prospectively authorized
one bounded activity-checkpoint selection amendment, without changing EgoHOS,
Grounding DINO/SAM 2.1, DINOv2, the strengthened public gates, or the governed-C
gate. Primary-source and actual-download screening froze exactly three
candidates before outcomes: EgoHOD EgoVideo-L zero-shot, VideoPrism-LvT-L
zero-shot, and V-JEPA 2 ViT-L with one public-only probe. Their immutable local
weight hashes and a subject-disjoint 48-development/48-holdout Charades-Ego
fixture commitment are recorded in schema 10 of the canonical proof config.
After a preserved safe-load engineering repair, the active public dependency
manifest is sealed at `5fb4a9d3…e81d`. All three candidates passed blind
one-item H100 sizing. The frozen 48-item public-development comparison then
returned a no-go: EgoHOD and VideoPrism cleared the classification floors but
failed the temporal-control floors, while V-JEPA 2 also missed classification
and shuffled-order floors. No candidate was eligible (`0/3`), under selection
commitment `e11727dd…c29d`. The holdout was not opened and governed C, LTX
preflight, generation, and synthetic training remain stopped. No substitution
or threshold relaxation is authorized.

That fourth no-go also remains sealed. A new user-authorized amendment now
targets the actual learner input instead of reopening broad context
classification: adapter-qualified yield, noun/adjective exposure,
utterance-centered referent visibility/dominance/ambiguity, cross-episode
recurrence, adjective–attribute contrast, hand/action coupling, and the
egocentric sensor regime. Broad activity/context is descriptive only, and
temporal sensitivity is tested only on localized inverse actions whose labels
depend on order. Schema 11 freezes the modular public gates, one-applicant
governed C transfer audit, missingness/matching rules, and one-H100 resource
ceiling before any new outcome. An implementation clarification anchors English
mentions to their accepted translation segment because OPUS-MT emits no
English-token alignment; no timestamp is fabricated. The active commitment is
`c9a48206…adaf`. The public artifact subgate passed at `8c787a01…b527`
without model inference or a restricted mount. The exact 53-package overlay
first passed preparation at `968f2570…1f10`, also without inference or a
restricted mount. The exact local overlay and compatibility
surface first advanced under commitment `ee70ae31…b41b`. A sizing attempt
stopped before model inference because the sealed NLTK resources needed their
standard `taggers/` and `corpora/` namespace; the repair verifies every sealed
hash and adds only scratch-local symlinks. The repaired overlay was resealed at
`9810a618…48f9`. The retry then stopped before Grounding DINO construction on
an unused visualization import. Active commitment `623225bf…09e4` removes only
that exact import after original/patched source-hash verification and was then
resealed before label-blind sizing resumed. That reseal passed at
`03c15506…2c15`, with no inference or restricted mount. A later label-blind
retry froze exact Grounding DINO padding-sentinel validation at
`afc936f7…a2d5`, then passed through DINOv2 before the pinned EgoBabyVLM package
initializer required missing Submitit. Active runtime commitment
`eb878d8c…fbea` adds only hash-pinned `submitit==1.5.3` and its absent
`cloudpickle==3.1.1` dependency. The 55-package overlay passed reseal at
`df15ff20…c0c4`, without model inference or a restricted mount. The subsequent
eight-module label-blind sizing passed at `b6275905…e029`, with no fixture
outcome or retained prediction. No
fixture outcome, model substitution, or gate change is involved. Under that
now-historical route, C measurement, LTX generation, and Synthetic-1h training
remained conditional. The one-hour cap
and descriptive-only claim boundary are unchanged.

The action-control implementation is prospectively frozen at fixture-protocol
commitment `506a1f41…251d`: eight inverse direction labels use fixed three-prompt
ensembles and subject/video-disjoint localized Charades-Ego development and
holdout clips. This is a genuinely order-dependent control, not a revival of
the failed broad-context temporal gate. Exact task-matched fixture manifests
remain pending and must be sealed before public development inference. The
source-selection and rendering recipe is separately frozen at
`1cc8d0e3…ff1d`: six task-matched families contain 312 items per partition,
with official COCO archive bytes pinned, selective VISOR files hash-recorded,
the existing Charades-Ego identities retained, and self-authored German audio
sealed outside Git. This preparation stage has opened no model outcome and
uses no restricted mount.

The first render preparation stopped before media creation because the frozen
3% COCO source-area floor under-yielded development sports-ball crops. The
stop is preserved. Prospective repair `e5fd286e…7048` removes only that
non-estimand floor while retaining the 48-pixel source-detail floor, then
requires one complete annotation-only COCO/VISOR/Charades feasibility pass
before any render or model inference.

That complete annotation-only feasibility run then fired its frozen
fixture-source no-go. Under the committed shared video cap and stratum order,
the VISOR development partition contained 16 hand/contact items, zero
hand/no-contact items, and 12 true-no-hand items; the hand/no-contact stratum
required 12. Holdout had the same 16/0/12 aggregate counts, and all checked
subject/video/object overlap counts were zero. Record commitment
`dee0a375…13e7` verifies the compact result. No fixture media or model outcome
was produced, the Charades archive was not downloaded, and public development,
governed C, LTX, and Synthetic-1h training were not opened. That no-go remains
final for its source semantics and sequential shared-cap allocation.

The user then prospectively authorized a new, narrowly scoped VISOR-HOS
fixture correction at commitment `31c1c26f…1bf8d4`, before any new source
inventory or extractor outcome. It freezes the official 158-file VISOR
train-plus-validation annotation set, explicit contact versus explicit
`hand-not-in-contact` truth, a separate visually verified no-hand task,
participant-first splitting, and a simultaneous order-invariant 48/48/48
contact/no-contact/no-hand allocation in each public partition. The official
VISOR-HOS repository is an unlicensed semantic reference only: its code is not
copied or executed. That frozen correction required public qualification to
collect every independent module metric and then apply one decision: all five
critical learner-effective axes, at least six of seven axes overall, and the
separate genuinely order-dependent action control had to pass. Broad activity
was descriptive. Only that combined public PASS could authorize the governed C
transfer audit; only a subsequent combined C PASS could authorize the
then-selected exactly-one-accepted-hour descriptive LTX arm.
The complete source stage under that correction is now sealed as
`NO_GO_COMPLETE_SOURCE_FEASIBILITY` at `5f4aeff2…13b37`: the order-action
family retained only 44/44 items and missed frozen `turn_on`/`turn_off`
direction quotas. The conditional public-model, governed-C, and LTX stages
therefore remained unrun. H3 and its license blocker are preserved as history
but are out of scope for this pilot; the current conditional route uses LTX-2.3
as the sole selected generator, subject to the public and governed-C gates.
