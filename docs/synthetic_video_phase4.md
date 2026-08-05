# Synthetic-video Phase 4 common evaluation assets

**Status:** **CORRECTED COMMON ASSETS PASS; H100 ENGINEERING-HEALTH RESOURCE
REDIRECT FROZEN; PRIOR LEARNER-EFFECTIVE NO-GOS PRESERVED** — Stage A remains PASS; the first Stage B assets remain
provisional/superseded, and the corrected lexical and temporal assets are
sealed. The prior mechanistic tuple-calibration fixture recipe fired its frozen
public source no-go and remains final. The later user-authorized VISOR-HOS
correction also reached a combined source no-go because the frozen first-person
order-action family missed exact yield. This is not confirmatory Phase 5; no
governed-C rerun, generator work, or synthetic arm ran.

**Scope:** public-only language-pipeline qualification followed, only on a
signed PASS, by governed construction and sealing of the two common evaluation
asset families. This phase excludes generator/TTS execution, synthetic
generation, learner training, score opening, and scientific evaluation.

## Stage A: prospectively frozen public-language gate

The closed candidate family contains exactly two official multilingual ASR
checkpoints loaded by `openai-whisper==20250625`: `base.pt` at SHA-256
`ed3a0b6b…2e34e` and `small.pt` at SHA-256 `9ecf7799…e794`.
Both use German transcription with word timestamps and the same
`Helsinki-NLP/opus-mt-de-en@1a922f3b32a8e809e17a47d4b32142d8105924e5`
translator. Whisper is MIT-licensed and the translator is Apache-2.0. No
candidate may be added after outcomes are observed.

The public ASR set is the first recording for each of the first eight distinct
sentence IDs in the immutable German FLEURS test manifest at
`google/fleurs@70bb2e84b976b7e960aa89f1c648e09c59f894dd`
(CC-BY-4.0). The self-authored set is synthesized locally with the fixed
macOS `Anna` German voice and includes child-directed vocabulary, a long
utterance, 10 dB noise, overlapping speech, and silence. It is used for
translation adequacy and edge-case behavior, not to replace public-set WER.

Before any inference, the following family gate is frozen:

- public-set corpus WER at most 0.45;
- self-authored translation chrF at least 0.45;
- mean non-abstained word confidence at least 0.35;
- abstention rate at most 0.20;
- 100% monotonic, in-bounds word timestamps on non-abstained items, with at
  most 50 ms numeric serialization error;
- 100% ID/word/timestamp round trip and manifest completeness;
- no crash or silent truncation;
- complete immutable revision, license, and SHA-256 manifests;
- successful offline reload with telemetry and external tracking disabled.

These deliberately broad thresholds reject unusable pipelines without treating
a small public sample as a benchmark claim. Selection among passing candidates
is lexicographic: lower public corpus WER, higher translation chrF, then the
smaller checkpoint resource rank. A candidate abstains on empty output for
nonsilent speech, missing/invalid timestamps, mean word confidence below 0.35,
language mismatch, or translation failure. Silence is expected to abstain and
is counted in the common maximum-abstention rate.

Decoding is deterministic: German transcription, temperature 0, beam size 5,
word timestamps enabled, and prior-text conditioning disabled. This was made
explicit after a pre-decision engineering rerun exposed Whisper's default
temperature fallback; candidates, data, thresholds, and selection order were
unchanged, and only results from the repaired frozen config are admissible.

The run is limited to one local public-only CPU/MPS process, six wall-clock
hours, 5 GiB download/model storage per frozen family, and zero paid cost. This
cannot trigger the DDP gate. Data, weights, audio, caches, logs, predictions,
and run manifests live only under ignored run roots.

If neither candidate passes, Phase 4 stops with a signed no-go. No candidate is
added and no threshold is relaxed. If one passes, its exact revision and
artifact hashes become the sole identical real/synthetic language pipeline.
Only then may Stage B begin.

### Stage A decision

Both frozen candidates passed. `whisper-small-opus-de-en` was selected by the
first lexicographic criterion: public-set corpus WER 0.147059 versus 0.223529
for `base`. Its translation chrF was 0.584424, mean word confidence 0.910202,
abstention rate 0.0769, timestamp-valid fraction 1.0, and round-trip fraction
1.0. A second deterministic aggregate rerun matched every gate metric and the
selected candidate. Local-files-only reload, complete artifact/license hashes,
and disabled telemetry/tracking passed. These are compact qualification
aggregates, not ChildLens or scientific outcomes. The signed compact decision
is `results/synthetic_video_language_gate.json`.

## Stage B boundary

Stage B must execute inside applicant-private UTD-managed storage. Its
repository-facing implementation may contain schemas, validators, public
source metadata, and commitment verification only. ChildLens media, audio,
text, filenames, identifiers, row-level manifests, embeddings, derived
prompts/statistics, restricted vocabulary, and governed paths must never enter
Git or terminal output. Compact non-identifying commitments are the only
permitted repository result.

One shared Machine-DevBench Lexical manifest and one shared held-out-real
temporal retrieval manifest must be sealed before learner training. Contract
validation requires all later arms to reference identical commitment hashes.
Machine-DevBench images are generated and are not held-out-real ChildLens
evaluation. The temporal safeguard is model-derived, supplies no referent
ground truth, and has no human German validation.

### Stage B resource gate

Applicant-only Juno SSH and the encrypted local ChildLens mount are reachable,
but no restricted corpus is currently staged in the Juno governed store. The
pinned upstream Machine-DevBench image path uses FLUX.2-klein-4B and documents
four-way GPU execution. Before restricted multi-process work, the frozen
controls require an approved public/dummy final-topology preflight. The
approved ceiling was 4 × Juno A30 GPUs for at most 30 minutes, followed only
on PASS by at most 4 × A30 for 12 hours. Juno exposes two two-GPU A30 nodes,
but one complete node is reserved through 2026-08-05. Because image generation
is independently sharded rather than DDP-coupled, the runnable final topology
was initially frozen at 1 node × 2 A30s. The user subsequently authorized the
technically appropriate Juno GPU type under the same time limits. H200 was
selected because FLUX fits comfortably and Juno exposes many two-GPU H200
nodes, preserving the single-node/two-shard topology while avoiding the A30
reservation. The final request is therefore 2 × H200 for a 30-minute
public/dummy preflight (1 aggregate GPU-hour), then at most 2 × H200 for
12 hours (24 aggregate GPU-hours) for the governed build. The unavailable A30
requests consumed no allocation. The two-process H200 public/dummy preflight
passed in 2:13 with commitment `3271b7b8…c4b7`. The governed build then ran
under the same final topology and offline controls.

### Provisional Stage B engineering record

Whole-child allocation retained all 18 C children and assigned the remaining
40 children as 28 training, 8 evaluation, and 4 validation. The shared lexical
asset contains 30 nouns and 24 adjectives in both official styles; its 524
manifest/image files have commitment `68f9490a…d5c1`. The held-out-real
temporal safeguard contains 3,672 eight-candidate query rows and 14,688 frames
from 8 evaluation children across 19 sessions; its 14,689 manifest/frame files
have commitment `0261dca9…45f`. Calibration/evaluation child overlap is zero,
all candidate counts are exact, 145 duplicate frames are recorded and retained,
and five complete query rows were excluded without substitution after a frame
decode failure. Every later arm referenced exactly these two commitments, but
they are now `PROVISIONAL_SUPERSEDED_PENDING_REPAIR`, not accepted assets.

The governed manifests retain row-level provenance, hashes, filters, and audits;
Git retains only the compact non-identifying decision in
`results/synthetic_video_phase4.json`. Machine-DevBench images are generated,
not held-out-real ChildLens evaluation. Temporal labels remain model-derived,
with neither referent ground truth nor human German validation.

The audit found validity-critical deviations in the keyed allocation deal,
shared abstention adapter, cross-split overlap audit, and pinned-upstream
lexical loader/filter path. The repair is outcome-independent because only
aggregate engineering and asset-count information was observed; no learner or
scientific score was opened. The allocation repair is paused before assignment
because Phase 3 did not freeze the byte value/encoding of `study_id` in
`HMAC(key, study_id || child_id)`; this must be resolved prospectively rather
than inferred from provisional assets. Confirmatory Phase 5, generator/TTS execution,
synthetic generation, score opening, and final evaluation remain unauthorized.

### Corrected asset and active exploratory status

The allocation ambiguity and the other audit findings were subsequently
resolved without using learner outcomes. The corrected common assets passed:
the lexical commitment is `3798fafc…17b4` and the held-out-real temporal
commitment is `3cc29f32…e46e`; the original commitments above remain
provisional/superseded. Later exploratory real-only runs and four preserved
calibration no-gos are recorded in the canonical preregistration and compact
Phase 4 result. Schema 11 now freezes a distinct mechanistic training-tuple
calibration amendment at commitment `c9a48206…adaf`. At that amendment point,
only public qualification was authorized. Its public artifact manifest passed at
commitment `8c787a01…b527` with seven components, four immutable repository
archives, and ten weight/resource files (14.62 GB), without model inference or
a restricted mount. The EgoHOS bundle is distributed by the official
MIT-licensed project but contains no separate archive license file; this record
asserts academic local prototype use only and no broader license conclusion.
The 53-package runtime overlay then passed resource-only preparation at
commitment `968f2570…1f10`, with neither a restricted mount nor model
inference. That runtime record is preserved but was superseded under active
runtime commitment `ee70ae31…b41b`: a label-blind sizing attempt stopped before
model load because the official NLTK archives needed scratch-local
`taggers/`/`corpora/` namespace symlinks. Every source resource hash is checked
before those links are made. The active runtime was resealed at
`9810a618…48f9`, again without model inference or a restricted mount.
The retry passed language, lexical, and sensor checks but stopped before
Grounding DINO construction because its model file imports an unused
visualization class that pulls in unpinned Matplotlib. Active runtime commitment
`623225bf…09e4` verifies the exact original source and removes only that
unreferenced import. The active runtime was resealed at `03c15506…2c15`, again
without inference or a restricted mount; local-files-only model reload and
blind sizing may resume. The next label-blind attempt reached the pinned
Grounding DINO forward pass and stopped because the previous generic raw-output
check treated the model's documented negative-infinity text-padding sentinels
as numeric failure. No fixture label, prediction, score, or scientific metric
was retained. Before retry, commitment `afc936f7…a2d5` froze the exact rule:
active tokenizer positions must be finite, only the exact tokenizer-padding
complement may be negative infinity, post-sigmoid scores must be finite, and
normalized boxes must be finite within `[0,1]`. This is an engineering sizing
guard and changes no model, fixture, threshold, or scientific gate.
That retry then passed the adapter, lexical, sensor, Grounding DINO, SAM 2.1,
and DINOv2 reloads before the pinned EgoBabyVLM alignment package initializer
required `submitit`; PE-Core had not yet been constructed. The base container
also lacks `cloudpickle`, the sole required runtime dependency of the selected
Submitit wheel. No fixture outcome was opened. Active runtime commitment
`eb878d8c…fbea` therefore adds only hash-pinned public `submitit==1.5.3` (MIT)
and `cloudpickle==3.1.1` (BSD-3-Clause). The prior 53-dependency runtime and its
commitment remain preserved. The 55-dependency overlay then passed preparation
in 9 minutes 16 seconds at commitment `df15ff20…c0c4`, with no model inference
or restricted mount. Final-runtime label-blind sizing then passed all eight
modules in 55.37 seconds at 0.70 GiB peak VRAM, with zero failures, external
calls, retained predictions, or scientific metrics; commitment
`b6275905…e029` verifies the record. This opens only public task-matched fixture
preparation and qualification at that stage.
Governed C measurement required that pass, and LTX generation remained further conditional and capped at exactly one
accepted credited synthetic hour. No confirmatory/equivalence claim is
authorized.

The task-matched fixture implementation is separately frozen before model
outcomes at commitment `506a1f41…251d`. Its action control uses fixed
open/close, take/put, sit-down/stand-up, and turn-on/turn-off prompts and
localized Charades-Ego intervals disjoint from the failed broad-context study.
Exact public fixture manifests were still pending and had to be sealed before
development inference. Preparation commitment `1cc8d0e3…ff1d` fixed the
source archives, partition selectors, self-authored German audio recipe,
deterministic composite/recurrence/sensor renderers, VISOR strata, and exact
48-per-direction-control sampling without opening a model outcome. Each of the
six fixture families has fixed development and holdout counts totaling 312
items per partition. Media, public row manifests, annotations, and audio stay
outside Git; only the eventual compact counts and complete-manifest commitment
may be retained.

Preparation job `313969` then stopped before any render or model inference:
the 3% COCO source-frame-area floor could not provide four development
sports-ball crops. That engineering stop remains recorded. Prospective repair
commitment `e5fd286e…7048` removes only the source-area floor because every
accepted alpha-mask crop is resized to fixed authored-composite geometry; the
48-pixel bbox floor and all scientific gates remain unchanged. A new
annotation-only feasibility preflight must verify all COCO, VISOR, and
Charades yields and zero cross-partition source overlap before rendering.

The complete annotation-only feasibility preflight subsequently stopped at
the frozen source-yield rule. The VISOR development hand/no-contact stratum had
zero eligible items against 12 required; development and holdout each had
aggregate hand/contact, hand/no-contact, and true-no-hand counts of `16/0/12`.
All checked subject/video/object overlap counts were zero. Compact record
commitment `dee0a375…13e7` verifies the result. No fixture media was rendered,
no model inference or public model outcome was opened, the large Charades video
archive was not downloaded, and governed C, LTX, generation, and synthetic
training were not run. This exact source no-go remains final for that frozen
relationship-derived label and sequential shared-cap recipe.

### Prospective VISOR-HOS fixture correction

Before any new source inventory or extractor outcome, the user authorized a
narrow correction at commitment `31c1c26f…1bf8d4`. It does not reinterpret the
`dee0a375…13e7` no-go. The correction pins the official VISOR DOI deposit's
115 training and 43 validation annotation JSONs (868,821,446 bytes) under a
locally resolved external-manifest commitment `771ce947…c095`. The deposit is
handled as CC-BY-NC-4.0 noncommercial academic data. The official VISOR-HOS
repository is pinned only as a semantic reference because it contains no
license file; repository code is neither copied nor executed. Its two relevant
conversion files are independently hash-pinned so their identities cannot be
conflated.

Contact truth is now per visible left/right hand instance: a resolved object
identifier is contact, exact `hand-not-in-contact` is no-contact, and
none-of-the-above, inconclusive, missing, null, malformed, or unresolved
relations abstain. Hand presence and contact state are separate tasks. No-hand
frames never enter contact F1 and must be visually verified before inference;
annotation absence merely nominates a frame. Participants are split first by
the frozen public-seed hash deal. A simultaneous order-invariant sampler then
targets 48 contact, 48 explicit no-contact, and 48 verified no-hand fixtures in
each partition, with a cap of four per video and stratum, one selected item per
frame, and zero participant/video/frame overlap.

Before development inference, the execution clarification also fixes the
nine-sample referent aggregation, exact lexical stage semantics, alpha-masked
DINOv2 recurrence input, pHash threshold, PE-Core attribute prompts,
single-factor sensor comparisons and maximum-motion cut statistic, target-side
EgoHOS mapping with the official keep-ratio mmseg test pipelines, and
sixteen-frame ordered/reversed/repeated action controls. The repaired public
audio seed uses attributive adjective–noun phrasing and is commitment-bound;
the earlier predicative seed is preserved but rejected as stale. These are
executable metric definitions, not post-outcome threshold changes.

The public stage must run every independent source and model check before one
combined decision. All five critical learner-effective axes, at least six of
seven axes overall, and the separate genuinely order-dependent action control
must pass their unchanged gates; one supporting axis may be unmeasured and
broad activity remains descriptive. Only a complete public PASS authorizes the
governed C transfer audit. Only a later combined C PASS can authorize the
already bounded local LTX path and exactly 3,600 accepted credited synthetic
seconds for the same three-seed descriptive learner run.

### Complete learner-effective source result — frozen no-go

The corrected source run completed all thirteen independent public families on
four CPU cores in `00:03:06`, with zero direct monetary cost. Its external
aggregate record is sealed at `5f4aeff2…13b37`. The official VISOR-HOS source,
semantic reference, contact and explicit-no-contact strata, no-hand nominee
queue, integrity checks, COCO composites, language/lexical fixtures,
referent/attribute composites, recurrence pairs, deterministic sensor clips,
and cross-partition independence all passed their source checks. The no-hand
items remained nominees only: applicant visual review and model inference were
not opened. The active self-authored attributive German audio seed was also
sealed beforehand with 112 files at `3379b1cc…21a3e`.

The single failing family was the genuinely order-dependent first-person
Charades-Ego control. After reconstructing the prior 96-item broad-context
exclusion set from its exact frozen public rule (reproducing 48/48 items,
24/19 subjects, and minimum label counts 9/8), the action sampler retained 44
development and 44 holdout intervals rather than 48 each. Development had 5
`turn_on` and 3 `turn_off` items against 6 required; holdout had 2 `turn_off`
items against 6 required. Subject, video, and object overlap counts remained
zero.

This is `NO_GO_COMPLETE_SOURCE_FEASIBILITY`. It is a combined source decision,
not an early stop on the first independent check, and it does not reinterpret
any earlier no-go. The frozen recipe prohibits adding a source, changing a
direction pair, relaxing six-per-direction yield, or replacing the action
control after observing this result. Consequently public model development and
holdout, governed C, LTX preflight or generation, and the Synthetic-1h learner
were not run. No downstream operation is authorized under this frozen route.

### Prospective ambitious route and MiniMax H3 gate — historical, unexecuted

The complete source no-go above remains final. A 2026-08-03 prospective
amendment (`d907d247…e2855d`) now preserves its 44/44 action inventory as a
nonblocking supporting diagnostic rather than claiming that the prior
48-per-partition requirement passed. The then-prospective combined gate was all five
critical learner-effective axes plus at least six of seven axes, with integrity
and privacy checks blocking. Broad activity and global visual similarity are
descriptive. The action-direction diagnostic cannot rescue any failed axis or
contribute an extra axis.

The then-prospective end-to-end goal was deliberately ambitious: finish public
development and sealed holdout qualification, the blind governed C transfer
audit, disclosure-safe C target measurement, feature-matched episode planning,
exactly 3,600 accepted synthetic seconds, and the same three matched learner
seeds. Generated attempts and accepted media must each be audited against the
frozen learner-effective tolerances. The Real-1h formal failure remains
unchanged, so any final result is descriptive only.

MiniMax H3 Base Ref2VA prospectively replaces LTX as the intended local
generator. MiniMax did not win the bakeoff—Gemini scored 27/28 and MiniMax
25/28—and the local pruned/quantized route has no inherited performance claim.
Native ComfyUI 0.30.0 is technically applicable as a governed headless runner:
its H3 graph is local and serializable, exposes fixed-seed sampling, and
supports image/audio references. The frozen plan prohibits Comfy Cloud,
manager/partner nodes, hosted H3 Context-IR and 2K regeneration, and all
network access after cache preparation. Modular German TTS is supplied as a
reference, H3 native audio is discarded, and the exact TTS input is remuxed
before the repaired shared adapter runs.

Generation is nevertheless blocked by the official MiniMax H3 Community
License. It excludes U.S. use of the open weights and separately prohibits
using H3 outputs to improve another AI model. This study is U.S.-based and
would train EgoBabyVLM on H3 output. Written MiniMax permission must expressly
cover both local U.S. academic execution and output-as-training-data use, with
any required UTD acceptance, before weights are downloaded, a ComfyUI
preflight runs, or synthetic media is generated. Public learner-effective
extractor qualification is independent of that license, but it unlocks only
after the canonical runner implements and tests the new nonblocking diagnostic
role without overwriting the prior source no-go.

### Construct-aligned LTX resume amendment

The MiniMax-H3 amendment above remains an immutable prospective history item;
no H3 weight download, inference, or generator outcome occurred. Before any
new public extractor, C, generator, or synthetic-learner outcome, the user
prospectively restored locally governed LTX-2.3 as the fallback generator. The
active amendment is sealed at `842d5a16…81a39` and preserves the H3 amendment
at `d907d247…e2855d`, the complete-source no-go at `5f4aeff2…13b37`, every
earlier calibration no-go, and the formal Real-1h failure.

The scientific hierarchy is unchanged: all five critical learner-effective
axes must pass and at least six of seven axes must validate. The exact existing
44-development and 44-holdout action fixtures are retained without source or
label substitution as a supporting diagnostic only. Development uses the
already frozen action grid. If no grid point meets the unchanged action floors,
the deterministic maximum-macro-F1 and then higher-margin fallback is sealed
solely so the same failed diagnostic can be opened once on holdout. Validly
measured action performance cannot block or rescue the combined learner-effective
decision. Decode, inference, serialization, provenance, privacy, nonfinite,
silent-truncation, or external-call failures remain blocking integrity failures.

The canonical schema-17 runner now implements exact reuse of the prior passing
source families and exact 44/44 action rows, preserves failed diagnostic points
as ineligible, accepts the diagnostic status only for the action module, and
keeps integrity failures blocking. Focused runner and privacy/provenance guard
tests passed before any new model outcome. The next gate is the blind applicant
review and lineage-bound seal of the public no-hand nominees; fixture
preparation and public inference remain fail-closed until it exists.

The first public-only no-hand preparation attempt then stopped as an
engineering failure before any model inference: one of 13 cached public frame
archives contained 425,725 bytes against the server-declared 285,776,823 bytes
and was not a complete ZIP. Twelve completed archives remain valid. The
canonical downloader now checks declared response length, retries silent
truncation, and fails closed at the existing retry ceiling. This repair changes
no fixture, threshold, label, split, or scientific gate; the bounded
preparation retry is authorized from the retained cache. Because the frozen
queue references 124 unique public archives, the sequential retry was stopped
before inference with 20 complete archives and one unpromoted partial retained.
The canonical preparation now uses four bounded download workers under the
already frozen four-CPU, two-hour topology; archive order in the sealed manifest
remains deterministic.

The cached preparation then completed on the fourth engineering attempt. The
sealed review bundle contains 384 public frames in 48 contact sheets, with zero
decode failures, no restricted mount, and no model inference. Aggregate wall
time across all attempts was 6,924 seconds and retained public storage was
13.246703 GiB, within the frozen two-hour and 200-GiB ceilings. The source-frame
materialization is committed at `d34f3105…e1b33` and the blind review queue at
`5ba8ae3e…f2b7`. This is readiness for applicant review, not public extractor
qualification. The authorized applicant must now code a contiguous prefix to
48 verified no-hand items in each partition, blind to EgoHOS output, and attest
their role, blindness, and that inference has not started.

While that review was still unsealed, the applicant prospectively corrected
the binary label semantics. At correction time 195 items had been coded: 193
yes/no values were swapped and two abstentions were unchanged. Development
changed from 6 yes, 184 no, and 2 abstain to 184 yes, 6 no, and 2 abstain;
holdout changed from 0/3/0 to 3/0/0. The label-record commitment changed from
`7dd640e1…e4fb5` to `2ffae4d1…eae8`, while the fixed review queue remains
`5ba8ae3e…f2b7`. This user-directed pre-seal correction changed no queue order,
fixture lineage, threshold, model, or scientific rule. Review remains in
progress; no seal, inference, or public scientific outcome exists.

The completed corrected review then sealed `PASS` in job 315425 with exit
`0:0`, empty stderr, and 13 seconds elapsed on four CPUs and 16 GiB at zero
direct monetary cost. Across two partitions, 251 items were coded; the seal
retained 96 verified no-hand items and compactly reports 15 visible-hand codes,
2 abstentions, 133 unreviewed items, and zero deficient partitions. Review
labels are committed at `723f218b…1edd` and the verified no-hand seal at
`a58ca3f1…1c97`. The authorized-applicant, blind-to-EgoHOS, and no-prior-EgoHOS-
inference attestations are all true. This authorizes exact public fixture
preparation. Model inference remains conditional on the fixture manifest and
lineage seal; no fixture, model, development, holdout, or later scientific
outcome has opened.

Two subsequent fixture-preparation attempts stopped as engineering failures
before inference. Job 315430 exited `1:0` after 5 seconds because the current
public root lacked the earlier sealed runtime manifest; the job-313924 and
job-314974 roots were distinct. The old runtime, base, and runtime-amendment
commitments exactly matched the config, so only `mechanistic-tuples`,
`activity-code`, and `activity-pydeps` were checksum-copied: 23,438,221,791
bytes (21.828545 GiB), leaving 535.877 GiB free within the 200-GiB ceiling.
The copy used no Git, network, restricted data, inference, or money. Job 315445
then exited `1:0` after 81 seconds with `E_TUPLE_FIXTURE_VIDEO_ENCODE` before a
fixture manifest, model, development, or holdout outcome. The final mux
temporary path ended in `.mp4.partial`; the canonical fix uses `.partial.mp4`.
The audit passed: frames, audio, timing, fps, codec, quality, duration, assets,
selections, thresholds, and final target are unchanged, and all fixture,
development, threshold, and holdout outputs remain absent. The current runner
SHA-256 is `33cd04d5…e272`; deterministic fixture retry is safe and authorized,
while model inference remains blocked until the fixture seal.

Job 315452 nevertheless reproduced `E_TUPLE_FIXTURE_VIDEO_ENCODE` after 11
seconds with exit 1, still before a fixture manifest or inference and despite
the suffix fix. The bounded public/dummy pinned-binary diagnostic chain found
that job 315453 rejected `anullsrc` option `d`, job 315455 accepted `atrim` for
silence but rejected `adelay` option `all`, and job 315457 stopped only because
Python `aifc` could not read one compression type during inventory before
encode. All 112 AIFF seeds are mono at 22,050 Hz. Job 315458 then passed the
pinned-binary smoke using
`anullsrc=r=22050:cl=mono,atrim=duration=<d>` and mono `adelay=2500`, with no
`d=` or `all=` options. Silent and speech muxes both exited 0, decoded as mono
22,050-Hz audio for 7.012426 seconds with 63 frames at 9 fps, and silence had no
active samples. Speech activity moved from source samples 14–21035 to output
samples 55198–76134; start/end errors were +59/-26 samples, both within one
1,024-sample AAC frame (about 46.4 ms). The audit passed and no further
diagnostic is required. Current runner SHA-256 is `1897c40b…6768`; the full
deterministic fixture retry is authorized while fixture, development,
threshold, holdout, and model outputs remain absent.

The subsequent geometry-compatible fixture retry passed in job 315501 and
sealed 824 public items/pairs at commitment `2758557f…03dae6`, with the blind
no-hand seal `a58ca3f1…1c97` preserved and no cross-partition source overlap.
After a pending H100 submission was canceled before allocation or inference,
the qualification topology was prospectively frozen to one A30. Development
job 315542 then ran all seven modules but sealed
`NO_GO_DEVELOPMENT_COMBINED_GATE`: only recurrence and the deterministic sensor
module completed, while five modules errored. This is final for that route at
commitment `4b7cd583…ec66bf`; its holdout, governed C, LTX, and learner stages
were not run.

A new user-authorized engineering-health amendment is frozen at
`d447a7e1…2c6205` without reinterpreting job 315542. It requires one
metric-withholding 28-case production-path microqualification before any new
scientific metric. The same public stack, fixtures, runtime, schemas, and
serializers must run on one A30 under a three-submission, 15-minute-each,
0.75-GPU-hour, 10-GiB, $0 ceiling. Ordinary implementation failures may be
repaired within that budget, but models, sources, labels, partitions,
thresholds, and scientific rules cannot change. Only a committed all-module
health PASS may open new-route development; any execution fault withholds all
partial scientific metrics and is an engineering blocker rather than a
scientific no-go.

Before that health run produced any output, A30 job 316158 remained pending
with zero elapsed time and was canceled before allocation. User-authorized
resource redirect `f7fc16f5…86db6d` now freezes the unchanged health run to
one `nvidia_h100_nvl_3g.47gb` MIG slice on partition `h100`, one process,
eight CPUs, 32 GiB, no DDP, and 15 minutes. The three-submission maximum,
0.75 aggregate slice-GPU-hours, 10-GiB storage, and $0 limits are unchanged.
This is scheduler-only: the A30 provenance and all scientific contracts remain
preserved, and no health or scientific outcome was used to make the redirect.

The bounded H100 route is now terminal. Attempt 1 (job 316325) stopped before
inference on the missing public language cache; the exact cache was restored
and sealed without scientific outcomes. Attempt 2 (job 316353) stopped before
container entry on a redundant wrapper topology probe; its prospective repair
preserved the exact topology and science. Final attempt 3 (job 316370) completed
after 20 seconds but returned `ENGINEERING_BLOCKER`, with 0/7 modules complete
and zero scientific metrics. Aggregate-only trace diagnosis found that the
runner tried to invoke `scontrol` inside the network-disabled Singularity
runtime, where it was unavailable, despite the outer wrapper already passing
the same authoritative scheduler predicates. Six other module traces were
declared preflight-blocked. Terminal commitment `644028ba…9d0f81` classifies
this as an exhausted engineering blocker—not a scientific no-go—and freezes
the protocol-accounted 0.504019 H100-slice GPU-hours and $0 direct cost. No
fourth health attempt, public development/holdout, governed C, LTX work,
generation, or synthetic learner is authorized under this route.

The user then explicitly authorized one distinct prospective repair route while
preserving that blocker unchanged. Amendment `3271499c…a7ff4` permits only
attempt 4 on the same H100 NVL `3g.47gb` slice, one process, eight CPUs, 32 GiB,
15 minutes, 0.25 additional slice-GPU-hours, 1 GiB new run storage, and $0.
The wrapper retains all seven exact Slurm assertions and writes a compact
mode-0600 topology attestation after they pass; the container validates its
exact job/topology binding and CUDA properties without calling `scontrol`.
Its hash is folded into the health dependency commitment. No model, fixture,
metric, threshold, seed, scientific gate, or downstream rule changes, and no
attempt 5 is permitted. Public development remains conditional on a committed
all-module attempt-4 pass.

That sole submission was H100 job 316478 and it sealed blocker
`59b1778b…0edde3`. All seven outer-wrapper scheduler predicates passed and the
compact topology attestation was written, but the prospectively committed CLI
parser still bounded `--attempt` to 1, 2, or 3. It rejected attempt 4 before
runner entry, dependency or fixture preflight, model loading, or module
inference. The 15-second job produced no full health result, private trace, or
scientific metric. Conservative accounting charges the full 0.25 authorized
slice-GPU-hour, bringing the protocol total to 0.754019 slice-GPU-hours and $0.
This is an engineering blocker, not a scientific no-go; the route is exhausted
with no attempt 5 and all downstream stages remain unauthorized.

The user subsequently authorized a distinct one-submission parser-bound repair
at commitment `d9cf3fea…48c8b`. It adds global attempt 5 to the CLI and moves
the wrapper to attempt 5 while keeping attempt 4 invalid and its blocker final.
No model, weight, fixture, preprocessing, threshold, metric, seed, module path,
or scientific gate changes. The topology remains one H100 NVL `3g.47gb`
slice, eight CPUs, 32 GiB, and 15 minutes, with at most 0.25 additional
slice-GPU-hours, 1 GiB, and $0. The full 28-case suite must restart; only a
committed 7/7 pass may open development, and no repair or attempt 6 follows a
failure.

Sole post-blocker job 316537 completed at the scheduler in 33 seconds and wrote
a valid committed `ENGINEERING_BLOCKER` record, `b05dc8da…deb9c`. The complete
suite stopped in production-dependency preflight: base-container immutable-file
verification returned `E_TUPLE_HEALTH_ARTIFACT_COMMITMENT` before model loading.
No module completed; six dependent modules were explicitly preflight-blocked;
there were zero unaccounted failures, external calls, invalid retained records,
or scientific metrics. The public container was present and matched the frozen
hash in a post-run host check, leaving the execution-time discrepancy unresolved
within this route. Attempt 6 and all downstream stages are unauthorized.

After that seal, the user explicitly authorized all outcome-independent
engineering attempts needed to reach a complete valid public scientific result.
Read-only namespace diagnosis showed that the public SIF entry is a symlink:
the resolved 3,731,320,832-byte target is present and byte-matches the frozen
hash on the host, but the target is absent inside the running SIF namespace.
Schema 27 freezes the minimal attempt-6 repair before any new outcome. The
wrapper verifies the symlink, resolved regular target, exact bytes, and SHA-256
before container launch and writes a mode-0600 canonical attestation; the runner
validates its job, run mode, attempt, predicates, hash, and bytes without trying
to dereference the host-only target. The same attestation path is used for later
scientific development/holdout so stable dependency commitments remain equal.
No scientific input or rule changes.

Attempt 6, Juno job 316604, then timed out at 15:24. Both private attestations
were present within eight seconds, but no root-level microfixture projection,
module inference, full result, trace, terminal output, or scientific metric was
written. This is sealed as engineering timeout `e559cd53…6d114`, not a
scientific no-go. Before attempt 7, schema 28 adds a private mode-0600 atomic
progress record containing only stable stage and module-ordinal fields. The
full 28-case, two-replicate production-path suite and every scientific input,
threshold, and decision rule remain unchanged.

Only a combined public pass may authorize the governed C transfer audit and
measurement. Only a combined C pass may authorize a public-word episode-plan
commitment and LTX final-topology preflight. The generator remains pinned to
`Lightricks/LTX-2@9377758131b1ffde4b7f766804590a6617bf2ab9`,
`Lightricks/LTX-2.3@4229404625088d21c4f112eb640fb04a0900ee25`, and
`google/gemma-3-12b-it-qat-q4_0-unquantized@68f7ee4fbd59087436ada77ed2d62f373fdd4482`.
The completed bakeoff remains Gemini 27/28, MiniMax 25/28, Seedance 24/28,
and LTX 19/28; choosing LTX locally is not a quality-ranking revision. No more
than 3,600 accepted credited synthetic seconds and the same three 4,668-step
learner seeds are permitted, with descriptive exploratory reporting only.

### LTX sole-generator and structured prompt-compiler amendment

Before the no-hand review or any new public model, governed C, generator, or
synthetic learner outcome, the user clarified that LTX-2.3 is the sole selected
generator for this pilot—not a fallback or one candidate among several. The
amendment is sealed at `cb4a7cd2…19c62`. MiniMax H3 remains preserved as
history but is out of scope; no further H3 research, download, inference, or
substitution is permitted. The completed bakeoff ranking remains unchanged.

If and only if the combined public and governed-C gates pass, one deterministic
episode-plan-to-prompt compiler must be implemented and validated on public or
self-authored dummy material before material generation. Its fixed schema and
template explicitly encode child-height first-person camera motion/framing;
room, light, clutter, distractors, occlusion, and single-shot continuity;
public noun and visible adjective contrasts; left/right hands, contact, action
phases, completion, and object persistence; exact approach, naming,
manipulation, recurrence, idle/transition, and exit beats; referent dominance,
ambiguity, and null cases; cross-episode recurrence/burstiness; modular German
TTS timing; and fixed negative constraints against third-person views, cuts,
drift, impossible physics, floating objects, extra anatomy, text, captions,
logos, and watermarks.

The compiler uses canonical JSON, a fixed template, seed master `314159`,
121 frames at 24 fps (`5.041666666666667` seconds), and no dynamic prompt
enhancement. Prompt text may use only permitted public words and the sealed
disclosure-safe C bins. Hand-authored or attempt-specific prompt improvement is
prohibited. Public/dummy preflight must seal source/schema/template hashes,
model and text-encoder provenance and licenses, decoding, camera/action
controls, TTS timing slots, retry reasons, acceptance rules, resource ceilings,
and prompt commitments before the one-hour run. Engineering corrections are
allowed only prospectively on public/dummy failures; evaluation material, C
text or vocabulary, learner scores, and generated-corpus learner outcomes may
never tune prompts, retries, or acceptance.
