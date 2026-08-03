# Synthetic-video Phase 4 common evaluation assets

**Status:** **CORRECTED COMMON ASSETS PASS; PROSPECTIVE VISOR-HOS FIXTURE
CORRECTION FROZEN** — Stage A remains PASS; the first Stage B assets remain
provisional/superseded, and the corrected lexical and temporal assets are
sealed. The prior mechanistic tuple-calibration fixture recipe fired its frozen
public source no-go and remains final. A new user-authorized correction is
frozen before new public or restricted outcomes; this is not confirmatory
Phase 5 and no synthetic arm has run.

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
