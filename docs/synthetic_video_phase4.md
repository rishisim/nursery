# Synthetic-video Phase 4 common evaluation assets

**Status:** **CORRECTED COMMON ASSETS PASS** — Stage A remains PASS; the first
Stage B assets remain provisional/superseded, and the corrected lexical and
temporal assets are sealed. The active exploratory mechanistic tuple-calibration
artifact gate passed and local-reload sizing plus public qualification remain;
this is not confirmatory Phase 5.

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
calibration amendment at commitment `c9a48206…adaf`. Only its public
qualification is presently authorized. Its public artifact manifest passed at
commitment `8c787a01…b527` with seven components, four immutable repository
archives, and ten weight/resource files (14.62 GB), without model inference or
a restricted mount. The EgoHOS bundle is distributed by the official
MIT-licensed project but contains no separate archive license file; this record
asserts academic local prototype use only and no broader license conclusion.
The 53-package runtime overlay then passed resource-only preparation at
commitment `968f2570…1f10`, with neither a restricted mount nor model
inference. Local-files-only model reload and blind sizing remain pending.
Governed C measurement requires that pass, and LTX generation remains further conditional and capped at exactly one
accepted credited synthetic hour. No confirmatory/equivalence claim is
authorized.

The task-matched fixture implementation is separately frozen before model
outcomes at commitment `506a1f41…251d`. Its action control uses fixed
open/close, take/put, sit-down/stand-up, and turn-on/turn-off prompts and
localized Charades-Ego intervals disjoint from the failed broad-context study.
Exact public fixture manifests are still pending and must be sealed before
development inference.
