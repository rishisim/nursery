# Synthetic-video Phase 4 common evaluation assets

**Status:** Stage A **PASS**; Stage B pending governed execution

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
first lexicographic criterion: public-set corpus WER 0.1412 versus 0.2824 for
`base`. Its translation chrF was 0.5742, mean word confidence 0.9100,
abstention rate 0.0769, timestamp-valid fraction 1.0, and round-trip fraction
1.0. Local-files-only reload and disabled telemetry/tracking passed. These are
compact qualification aggregates, not ChildLens or scientific outcomes. The
signed compact decision is `results/synthetic_video_language_gate.json`.

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
