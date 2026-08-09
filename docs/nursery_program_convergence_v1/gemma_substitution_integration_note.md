# Gemma substitution integration note

Status: `DESIGN_ONLY_LICENSE_PROVENANCE_BLOCKED`

This note specifies the smallest safe integration for a Gemma visual-referential
substitution calibration. It was derived only from public code, frozen protocols,
and aggregate-safe receipts. It does not authorize or perform restricted access,
model acquisition, model execution, or a scientific outcome.

## Historical state that must remain immutable

The completed ChildLens pseudo-calibration remains the historical record:

- public receipt status: `CALIBRATION_REVISE`
- frozen primary sample: 15 items, 900 seconds, and 137 referential windows
- public receipt schema: `nursery-childlens-pseudo-calibration-receipt-v1`
- frozen protocol SHA-256:
  `db5336726f4841e592cb9c73df5ab22a0887d1bd34c7cf673146f38de9e86961`
- activation receipt SHA-256:
  `c926207bbb9fd02b7cc23316cd17bd1c87298e5f64593ef7ed0a80c32271edf5`
- completed public receipt SHA-256:
  `c8da2349663f7e49291381c42e15f8972bb3b13b0b739dad4e1221ee937dfc59`
- completed Qwen3 worker SHA-256:
  `e785ea475aee62fd72ab9bc064f3c9c3dcbc189b4361e996963edf9f2ab14f88`
- completed coordinator SHA-256:
  `810ba0caf4bf5cfbb94917ecd4747c328633d21a8fb5733be28e6fc6bc5faaed`

The current frozen protocol and activation receipt mark
`gemma4_e4b_it_4bit` inactive because its exact conversion did not pass the
license/provenance gate. A public synthetic bakeoff demonstrating joint audio
and five-frame inference does not resolve that gate and does not demonstrate the
frozen referential common schema. Restricted Gemma execution therefore remains
fail-closed until a separately frozen substitution amendment establishes all of
the activation evidence below.

The existing receipt must not be overwritten. A substitution run, if later
authorized, writes a new receipt:

`output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json`

## Activation preconditions

Before any restricted input is opened, a substitution activation seal must
bind all of the following:

1. A separately frozen substitution amendment that permits the substitution
   without changing the sample, windows, thresholds, or scientific use.
2. Unambiguous license permission and provenance for the exact upstream model,
   exact conversion, and every required local runtime artifact.
3. An exact artifact manifest containing immutable revisions and SHA-256 hashes.
4. A public synthetic canary that exercises joint raw audio plus the fixed five
   frames and produces the exact frozen referential common schema.
5. Fail-closed network denial, disabled telemetry, owner-private temporary
   storage, and descriptor-only restricted input handling.
6. The new worker and coordinator hashes, prompt/schema digest, audio rule, and
   frame offsets.

Any absent, ambiguous, mutable, or mismatched field blocks the run. Technical
availability, a previously cached model, or a successful unrelated Gemma
example is not permission evidence.

## Smallest code change

Historical code and outputs should be preserved. Add two new scripts and one
new test module, with only one backward-compatible change to the completed
coordinator:

| File | Change |
|---|---|
| `scripts/nursery_gemma4_referential_worker.py` | New descriptor-only, network-denied Gemma worker for the fixed joint audio/five-frame referential job. Keeping it separate prevents a new worker hash from being confused with the completed Qwen3 binding. |
| `scripts/nursery_gemma_substitution_calibration.py` | New thin coordinator that imports content-agnostic helpers from `nursery_pseudo_calibration.py`, loads the completed Qwen3 outputs read-only, runs only Gemma, and writes a distinct public receipt. |
| `scripts/nursery_pseudo_calibration.py::_calibration_ranges` | Add optional left-path and right-path labels. Defaults must remain `existing_whisper_qwen2_path` and `challenger_qwen3asr_qwen3vl_path`, preserving current behavior and tests exactly. |
| `tests/test_nursery_gemma_substitution_calibration.py` | New fail-closed binding, checkpoint, privacy, schema, and immutability tests. |

No other completed function needs semantic modification. In particular,
`_restricted_inputs`, `_speech_diagnostics`, `_usable_gate`, and
`_validate_public_receipt` can be reused without weakening their invariants.
Duplicating `_calibration_ranges` in the new coordinator is acceptable if even
the backward-compatible signature change is judged too risky, but it creates a
larger maintenance surface.

The new coordinator should provide these narrow functions:

- `_public_gemma_seal()` — validate the separately frozen amendment, exact
  license/provenance record, artifact manifest, worker hash, prompt/schema
  digest, and strict-schema public canary.
- `_load_qwen3_outputs_read_only(root, sample_digest)` — validate and return the
  completed Qwen3 ASR and Qwen3-VL documents without calling
  `_challenger_outputs` and without permitting regeneration.
- `_invoke_gemma_worker(...)` — create the fixed descriptor-backed job and
  execute the independently sealed worker.
- `_gemma_checkpoint_output(...)` — validate, resume, compact, and seal the
  fixed 137-window output.
- `restricted_execute_gemma()` — call the existing restricted-input loader,
  the read-only Qwen3 loader, the Gemma worker, `_calibration_ranges`,
  `_speech_diagnostics`, `_usable_gate`, and the unchanged public validator.

For the substitution receipt, pass these model-path labels to
`_calibration_ranges`:

- `qwen3asr_plus_qwen3vl_path`
- `qwen3asr_plus_gemma4_path`

Retain `model_intersection`, `model_union`, and `conservative_envelope`. The
top-level public schema and aggregate structures remain unchanged; only the two
instrument-path labels differ.

## Read-only Qwen3 reuse

The substitution compares the new Gemma referential hypotheses with the
already completed Qwen3-VL hypotheses while reusing the completed Qwen3 ASR
diagnostics. It must never rerun or rewrite Qwen3.

`_load_qwen3_outputs_read_only` must fail closed unless all of these hold:

- owner-only regular files, no symlinks, in the expected private namespace;
- the completed execution-binding schema and exact private sample digest match;
- the original Qwen3 worker hash and exact ASR, aligner, and VLM revisions match
  the values pinned by the substitution amendment;
- bound Qwen3 ASR and VLM output hashes match the files byte-for-byte;
- both output documents satisfy their completed schemas;
- exactly 15 primary items are present, with no duplicate opaque keys;
- the Qwen3-VL rows match the exact private ordered window crosswalk and contain
  exactly 137 windows; and
- no completed file is opened for writing.

The new Gemma execution binding must include SHA-256 hashes of the original
Qwen3 execution binding, Qwen3 ASR output, and Qwen3-VL output. A mismatch is a
hard error, not a trigger to rerun Qwen3. Tests should monkeypatch
`_challenger_outputs` to raise and prove that the substitution path never calls
it.

## Gemma checkpoint and binding rules

Use a new owner-private quarantine namespace such as
`provisional_calibration_v1/gemma_substitution_v1`. Its binding should contain:

- binding schema version;
- substitution amendment and activation-seal SHA-256 hashes;
- private primary-sample digest and private ordered-window-set digest;
- fixed `window_count: 137`;
- hashes of the completed Qwen3 binding, ASR output, and VLM output;
- Gemma worker SHA-256;
- exact upstream model and conversion revisions plus artifact manifest hashes;
- exact license-evidence receipt hash;
- strict common-schema and prompt digest;
- fixed five frame offsets;
- fixed raw-audio clip rule;
- final compacted Gemma output SHA-256;
- `network_disabled: true`; and
- `not_ground_truth: true` and `not_evaluation_truth: true`.

A completed output may be reused only when every field and hash matches. Partial
resume must be prefix-bound rather than content-selected:

1. The worker writes one parsed record per fixed window to an owner-private,
   descriptor-backed checkpoint stream and flushes it durably.
2. On restart, the coordinator validates that the checkpoint is a canonical
   prefix of the frozen ordered crosswalk.
3. Only the missing fixed indices are sent to the worker.
4. No confidence, label, lexical content, apparent success, or failure category
   may affect retry or selection.
5. After all 137 records validate, compact them into a single 15-item document,
   hash it, and seal the binding.

If canonical prefix-resume cannot be implemented safely, a full identical
technical rerun is preferable to selective/content-dependent retry, but it must
still use the same 137 windows and be recorded as a rerun.

## Exact joint audio-frame job

The Gemma worker receives exactly the frozen 15 items, 900 selected seconds,
and 137 ordered referential windows. Each window job has:

- an opaque item key and an already-open media file descriptor; no source path,
  filename, identifier, or restricted value in argv or the environment;
- the exact same per-item window ordering used by the completed Qwen3 crosswalk;
- five frames at offsets `-5.0`, `-2.5`, `0.0`, `+2.5`, and `+5.0` seconds from
  the frozen window center, clipped only to media bounds;
- one mono 16-kHz raw-audio clip from center minus 5 seconds through center plus
  5 seconds, clipped only to media bounds; and
- one response conforming to the exact common schema.

The Gemma referential job should consume raw audio, not the Qwen3 transcript.
Passing the Qwen3 transcript into Gemma would make Qwen3 ASR an ancestor of the
Gemma path and confound the intended instrument comparison. Qwen3 ASR remains a
separate reused speech-diagnostics path.

The only accepted scientific response keys are:

- `candidate_count_bin`
- `visibility_bin`
- `referential_status`
- `lexical_support`
- `lag_event_unit_bin`

Internal `window_index`, bounded start/end coordinates, and `schema_valid` may
exist only in the restricted checkpoint. Prose, extra keys, confidence,
transcript text, language text, source-role text, or identifiers make the row
schema-invalid. Raw responses never enter the public receipt.

Run exactly one MPS-heavy process. Qwen3 must not run concurrently because it
must not run at all. CPU decode may use the already measured safe worker limit,
but each temporary audio/frame bundle must be consumed once and deleted before
advancing.

## Privacy and scientific failure modes

The integration must fail closed against these risks:

- **Incidental transcription or identity text:** a joint audio model may emit
  speech, names, roles, or prose even when not requested. The exact-key parser
  rejects it; raw output remains restricted and is never exported.
- **Path leakage:** media paths, filenames, opaque-key mappings, timestamps, and
  identifiers must not appear in argv, environment variables, logs, exceptions,
  reports, or tool output. Use inherited file descriptors and fixed error codes.
- **Temporary derivative leakage:** audio clips, frames, prompts, raw responses,
  and checkpoints remain owner-private in quarantine and are deleted or retained
  only under the controlling agreement.
- **Network or telemetry:** enforce a network-denied sandbox, offline runtime
  flags, disabled telemetry, read-only model/cache mounts, quarantine-local
  `TMPDIR`, and a live fail-closed sentinel. Redirect worker stdout/stderr to
  `/dev/null`; diagnostic output is fixed-code only.
- **Learner/evaluation ancestry:** Gemma and Qwen3 outputs remain measurement
  hypotheses in quarantine. No weights, features, embeddings, tokenizer,
  vocabulary, confidence, or pseudo-label may enter learner ancestry or primary
  evaluation truth.
- **Small-cell leakage:** the count 137 and full-sample aggregates may be public;
  per-item counts, opaque keys, exact times, examples, and item-level results are
  restricted. Preserve K=5 complementary suppression and outward rounding.
- **Historical overwrite:** refuse any output path equal to the completed public
  receipt or any completed restricted Qwen3 file.
- **Outcome drift:** the substitution produces calibration envelopes only. It
  must not train a learner, alter a tokenizer/checkpoint, run a causal arm, or
  inspect an acquisition outcome.

## Required tests before restricted execution

The new test module should prove:

1. missing or ambiguous substitution activation, license evidence, exact model
   manifest, or strict-schema public canary blocks before restricted access;
2. Qwen3 loading is byte-for-byte read-only and rejects binding, hash, revision,
   item, window-count, order, or private-digest mismatch;
3. the substitution path cannot call `_challenger_outputs` or write Qwen3 files;
4. the frozen ordered set is exactly 137 windows and is shared by both visual
   paths;
5. frame offsets and raw-audio rules are exact and independent of predictions;
6. jobs use descriptors, not restricted paths or values in argv/environment;
7. network and telemetry controls fail closed, including an active sentinel;
8. resume accepts only a canonical fixed-order prefix and never selects by
   model output or confidence;
9. strict parsing rejects prose, extra keys, transcript text, role text,
   confidence, and identifiers;
10. optional `_calibration_ranges` labels preserve all existing default behavior
    and produce the two explicit substitution labels when requested;
11. the distinct substitution receipt passes the unchanged public validator,
    K=5 suppression, outward rounding, privacy, and no-outcome checks; and
12. hashes of the frozen protocol, activation receipt, completed public receipt,
    completed Qwen3 worker, and completed coordinator remain unchanged.

## Decision

The integration is implementable without regenerating Qwen3 and without
changing the aggregate receipt schema. It is not yet executable. The next safe
step is to create and freeze the separate Gemma substitution activation only
after resolving exact conversion/upstream licensing and provenance, then pass a
public synthetic joint-audio/five-frame strict-common-schema canary. Until all
activation preconditions pass, restricted Gemma execution remains blocked.
