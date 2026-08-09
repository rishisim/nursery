# Gemma 4 E4B opaque-instrument activation rationale v1

Date: 2026-07-22  
Frozen amendment: `frozen_gemma4_e4b_opaque_instrument_provenance_amendment_v1.json`  
Amendment SHA-256: `93874c69153daaf41d902216351666c59b12aeb656b9739ecd07d1e6e4fe42b1`

## Result

The exact already-cached Gemma 4 E4B 4-bit artifact passes a newly frozen **opaque, internal-only prototype provenance gate**. It is not yet fully activated for restricted inference. The exact common-schema public canary, runtime/adapter hashes, quarantine checks, and every original scientific gate remain mandatory.

This is an additive new branch. It does not overwrite or declare erroneous either prior terminal result:

- `CALIBRATION_STOP_MODEL_TRIANGULATION` remains immutable at terminal-decision SHA-256 `a9f955d53768e0295d61a58e8ecf73d7e8658890ec56a4408772cd6bb10612b6` and disposition SHA-256 `831a7d96d598f45e81514fb1cec5a0c4eaf2f84d6726ae4eb86dcd161e932be6`.
- `CHILDLENS_LEXICAL_FEASIBILITY_STOP_CURRENT_RESOURCES` remains terminal. No model-only branch supplies the missing qualified human German evidence.

The original STOP correctly applied the original requirement for unambiguous, reconstructible conversion provenance. This amendment does not repair that provenance. It prospectively adopts a narrower admissibility standard for one fixed instrument in one internal prototype: exact cached bytes may be treated as an opaque measurement instrument when the byte manifest, official-source rationale, local confinement, scientific limits, and non-redistribution covenant are all frozen.

## Exact artifact binding

Only this artifact is covered:

| Field | Frozen value |
|---|---|
| Repository label | `mlx-community/gemma-4-e4b-it-4bit` |
| Conversion revision | `475b9088d29754a3379866cf5aeb6b41acd313c2` |
| Files | 10 |
| Bytes | 5,179,241,512 |
| Complete tree-manifest SHA-256 | `34310498dc5b809bf4baa79a5290c9e675022d12b725993317bccbffacf1d3ae` |
| Converted weight bytes | 5,146,800,534 |
| Converted weight SHA-256 | `932b8271fc3fe65adcc78b96c10c6268bbfb13e8f67d1358727c0d6ee97e1eff` |
| Configured quantization | affine, 4-bit, group size 64 |

The cached snapshot was rehashed and exactly matched the earlier public resource receipt. Every individual file hash is frozen in the amendment. A changed byte, added/removed file, other revision, adapter, tokenizer, processor, template, or quantization is a different instrument and fails closed.

No new model was downloaded. No click-through, terms, authentication token, or license acceptance was performed.

## Official-source rationale and its limit

The conversion repository names `google/gemma-4-E4B-it` as its base model. Public Hugging Face metadata for the declared source revision `fee6332c1abaafb77f6f9624236c63aa2f1d0187` and current pinned official revision `ee0ef6023621cff504d758262d4e04895a5af4a2` reports both as public, ungated, and Apache-2.0. Both revisions expose the same 15,992,595,884-byte source weight with SHA-256 `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`. Google also publishes the Gemma 4 Apache-2.0 license on its official developer site.

That is a bounded research-governance rationale, not legal advice. It does not cryptographically prove that the cached conversion was produced from those bytes, identify the converter version, reconstruct the exact conversion, resolve the conversion card's `license: gemma` metadata, or grant redistribution rights. The project explicitly keeps those facts unresolved and visible.

If institutional policy or qualified legal review requires exact conversion lineage or treats the metadata conflict differently, this exception immediately fails closed.

## Why an opaque instrument can be scientifically bounded

For this descriptive prototype, reproducible identity of the instrument's **input/output behavior** can be established by fixing all artifact bytes, runtime/adapter hashes, prompt, parser, schema, sample, and execution controls. The instrument is not a learner ancestor, evaluation oracle, or source of ground truth. Its outputs are hypotheses used only to widen model-specific uncertainty ranges.

This narrower behavior-level reproducibility is sufficient for an internal prototype sensitivity instrument only because:

1. the artifact is already cached and fully hash-bound;
2. it never leaves the owner-controlled host;
3. all inference is network-denied and outputs remain in quarantine;
4. no weights, tokenizer, embeddings, features, vocabulary, scores, or encodings enter learner ancestry;
5. no model agreement is interpreted as correctness;
6. simulator-oracle labels remain the only later causal-evaluation truth;
7. every report must disclose the unresolved conversion provenance.

It is not sufficient for redistribution, a reusable released checkpoint, publication-grade annotation truth, or a claim that another researcher can regenerate identical weights from the upstream model.

## Permitted and prohibited use

Permitted after a separate full activation receipt:

- one local, network-denied Gemma-versus-Qwen3 prototype calibration pass over the unchanged frozen sample and windows;
- quarantine-only pseudo-label hypotheses under the exact five-field schema;
- K=5-protected, outward-rounded nonidentifying aggregate uncertainty ranges;
- simulator sensitivity analysis with simulator-oracle evaluation truth.

Prohibited:

- redistributing, publishing, uploading, mirroring, emailing, sharing, or externally processing any cached artifact file;
- representing the artifact as an official Google conversion or as reproducible from an identified converter;
- claiming that this is legal advice or institutional legal clearance;
- changing or replacing any artifact byte under the exception;
- using model artifacts or outputs in learner/checkpoint/tokenizer ancestry or as primary evaluation truth;
- changing sample selection, prompts, schemas, thresholds, privacy controls, K=5 suppression, rounding, or causal claims;
- running the learner or causal outcome under this provenance amendment.

## Activation state

The provenance branch is `OPAQUE_INTERNAL_ONLY_PASS`. The instrument remains `NOT_FULLY_ACTIVE` because the amendment intentionally leaves all non-provenance gates unchanged.

The next task is limited to freezing the runtime and adapter hashes and running the exact frozen common-schema public joint audio-plus-five-frame canary under OS network denial. Only a separate full activation receipt may then authorize restricted Gemma inference. The canary itself cannot authorize restricted use.

Any failed hash, canary, network, quarantine, schema, coverage, abstention, K=5, or scientific gate returns this branch to the immutable `CALIBRATION_STOP_MODEL_TRIANGULATION` state.

## Privacy and scientific boundary

No restricted content was accessed to draft or freeze this amendment. No transcript, frame, audio, identifier, timestamp, filename, prediction, or small cell appears here. No hosted model or external inference was used. AEA and BabyView empirical ancestry remain absent. The prior lexical STOP, lack of human validation, no-ground-truth boundary, and simulator-oracle evaluation boundary are unchanged.

Primary public evidence: [Google's Apache-2.0 Gemma license](https://ai.google.dev/gemma/apache_2), [declared official source revision](https://huggingface.co/google/gemma-4-E4B-it/tree/fee6332c1abaafb77f6f9624236c63aa2f1d0187), [current pinned official revision](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2), and [exact conversion revision](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit/tree/475b9088d29754a3379866cf5aeb6b41acd313c2).

