# Proposed ChildLens annotation-instrument contract amendment v1.1

Version: `childlens-annotation-instrument-amendment-v1.1.0`  
Status: **PROPOSED — not effective until project authority explicitly approves it and every instrument code/weight license check passes**  
Scope: ChildLens measurement instruments only  
Scientific learner outcome authorized: **no**

## Purpose and non-adoption statement

The existing construction boundary prohibits external pretrained models from being ancestors of scientific artifacts. ChildLens does not currently provide an established timed lexical transcript, speaker-role transcript, or referential annotation in the v1.1 evidence. This proposed amendment defines a narrow measurement-process exception so a fixed pretrained tool may generate quarantined annotation proposals without becoming a learner ancestor.

Creating or citing this document does **not** adopt the exception. Project authority must explicitly approve and version the amendment before any external pretrained instrument is acquired or used on ChildLens. Code and weights require separate license and access-condition receipts. An absent, ambiguous, gated, or user-acceptance-dependent license fails closed. No task agent may accept terms on the user's behalf.

The human-only measurement route needs no pretrained-instrument exception and remains the controlling fallback.

## Controlling rule if approved

For one release-bound, authorized ChildLens pilot, a fixed pretrained instrument may operate only inside an isolated local measurement quarantine to propose restricted intermediate annotations. Every learner-visible or calibration-eligible ChildLens derivative must be grounded in the source media, fully dispositioned by a qualified human under the frozen protocol, and pass the applicable reliability gate. Raw instrument output is never gold.

Instrument weights, checkpoints, prompts, tokenizers, vocabularies, token IDs, hidden states, features, embeddings, logits, probabilities, confidence scores, alternate hypotheses, VAD segments, diarization clusters, detector labels, and unadjudicated text/timing/role/referential output are prohibited from scientific learner inputs, tokenizer-training inputs, model checkpoints, causal-arm payloads, and evaluation payloads.

This is an exception for the measurement process only. It does not create a scientific learner-ancestry exception.

## Two mandatory provenance graphs

If adopted, the project must maintain two distinct directed provenance graphs:

1. **Measurement-process provenance.** Records the fixed instrument, code and weight receipts, license review, input allowlist, execution settings, raw proposal quarantine, human disposition, reliability block, adjudication, and retention/deletion status. An instrument's influence may not be hidden.
2. **Scientific-artifact ancestry.** Begins with the bound ChildLens release and accepted, fully human-dispositioned ChildLens derivatives. It may later include a fresh ChildLens-local tokenizer and scratch-initialized learner. No external instrument artifact or representation has an edge into this graph.

An ASR-assisted transcript may enter the second graph only after a qualified human verifies or corrects the full source-language record against the source audio and explicitly accepts it. The raw hypothesis remains in the first graph's quarantine. A prespecified ASR-hidden human reference subset must quantify anchoring bias.

## Permitted and prohibited instrument roles

| Instrument class | Permitted quarantine role after approval | Required human action | Prohibited transfer |
| --- | --- | --- | --- |
| ASR / speech segmentation | Propose source-language text and speech/utterance boundaries | Verify or correct the entire accepted record; mark uncertain/unusable rather than invent text | Raw/unreviewed words or times; model tokenizer/vocabulary; scores; alternate hypotheses; representations |
| Forced alignment | Propose word/utterance timing after human language routing | Check accepted utterance boundaries and censoring; use manual timing when route unsupported | Aligner vocabulary/features/scores; unchecked exact times |
| VAD / diarization / voice-type classifier | Propose speech regions, anonymous clusters, or role hypotheses | Assign only `NON_CHILD`, `CHILD`, `OVERLAP`, `UNCERTAIN`, or `NONSPEECH`; validate role accuracy | Voiceprints, speaker embeddings, identity inference, cluster IDs, confidences, unreviewed roles |
| Vision detector / tracker / temporal proposal tool | Propose candidate regions or order work only after content-blind pilot selection | Human applies the frozen visible/null/ambiguity ontology and timing rules | Class vocabulary, boxes/tracks/features as learner channels, scores, pseudo-targets, detector-selected items |

No instrument may choose or replace pilot media based on apparent lexical richness, speech quality, visual salience, referent yield, or groundability. No instrument may be fine-tuned on ChildLens during this feasibility task.

## V1.1 instrument-specific disposition

The v1.1 local preflight permits no instrument execution yet:

- FFmpeg `8.0.1_4` is an approved deterministic local decode executable for measurement preprocessing. Its local use does not make it a learner ancestor; its GPL-compiled binary is not redistributed.
- `faster-whisper==1.2.1`, `Systran/faster-whisper-large-v3` revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`, and `whisperx==3.8.5` are fixed candidate specifications but are not installed. Their code/model licenses, hashes, acquisition record, and execution lock still require approval and receipt before ChildLens use.
- VTC remains `NOT_ACQUIRED` and prohibited because explicit code and weight licenses were not established.
- Pyannote remains `NOT_ACQUIRED` and prohibited because acquisition requires user acceptance and information sharing that this task cannot perform. Even if separately authorized later, anonymous clusters would remain proposals requiring human role coding.
- No pretrained vision detector, tracker, temporal ranker, or external label vocabulary is approved.

Public availability is not permission. Project-level ChildLens access does not substitute for instrument code/weight licensing, and an instrument license does not substitute for ChildLens data-use authority.

## Quarantine and one-way release

The measurement workspace must be local, restricted, release-specific, outside `docs/` and `output/`, untracked by Git, excluded from synchronization/indexing/backups as required by the approved security plan, and network-disabled during instrument execution. It holds raw media, decoded audio, raw/corrected text, exact times, frames, identifiers, instrument output, confidences, local clusters, human labels, and adjudication records.

Release from quarantine is one-way and field-allowlisted:

- The report namespace may receive only terms-permitted, nonidentifying, cell-suppressed aggregates, protocol/model identities, and digests that cannot recover restricted values.
- A later learner store may receive only protocol-authorized ChildLens RGB/text records whose text/timing/role has a complete human disposition and whose inclusion is independent of instrument quality or a learner outcome.
- Instrument artifact IDs belong in provenance receipts, never learner example records.
- Exact timing may build internal windows but is never exported in reports.
- Hidden referential targets remain physically separate from the weak learner and all evaluation calls.

A fail-closed export sentinel must reject media, frames, transcript fragments, identifiers, paths, exact timestamps, small cells, raw proposals, model outputs, and unknown fields.

## Human validation requirements

No instrument-assisted derivative is accepted until genuine qualified humans complete the frozen block:

1. Establish actual source language or mixed/undecidable status before alignment.
2. Double-code timing and speaker role until the first of 300 utterances or 30 speech minutes.
3. Achieve speaker-role macro-F1 at least 0.80 under the frozen role definition and report classwise errors plus nonspeech false detections.
4. Measure source-language transcript error against an ASR-hidden adjudicated reference; raw ASR is never the reference.
5. Independently double-code at least 20% of referential items and achieve nominal status alpha or prespecified equivalent at least 0.67.
6. Retain and report null, ambiguity, overlap, uncertainty, undecidability, unusability, coverage, adjudication effort, and participant-cluster uncertainty.

Codex and any model-assisted agent may build the packet, audit assignment, run metric code, and check leakage. They may not be counted as human auditory, language, speaker-role, or referential judges. If qualified humans are unavailable, the gate remains unresolved.

## Minimum immutable receipt

Every approved instrument requires a machine-readable receipt with:

- purpose, class, project/release/pilot digests, quarantine namespace receipt, and approval identity/date;
- package/repository, immutable code revision, exact weight/model revision and SHA-256, model card, acquisition source/date, and dependency/environment lock;
- separate code and weight license reviews, access conditions, reviewer/date, and unresolved clauses;
- exact input allowlist, output inventory, inference settings, prompts if any, language-routing rule, thresholds, seeds, determinism limits, and failure behavior;
- human protocol, qualifications, blinding, double-code assignment, adjudication, reliability, coverage, and anchoring-bias evidence;
- proof that prohibited instrument artifacts are absent from learner/evaluation stores and report artifacts; and
- restricted retention/deletion disposition plus aggregate-export review.

Missing, mutable, ambiguous, or self-asserted receipts fail closed.

## Learner boundary remains unchanged

If a later protocol is authorized, the learner remains a compact temporal CLIP+-style system with a random/scratch vision tower, a corpus-local scratch tokenizer and text tower, and scratch self-supervised/contrastive objectives. No off-the-shelf learner weights are allowed. The tokenizer may be trained only on the authorized ChildLens training partition after complete human disposition. No ASR tokenizer, external vocabulary, detector inventory, public benchmark text, or other child corpus contributes tokens, initialization, labels, or calibration.

Simulator-only IMU, contact/touch, proprioception, and motor streams remain explicitly counterfactual. No annotation instrument may estimate or relabel them as ChildLens-measured.

## Adoption procedure

To activate this amendment, project authority must create a signed/versioned adoption receipt before instrument acquisition or ChildLens processing. The receipt must identify this exact amendment digest, the unchanged one-corpus and no-outcome clauses, every approved instrument receipt, the bound release/pilot receipts, and the retention/export controller. Any unlisted instrument remains prohibited.

Until that occurs, this document is a proposal. It does not authorize model acquisition, license acceptance, ChildLens processing, tokenizer construction, learner training, causal-arm execution, or publication.

