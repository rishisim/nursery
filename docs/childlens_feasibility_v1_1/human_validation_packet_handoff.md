# ChildLens v1.1 blinded human-validation packet handoff

Version: `childlens-human-validation-handoff-v1.1.0`  
Handoff date: 2026-07-21  
Packet status: **SKELETON — not ready for human coding**  
Scientific learner outcome authorized: **no**

## Handoff summary

A restricted, blinded 15-item packet skeleton has been created outside the report namespace. Its canonical SHA-256 is:

`ae8503fc5c21fc6df0b08c763202eab074b2526f83ef847d2faca3ae4eb217a5`

The skeleton is bound to the content-blind preselection and contains one opaque assignment row per selected item, null judgment fields, and explicit `NOT_STARTED` status. This report does not reproduce those row keys or any restricted path. The skeleton contains no media, decoded audio, transcript, timestamp, frame, or human label.

No ChildLens video/audio has been acquired, and no human validation has begun. Exact remote bytes remain unresolved and the canonical final restricted manifest is incomplete. Therefore the packet must not yet be distributed to annotators or populated with media. Its current digest is a procedural blinding receipt only.

## Fail-closed blockers before packet population

The coordinator must satisfy all of the following without changing item selection for content reasons:

1. Resolve exact remote byte sizes for every selected object and replace placeholder planning values with a final deterministic restricted manifest.
2. Re-run admission immediately before each selective shard: raw pilot no more than 20 GiB, namespace peak no more than 73 GiB, and projected post-peak free space at least 50 GiB.
3. Verify the untracked quarantine, owner/mode, backup/synchronization/indexing exclusions, no-egress execution, restricted logs, and deletion/retention controller.
4. Acquire only the selected objects and required metadata/annotations; never download the full archive.
5. Hash each admitted object locally, verify its expected immutable identity/size, and stop on mismatch without substituting content.
6. Populate only local blinded windows. Keep media, exact times, row-level metadata, participant/session keys, transcripts, and annotations in quarantine.
7. Freeze and unit-test metric implementations before receiving labels.

Failure of exact-byte or resource admission may use only the frozen 12–18 access/size/empty-stratum adjustment rule before content inspection. It cannot use language, speech density, lexical content, visibility, or apparent grounding quality.

## Staffing and role separation

The actual language is unknown, so final staffing cannot be selected by assumption. Use a two-stage handoff:

1. Two qualified humans independently perform source-language discovery on the content-blind audio allocation and record ISO 639, `MIXED_OR_CODE_SWITCHED`, or `UNDECIDABLE`.
2. Once a route is established, assign language-matched independent coders for transcription/timing/role and qualified independent coders for referential judgments. A qualified adjudicator sees both locked records only after the independent pass.

At minimum, the same person cannot see a peer's label before locking their own. Production-correctors who see optional ASR proposals must be distinct in record and interface from annotators creating the ASR-hidden reference. Annotators remain blind to split assignment, lexical counts, model design, endpoint support, peer labels, and every learner result.

Codex may implement the local UI, verify assignment/digest consistency, compute frozen metrics, and audit privacy. Codex is not a genuine human auditory, language, speaker-role, visual-referential, or adjudication judge and cannot fill these fields.

## Packet layers

### Layer A — language and audio integrity

For every allocated item, independently record:

- established language code, `MIXED_OR_CODE_SWITCHED`, or `UNDECIDABLE`;
- language competence of the coder;
- audio decode/integrity state;
- presence of speech, overlap, unintelligibility, and unusable intervals; and
- no transcript text in reportable logs.

Machine language identification, if later approved, is hidden from the independent pass and is advisory only.

### Layer B — utterance timing, source text, and speaker role

Create restricted utterance rows with an opaque utterance key, accepted onset/offset, source-language verbatim text or explicit unintelligible marker, language route, one of `NON_CHILD`, `CHILD`, `OVERLAP`, `UNCERTAIN`, or `NONSPEECH`, validity flags, coder ID pseudonym, lock status, and adjudication status.

All timing and role validation items are independently double-coded until the first of 300 utterances or 30 speech minutes. At least 20% of the reference transcription subset must be produced without seeing ASR/VAD/role proposals. Translation and guessing are prohibited.

### Layer C — referential status and boundaries

For every accepted `NON_CHILD` utterance, view only its fixed ±5-second local RGB context and independently record:

- `VISIBLE_SINGLE`, `VISIBLE_MULTIPLE`, `NULL_NOT_VISIBLE`, `IRRELEVANT`, `UNDECIDABLE`, or `UNUSABLE`;
- noun/object, verb/action, or neither mention family;
- per-mention visible/null status;
- candidate count band and ambiguity;
- candidate visibility/action boundaries and censoring where applicable; and
- technical unusability/occlusion/uncertainty without inventing a target.

At least 20% of referential items are double-coded using the already frozen release-bound assignment and fixed participant/speech/activity stratification. No detector, VLM, ASR text, lexical frequency, or apparent groundability may select the double-code block.

### Layer D — locked adjudication and effort

Only after both independent records are immutable may a qualified adjudicator resolve coding disagreements. Preserve both original labels. Record adjudication reason, unresolved status, adjudication fraction, training time, labeling time, correction real-time factor, and bounded remediation cycle if invoked.

One remediation cycle may clarify instructions with non-study examples and use a disjoint preselected block. It may not change the status ontology, ±5-second window, lag anchor, sample, or threshold.

## Frozen acceptance metrics

The packet supports, but does not yet satisfy, the following:

| Gate evidence | Required rule |
| --- | --- |
| Audio decode (`G3`) | At least 80% of sampled duration decodable. |
| Timing (`G3`) | At least 70% of speech-bearing windows human-correctable; accepted onset and offset each within 500 ms of adjudicated reference. |
| Speaker role (`G4`) | Four-class corrected macro-F1 at least 0.80, with classwise errors and nonspeech false detections. |
| Transcript (`G4`) | Source-language token/grapheme error against ASR-hidden adjudicated reference, before/after correction, plus uncertainty and labor. No machine output as gold. |
| Referential (`G6`) | At least 15% of validated non-child utterances have visible candidate(s); six-status pre-adjudication alpha or equivalent at least 0.67; null/ambiguity retained. |
| Boundary/ceiling (`G8`) | At least 80% of independently matched uncensored boundaries within 1 second. |
| Lexical support (`G5`/`G7`) | At least 8 noun/object and 6 verb/action types satisfy every frozen train/held-out participant and visible-link support rule in both split constructions. |

Metric output must include coverage, missingness, uncertainty, classwise errors, language strata, participant-cluster uncertainty, adjudication fraction, and effort. A point estimate cannot be improved by dropping uncertainty, null, ambiguous, overlap, undecidable, or unusable records.

## Instrument firewall during handoff

The default handoff is human-only. If the proposed annotation-instrument amendment is explicitly approved and every code/weight license is receipted, machine proposals may exist in a separate quarantine layer. Independent reference coders do not see them. Production-correctors must disposition the entire accepted record.

No instrument weight/checkpoint, tokenizer/vocabulary, prompt, feature, embedding, hidden state, score, confidence, token ID, alternate hypothesis, VAD segment, cluster, detector class, raw text/timing/role, or unreviewed label may enter a learner store. A later scratch tokenizer may use only protocol-authorized, fully human-dispositioned ChildLens training text. No learner or causal arm is authorized by packet completion.

## Return package from qualified humans

The restricted return stays inside quarantine and must contain:

- immutable coder assignment and blinding receipts;
- qualifications/language-competence and training records;
- locked independent Layer A–C records and preserved pre-adjudication labels;
- adjudication records and unresolved cases;
- metric code/version/test digest and aggregate reliability output;
- effort and correction-time records;
- accepted-derivative manifest digest and restricted retention status; and
- a privacy/export review.

The report namespace receives only terms-permitted, nonidentifying, cell-suppressed aggregates and digests. It receives no item key, participant/session identifier, path, filename, media, frame, audio, transcript fragment, lexical token, exact timestamp, or small cell.

## Handoff disposition

`HUMAN_VALIDATION = NOT_STARTED`.

The exact next task is: resolve exact-byte acquisition admission; populate the unchanged blinded skeleton inside the restricted local quarantine; freeze scoring code; and hand it to qualified language-matched humans for independent timing/text/role and referential coding plus adjudication. Return only the restricted completion receipt and permitted aggregates for mechanical application of frozen gates `G3`–`G8`.

Do not train a CLIP+ learner, build causal arms, inspect an acquisition effect, replace selected items for content reasons, or represent completion of the skeleton as completion of human validation.

