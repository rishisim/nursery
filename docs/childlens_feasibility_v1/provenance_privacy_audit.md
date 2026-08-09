# ChildLens feasibility provenance and privacy audit

Version: `childlens-feasibility-provenance-privacy-audit-v1.0.0`  
Audit date: 2026-07-21  
Scope: governance and validation design; no scientific outcome

## Audit conclusion and current limit

The ChildLens feasibility namespace is designed to have exactly one empirical source: the currently accessible, version-pinned ChildLens Keeper release. Historical adult-corpus empirical material and BabyView empirical material are prohibited from its calibration, vocabulary, tokenizer, learner, evaluation, selection, and interpretation ancestry. Public BabyView/EgoBabyVLM papers or public code interfaces may appear only as identified methodological references; their data, measurements, distributions, vocabulary, tokenizers, weights, checkpoints, and results remain excluded.

No raw ChildLens media, frame, or transcript was downloaded or written to the feasibility report namespace. However, the authenticated Keeper metadata inspection had a tooling privacy incident: an initial full-page DOM response and a later malformed schema-extraction response transiently emitted restricted filename/participant/timing values into browser tool responses. This report intentionally does not repeat any value. The coordinator reports that none of those values was downloaded as a corpus file, saved to disk, copied into an artifact, or sent to an external service, and subsequent extraction was aggregate/redacted.

The incident means this audit is **CONDITIONAL / FAIL-CLOSED**, not evidence of zero tool-output leakage. Historical tool responses cannot be made nonexistent by a clean report-directory scan. The audit otherwise reviewed only the new feasibility namespace and the named ChildLens construction contract, measurement specification, readiness report, and parallel design; it did not inspect historical empirical namespaces.

This report records the fail-closed design and the checks available at audit time. Final “no restricted payload leakage” evidence requires a clean run of `scripts/validate_childlens_feasibility_v1.py` after every required artifact, including the final decision record and evidence index, exists. A partial namespace or a nonzero validator result is not a privacy clearance.

## Allowed ancestry

The only permitted empirical flow is:

`pinned ChildLens release → restricted human-validated derivatives → permitted nonidentifying aggregates and/or fresh ChildLens-local scientific artifacts`

The following boundaries are mandatory:

- The exact release identity and its nonidentifying aggregate manifest are versioned; file-level paths, names, IDs, exact timestamps, and row linkages stay in a restricted manifest.
- A fresh tokenizer and every scientific learner initialize within this ChildLens instance. No parent checkpoint or cross-corpus cache is accepted.
- Raw modalities documented as absent—IMU, touch/contact, proprioception, and motor streams—remain explicitly unavailable. Simulator-defined counterfactual streams have procedural definitions, not ChildLens measurement ancestry.
- Fixed pretrained ASR, diarization, VTC, or vision tools are measurement instruments only under the proposed amendment. Their raw text, labels, vocabulary, representations, scores, and weights do not cross into learner artifacts.
- Methodological references are recorded as references, never as empirical inputs.

## Restricted/reportable separation

| Restricted, never reportable | Reportable only when terms permit |
| --- | --- |
| Video, audio, frames, thumbnails, clips | Release-level counts and byte totals |
| Raw or corrected transcript text and lexical strings unless an explicit vocabulary export is permitted | Cell-suppressed distribution summaries without text |
| Participant, child, session, episode, media, speaker-cluster, and row identifiers | Counts after participant-aware aggregation and suppression |
| Exact media timestamps, utterance boundaries, track coordinates, file paths/names, hashes linkable to individual files | Aggregate durations, coverage, uncertainty, and procedural digests |
| Instrument hypotheses, embeddings, logits, scores, voiceprints, and detector/VTC features | Instrument identity/version and human-reliability aggregates |
| Small or complementary cells that could reconstruct a person or lexical example | Cells of at least five after complementary suppression/coarsening |

The report namespace is not a de-identification workspace. Redaction after copying is insufficient; restricted row-level fields must never be written there.

## Nonidentifying manifest rule

The reportable manifest may contain release identity, source observation date, counts and byte totals by broad top-level class, release-level media/annotation totals when permitted, field-availability booleans, grouping counts after suppression, and canonical digests of reportable metadata. It must not contain file-level records, relative paths, filenames, remote object versions, participant/session keys, per-media checksums, codecs linked to a file, or exact per-media duration/timing.

An internal restricted manifest may contain those operational fields only inside quarantine. Internal keys should be release-namespaced HMAC or cryptographic derivations, not copied source identifiers. Its existence and aggregate checksum may be receipted without publishing its contents or path.

## Annotation-instrument privacy controls

The proposed amendment in `proposed_annotation_instrument_contract_amendment.md` is part of this audit. It requires:

- content-blind metadata sampling before instrument output exists;
- local, release-specific quarantine with a field allowlist and one-way release;
- no cloud upload or remote inference over restricted payloads;
- full instrument/model/version/license receipts without restricted example values;
- human verification or correction of every learner-visible text record;
- an ASR-hidden independently transcribed subset to quantify anchoring bias;
- double-coded timing/speaker validation and double-coded referential reliability;
- no speaker identity inference or reusable voiceprint;
- no raw hypothesis, token ID, vocabulary, score, feature, embedding, weight, box, track, or cluster ID in learner/evaluation records; and
- retention/deletion receipts for quarantined intermediates.

## Validator design

The validator is intentionally conservative and reads only `docs/childlens_feasibility_v1/` and `output/childlens_feasibility_v1/`. It:

1. requires the canonical document and evidence inventory;
2. rejects symlinks and media/subtitle payload extensions in the report namespaces;
3. parses every JSON artifact and flags row-level identifier, transcript/media, file-level locator, and exact media-timing keys;
4. flags direct identifier assignments, media filenames, subtitle/dialogue patterns, and clock-level media timestamps in report text without echoing the matched value;
5. flags forbidden empirical ancestry in machine-readable source, calibration, tokenizer, checkpoint, weight, or learner fields, while allowing explicit exclusion receipts and identified public method/citation fields;
6. loads the frozen rubric, requires one assessment for every gate, verifies the terminal decision against gate statuses, and requires the executive report to name the same single terminal state; and
7. fails when the final decision record is absent. During active assembly, that failure is expected and must not be waived for a terminal audit.

Pattern checks are defense in depth, not a proof of anonymization. A human privacy review must also confirm that aggregate combinations, prose, and complementary cells do not disclose a person or recoverable utterance.

## Tooling privacy incident and remediation

Incident class: restricted metadata disclosure to an authenticated browser-tool response. The affected response classes were a broad DOM snapshot and a malformed attempt to extract table/CSV schema. No underlying value is reproduced here. The incident did not authorize expanding inspection and is not converted into evidence for any feasibility gate.

Containment recorded by the coordinator:

- stopped broad DOM/table/CSV response emission;
- did not download a media archive or restricted corpus file;
- did not copy response values into documents, evidence JSON, commands, filenames, or local manifests; and
- limited subsequent browser extraction to aggregate/redacted output.

Required remediation before any further authenticated inspection:

1. Perform redaction and aggregation inside page-local browser JavaScript before a value can be returned to the tool response. Return only an explicit allowlisted object of counts, byte totals, booleans, broad schema categories, and public release identity.
2. Prohibit full-page DOM snapshots, raw table text, directory listings, response bodies, CSV rows, and generic object serialization on authenticated ChildLens pages.
3. For schema checks, parse headers in page memory and emit only allowlisted field-availability booleans. A parse failure must return a constant error code, never fallback content or the malformed input.
4. Exercise the extractor against a synthetic page containing sentinel identifiers and timestamps; fail unless no sentinel reaches the returned object or error path.
5. Apply output-size and key allowlists before every browser call, then inspect the proposed script itself for fallback branches that could serialize source content.
6. Keep a value-free incident receipt in the evidence index and have a second reviewer confirm that the on-disk feasibility namespace contains none of the transient values.
7. Do not use the incident response content to choose pilot items, estimate empirical quantities, populate a manifest, or adjudicate a gate.

These controls reduce recurrence risk but do not retroactively erase the incident. Any final handoff must disclose it and distinguish “no restricted payload in report artifacts” from the false stronger claim “no restricted payload ever appeared in tool output.”

## Evidence-index requirements

The final `output/childlens_feasibility_v1/audit_evidence_index.json` should list every required public artifact, its SHA-256 digest, origin workstream, evidence class, restriction class, and validation status. Digests must be computed over reportable artifacts only. The index must not include restricted filesystem paths or file-level corpus checksums.

The final audit handoff must record:

- validator command and exit status;
- unit-test command and result;
- machine-readable parse result for every JSON artifact;
- manual review of all exported prose and tables for examples, identifiers, small cells, and complementary disclosure;
- confirmation that no license was accepted, no Keeper share changed, and nothing was uploaded, emailed, committed, pushed, or published; and
- unresolved limits, including any term-dependent processing or language/transcription evidence not obtained.

## Fail-closed disposition

Any restricted payload finding, unexplained empirical ancestry, missing receipt, mismatched decision, incomplete gate inventory, or missing required artifact blocks a clean audit. The finding must be corrected in the report namespace or escalated as a blocker; it cannot be silenced by changing the scientific claim. The validator does not decide scientific feasibility and this document makes no terminal feasibility decision.
