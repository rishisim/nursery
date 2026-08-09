# ChildLens v1.2 human-validation workflow handoff

Status: **ready for authorized human validation**. The local blinded workflow is documented in [human_validation_workflow_v1_2.md](human_validation_workflow_v1_2.md), with operational decisions and reliability rules frozen in [human_validation_codebook_v1_2.md](human_validation_codebook_v1_2.md).

All 15 frozen media objects have been selectively acquired, verified, opaquely prepared, and admitted to the restricted local packet. The repository-safe `human_validation_workflow_receipt.json` was computed from that actual packet. It records 135.25 minutes in 912 official speech-presence windows. These windows are candidate review intervals, not utterances; the genuine utterance count remains zero until qualified humans segment and adjudicate speech.

Byte-identical selected source objects may share one content-addressed local media path. Their distinct frozen source-object keys and each key→object→SHA/path binding remain mandatory and are checked against both the frozen restricted input and sealed native-transfer receipt. The runtime database therefore preserves 15 blinded items without requiring 15 duplicate full-resolution files.

Once populated, the human app starts only through the zero-argument launcher. It discovers exactly one initialized owner-only runtime and fails closed on none or ambiguity. The restricted root is not accepted in launcher argv and is removed from child environments; the browser receives relative profile/cache/download names from a quarantine-confined working directory.

The first authorized human may complete one assigned independent pass. A different qualified and authorized person must complete each paired second pass. A qualified adjudicator receives the locked A/B records only after independence is preserved. Codex cannot fill any human or adjudicator slot.

## Actual aggregate workload

- Selected media source count: **15**.
- Candidate review material: **135.25 minutes** across **912 official speech-presence windows**.
- Frozen validation stopping rule: the first of **300 adjudicated linguistic utterances** or **30 adjudicated speech minutes**.
- Referential double coding: **at least 20%**, deterministically frozen only after the eligible adjudicated `NON_CHILD` inventory exists.
- Estimated first authorized-human effort: **420 minutes** (about 7 hours).
- Estimated second independent-human effort: **228 minutes** (about 3.8 hours).
- Estimated adjudication effort: **251 minutes** (about 4.2 hours).
- Estimated total qualified-human effort: **899 minutes** (about 15 hours), in small resumable batches.
- Actual language route: **not yet established**; both initial language coders must be qualified for the language they report, and adjudication fixes the route before transcription proceeds.
- Remaining feasibility evidence: genuine independent human validation only.

The estimates are scheduling estimates from the populated packet, not scientific evidence. Actual effort can vary with language, intelligibility, overlap, and the number of adjudicated `NON_CHILD` utterances.

## Launch and pass assignment

From the repository root, run exactly:

```bash
python3 scripts/launch_childlens_human_validation_v1_2.py
```

The launcher takes no arguments, discovers the single initialized owner-only runtime, and opens the confined local browser session. Do not copy its URL into a general browser.

The first qualified human completes only the assigned A passes. A different qualified and authorized human must complete the corresponding B passes; a second token used by the same person is not independent. A qualified adjudicator works only after the paired records lock. The required sequence is language A/B and language adjudication; timing, source-language text, and role A/B to the frozen stopping point; timing adjudication; deterministic referential inventory freeze; referential A and the assigned B subset; then pre-adjudication reliability and adjudication.
