# ChildLens v1.2 privacy-validation handoff

This document defines the repository-side, aggregate-only acceptance boundary
for the v1.2 continuation. The validator reads only `docs` and `output`
artifacts plus the historical v1/v1.1 files needed to recompute immutability
digests. It does not inspect the restricted quarantine, Keeper, browser state,
email, media, annotations, transcripts, or identifiers.

## Required aggregate receipts

The coordinator must provide these JSON objects in the v1.2 output namespace:

- `acquisition_receipt.json`: frozen count, acquired and verified counts, total
  raw bytes, admission method/cap, and negative boundary attestations.
- `automated_diagnostics_receipt.json`: aggregate probe/decode/preparation
  counts and the local-only measurement-instrument firewall.
- `human_validation_workflow_receipt.json`: populated workload, frozen minimum,
  double-code target, blinding, resumption, autosave, and human-evidence state.
- `decision_record.json`: one terminal state and explicit scientific,
  cross-corpus, egress, privacy, and historical-preservation flags.
- `immutability_receipt.json`: recomputed canonical v1 and v1.1 set digests.

No row, name, source filename, participant/session/media identifier, local path,
exact timestamp, transcript text, frame, audio, video, manifest, or small cell is
permitted in this namespace. Digests and sufficiently aggregated counts are
permitted.

## Human-ready acceptance rule

The human-ready state is accepted only when all 15 frozen media objects were
acquired and verified, automated preparation was attempted for all 15, and the
local workflow is populated with at least 300 candidate utterances or 30 speech
minutes. It must retain a double-code target of at least 20%, hide identifiers
and source filenames, run and autosave locally in the restricted quarantine,
support resumable validated batches, preserve required blinding, and state that
genuine human validation remains incomplete. Model or Codex review can never be
recorded as human evidence.

## Mandatory incident disclosure

The v1.2 decision must explicitly disclose the local-tool-log quarantine-path
redaction incident using only a category-level description. The literal path
must not be repeated. The decision must separately attest that no corpus payload
was exposed in tool output and no restricted data left the local environment.

## Running the validator

Run:

```bash
python3 scripts/validate_childlens_feasibility_v1_2.py
pytest -q tests/test_validate_childlens_feasibility_v1_2.py
```

Failures suppress matched content. A passing result is a repository privacy and
consistency check, not a scientific feasibility result and not evidence that a
human completed the validation packet.
