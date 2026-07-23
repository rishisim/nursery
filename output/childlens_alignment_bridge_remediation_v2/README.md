# ChildLens alignment-bridge remediation v2 artifacts

Canonical artifacts:

- `go_no_go_report.md` — concise scientific decision and diagnosis.
- `go_no_go_report.json` — machine-readable decision.
- `remediation_report.json` — direct aggregate output from the frozen evaluator.
- `freeze_receipt.json` — v1, instrument, and development-manifest bindings.
- `segment_freeze_receipt.json` — shared-segment hash frozen before ASR.
- `validation_receipt.json` — immutability, artifact-hash, validation, and
  mechanical-repair record.

All row-level segments, transcripts, audio, embeddings, identifiers, paths, and
scores remain in the owner-private quarantine. V1 artifacts were neither
modified nor overwritten.
