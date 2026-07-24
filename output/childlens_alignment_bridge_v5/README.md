# ChildLens calibration recovery v5

This is the privacy-safe public record for the separately versioned
“scale and lag-estimand correction” attempt.

V5 stopped fail-closed at Stage 0 because inventory parsed legacy mixed-scope
manifests; zero locked-row loading therefore cannot be certified. No new media
was acquired or decoded, and no ChildLens embeddings, training, scoring, lag
curve, or simulator work ran.

- `stage0_freeze_and_stop_receipt.json` binds the prospective protocol, code,
  public model revisions/hashes, immutable v1–v4 trees, and stop condition.
- `synthetic_positive_control.json` records the outcome-blind embedding-level
  code sensitivity check; it is not ChildLens evidence.
- `development_decision.{json,md}` records the terminal
  `NO_GO_UNINFORMATIVE`.
- `validation_receipt.json` validates the terminal record and privacy guard.

The exact v5 protocol is in
`docs/childlens_alignment_bridge_v5_protocol.md`; all thresholds remain
prospective because no ChildLens audiovisual outcomes were computed.
