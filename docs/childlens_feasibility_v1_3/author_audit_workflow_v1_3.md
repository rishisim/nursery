# ChildLens v1.3 blinded single-author audit workflow

Version: `childlens-author-audit-workflow-v1.3.0`  
Status: initialized and ready for the blinded qualified-author pass  
Scientific learner or causal outcome authorized: **no**

## Purpose

This workflow replaces neither the v1.2 evidence nor its original two-human
gold-standard design. It implements the narrower v1.3 feasibility protocol: one
qualified author labels a deterministic, model-independent 15-minute sample,
then the locked human record may be compared with separately generated local
pseudo-labels. Inter-human reliability is explicitly unavailable. The only
agreement claim available from v1.3 is model–human agreement on the locked
subset.

The implementation was written and tested using synthetic temporary roots. No
real quarantine, frame, audio, transcript, identifier, timestamp, filename, or
model-label payload was opened during development.

## Components

- `scripts/childlens_author_audit_v1_3.py` is the owner-private SQLite
  controller and structural validator.
- `scripts/childlens_author_audit_app_v1_3.py` is the streamlined Streamlit UI.
- `scripts/initialize_childlens_author_audit_v1_3.py` is the zero-argument
  initializer. It inherits the unique verified v1.2 quarantine, locates the
  sealed primary packet by the public canonical digest, creates a separate v1.3
  policy receipt, and initializes no prediction path or mount.
- `scripts/launch_childlens_author_audit_v1_3.py` is the zero-argument app
  launcher. It uses a dedicated quarantine-confined browser profile and binds
  only to `127.0.0.1:8502`.
- `scripts/export_childlens_author_workflow_receipt_v1_3.py` is the
  zero-argument aggregate exporter. It reads only controller progress/state and
  writes the fixed repository-safe workflow receipt; it has no row, media,
  transcript, timing, identifier, prediction-payload, or credential export
  route.

## Frozen state machine and prediction firewall

The controller enforces:

1. `SAMPLE_FROZEN`
2. `AUTHOR_BLIND_OPEN`
3. `AUTHOR_LOCKED`
4. `PREDICTION_JOIN_ENABLED`

Initialization accepts only the primary sample. It has no prediction-binding
argument, does not store a prediction path or digest, and does not create a
prediction snapshot. The author app contains no prediction accessor or display
route. Only `attach_prediction_store_after_lock` can bind a separately produced
prediction store, and that operation rejects every state before the complete
author record is irreversibly locked.

This design is stronger than merely hiding preloaded predictions: the author
process cannot mount, parse, or address them while labels remain mutable.

## Author task

Every one of the 15 participant-distinct selected items contributes at least
one second and the primary sample totals exactly 15 speech-minutes. The UI
shows only opaque audit keys and the frozen raw audio/video clips. It asks the
author to record:

- audible language, language competence, audio usability, and whether
  whitespace tokenization is linguistically valid for WER;
- clip-relative utterance boundaries, verbatim source-language text, and
  `NON_CHILD`, `CHILD`, `OVERLAP`, `UNCERTAIN`, or `NONSPEECH`;
- coarse `VISIBLE_CANDIDATE`, `NULL_NOT_VISIBLE`, `IRRELEVANT`, `UNDECIDABLE`,
  or `UNUSABLE` status;
- noun/object and verb/action candidate decisions; and
- eligible mention spans, candidate-count bands, visible candidate time bands,
  and a reason for undecidable/unusable records.

The author must understand the encountered language well enough to correct
source-language transcripts and assign source role. Otherwise they must choose
the insufficient/uncertain/unusable routes and not guess.

Each save commits immediately with SQLite `FULL` synchronization. Drafts are
resumable. Item locks are irreversible. The final pass lock requires all 15
items, an explicit blinding confirmation, and includes the protocol digest,
restricted sample digest, every author row, and aggregate completeness in an
HMAC-protected record.

The initial scheduling estimate is **75 minutes**: 15 media minutes at four
annotation minutes per media minute, plus 15 minutes for item review and final
locking. It is an honest pre-use workload estimate, not a completed-author
measurement; live progress decreases it according to locked items.

## Local operation

After the model-independent primary packet exists inside the approved
quarantine, initialize once:

```bash
python3 scripts/initialize_childlens_author_audit_v1_3.py
```

Then launch or resume:

```bash
python3 scripts/launch_childlens_author_audit_v1_3.py
```

After initialization, export or refresh the aggregate readiness receipt:

```bash
python3 scripts/export_childlens_author_workflow_receipt_v1_3.py
```

All three commands reject every argument. The app launcher discovers exactly one
initialized owner-private runtime, uses an ephemeral URL nonce, authenticates
the sole local author pass without displaying its stored token, disables
browser background networking and telemetry, removes proxy variables, blocks
external host resolution, and confines browser profile/cache/download state to
quarantine. Close the dedicated browser window to exit safely. Run the same
launcher again to resume autosaved work.

Do not copy the local URL into a general browser, tunnel the port, enable remote
access, or attach predictions before the final author lock.

## Scientific boundary

Machine outputs remain pseudo-labels and measurement-instrument hypotheses.
Unaudited pseudo-labels may support only nonidentifying aggregate simulator
calibration and candidate generation. They cannot serve as primary evaluation
truth. The later synthetic causal evaluation must use simulator oracle labels.
No learner, corpus tokenizer, checkpoint, causal arm, or acquisition outcome is
created by this workflow.

## Synthetic validation

Run:

```bash
python3 -m pytest -q \
  tests/test_childlens_author_audit_v1_3.py \
  tests/test_launch_childlens_author_audit_v1_3.py
```

The tests cover model-independent sampling admission, variable per-item
allocation with an exact 15-minute total, quarantine confinement, autosave and
resume, language/WER decisions, ontology validation, mention spans and visible
time bands, uncertain/unusable routes, irreversible item and global locks,
prediction non-mounting before lock, post-lock join attachment, simulator-oracle
evaluation separation, exact schema compatibility, loopback browser confinement,
external host blocking, zero-argument initialization, and fixed non-disclosing
errors.
