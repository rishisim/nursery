# ChildLens v1.2 post-acquisition orchestrator

## Purpose and boundary

`run_childlens_post_acquisition_v1_2.py` is the no-argument bridge between a
sealed native-transfer receipt and the local human-validation workflow. It does
not acquire data, access Keychain, use a browser, contact a network service,
run ASR, train a learner, infer lexical units, or evaluate a causal arm.

The implementation treats the official speech annotations as coarse timed
speech-presence bouts. It coalesces only overlapping or touching copies of
those explicit intervals. It never calls an official bout an utterance, never
fills an unannotated temporal gap, and reports zero candidate utterances until
qualified humans create and adjudicate them. An item with no official speech
window remains an explicit empty-window item; it is not dropped or silently
converted into speech.

## Fail-closed run sequence

The command accepts no arguments and performs the following operations:

1. Uniquely discover the same owner-only hidden transfer bundle used by the
   native launcher.
2. Revalidate the sealed complete 15-item transfer receipt against its frozen
   plan and config.
3. Recheck owner-only access, the indexing sentinel, signed-control
   attestations, the retention deadline, the namespace cap, and the free-space
   floor.
4. Locate exactly one v1.1 skeleton and exactly one v1.1 restricted input by
   their frozen canonical digests.
5. Verify every acquired media byte count and content hash and every linked
   annotation file hash before parsing the annotation document.
6. Extract only explicit speech-like timed bouts and build an owner-only local
   measurement manifest. Each row records a PRESENT or ABSENT speech
   expectation, including an empty interval list where appropriate.
7. Build the owner-only 15-row human packet binding opaque key, source object,
   duration, metadata stratum, annotation linkage digest, and acquired-media
   digest. Rows are assigned to three five-item resumable batches by a
   release-bound opaque hash order.
8. Run the local restricted-media structural audit. The media tools are
   protocol-limited to local inputs, their output is parsed or discarded in
   process, the CPU worker cap is two, and heavy decode concurrency is one.
9. Bootstrap the human-workflow policy only from the already reverified
   controls, initialize the immutable workflow database, and run structural
   readiness validation.
10. Recheck capacity and atomically publish the three required aggregate
    receipts. Restricted manifests, row bindings, exact intervals, names,
    source locations, and per-item findings stay in quarantine.

## Public receipt contract

The only repository outputs created by a successful run are:

- `acquisition_receipt.json`, containing the frozen conservative admission
  arithmetic, aggregate 15-item/hash verification, release-binding limitation,
  actual capacity evidence, and acquisition-boundary attestations;
- `automated_diagnostics_receipt.json`, containing cell-suppressed structural
  metrics, actual-source digests, and the unioned official speech-window
  duration; and
- `human_validation_workflow_receipt.json`, containing the honest zero-
  utterance pre-referential state, the frozen 20% future double-code rule,
  workflow controls, and packet-specific first-human, second-human,
  adjudication, and total labor estimates.

For an all-pass sample the binary structural metrics expose exactly 15 passes
and zero failures. Mixed metrics suppress either side when a nonzero cell would
be smaller than five. A nonzero speech-window inventory smaller than five is
also suppressed and cannot support a human-ready handoff.

The labor calculation uses actual selected-item durations and the unioned
official speech-window duration, capped at the frozen 30-minute stopping-rule
route. It applies the already documented base planning rates: five times media
duration for timing/transcript/role work, eight times for referential work,
20% independent referential duplication, and a 40% adjudication allowance.
The estimate is operational planning evidence, not a measured annotation rate.

## Invocation and terminal output

Run only after the native launcher reports a sealed complete transfer:

```bash
python3 scripts/run_childlens_post_acquisition_v1_2.py
```

Successful stdout is one fixed JSON status line. Failures print only a fixed
error category; exceptions, paths, object keys, source names, exact windows,
and restricted values are never printed.

## Synthetic validation

The dedicated tests use only temporary synthetic bytes and annotations. They
cover explicit-window extraction, empty-window behavior, invalid speech timing,
canonical-digest uniqueness, annotation/media hash verification, owner-only
restricted outputs, hardened public receipt schemas, small-cell suppression,
fixed stdout, atomic public output, and an end-to-end initialization of the
real local workflow database with mocked structural measurement. They do not
inspect the real quarantine, Keychain, browser, or network.
