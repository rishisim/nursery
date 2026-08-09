# Causal outcome packaging interface v1

Status: **implemented and tested; no outcome read or packaged**

This additive interface is the only intended bridge from a completed v2 one-shot run to the frozen Michael-facing reporting shell. It accepts repository-public aggregate artifacts only. It never opens the owner-private authorization seal: the caller supplies its SHA-256 as an opaque binding.

## Inputs

The packager accepts exactly:

1. the passed public Gemma-plus-Qwen calibration receipt;
2. the passed v2 pre-outcome receipt;
3. the 64-hex SHA-256 of the owner-private one-shot authorization seal;
4. v2 `aggregate_results.json`; and
5. v2 `execution_receipt.json`.

It rejects fixture results, failed or incomplete gates, automatic fallback, unexpected schemas, symlinks, nonfinite values, row-level additions, invalid bundle inventory, merge-digest mismatch, and existing output files. It validates the original 40-corpus aggregate cells, changes only the field name `corpus_count` to `synthetic_corpus_count`, and omits seed order and bundle hashes from the reporting projection.

No calibration ranges, empirical counts, identifiers, payloads, paths, examples, per-seed rows, or authorization-seal contents are copied into the projection.

## Three-step interface

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/package_nursery_causal_outcome_reporting_v1.py \
  --calibration-receipt <PUBLIC_CALIBRATION_RECEIPT> \
  --preoutcome-receipt <PASSED_PREOUTCOME_RECEIPT> \
  --authorization-seal-sha256 <64_HEX_SHA256_ONLY> \
  --aggregate-results <V2_AGGREGATE_RESULTS> \
  --execution-receipt <V2_EXECUTION_RECEIPT> \
  --output-projection <NEW_AGGREGATE_PROJECTION>

PYTHONDONTWRITEBYTECODE=1 python3 scripts/generate_nursery_causal_central_figure_v1.py \
  --aggregate-input <NEW_AGGREGATE_PROJECTION> \
  --output-svg <NEW_CENTRAL_FIGURE_SVG> \
  --output-summary <NEW_FIGURE_SUMMARY>

PYTHONDONTWRITEBYTECODE=1 python3 scripts/synthesize_nursery_causal_outcome_memo_v1.py \
  --projection <NEW_AGGREGATE_PROJECTION> \
  --figure-summary <NEW_FIGURE_SUMMARY> \
  --figure-svg <NEW_CENTRAL_FIGURE_SVG> \
  --output-memo <NEW_MICHAEL_MEMO>
```

Every output is exclusive-create. Re-running with an existing target fails rather than overwriting history.

## Decision behavior

- A valid one-shot result with all 40 cells passing becomes `POSITIVE_PROTOTYPE_EFFECT`.
- A valid one-shot result with any directional cell not passing becomes `VALID_NULL_OR_NONPROMOTION`.
- A Gemma, calibration, pre-outcome, privacy, ancestry, pairing, falsification, or schema failure produces no projection, figure, or result memo. Under the exact override SHA-256 `cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7`, the next state is a user decision point with no automatic fallback.

The memo includes the fixed caveat box verbatim and cannot convert a technical failure into a valid null.
