# ChildLens pseudo-calibration coordinator design

Status: `REVISE_BEFORE_IMPLEMENTATION`  
Date: 2026-07-22  
Scope: protocol/code compatibility design only; no restricted or quarantine access

## Result

The frozen two-by-two instrument comparison can reuse the existing v1.3 privacy boundary, but it cannot reuse the existing runner unchanged. The smallest safe implementation is one new two-phase coordinator, `scripts/nursery_childlens_pseudo_calibration_v1.py`, with a pure aggregate-export function inside the same file and one synthetic-only test module, `tests/test_nursery_childlens_pseudo_calibration_v1.py`.

The coordinator must not modify or monkeypatch the historical v1.3 profile registry, runner, adapter, checkpoints, or receipts. It imports only stable content-agnostic firewall services, reads restricted rows only after re-exec under OS network denial, keeps normalized rows and comparisons in a new quarantine-only namespace, and returns one exact-schema aggregate receipt through a bounded inherited file descriptor. No item, text, timestamp, path, identifier, frame, audio, confidence, or model payload crosses into the repository.

Implementation should pause before model execution because two protocol issues require a frozen clarification:

1. The calibration protocol requires conservative envelopes rounded outward to the simulator-grid resolution but does not declare a resolution for any dimension.
2. `frozen_minimal_realism_protocol.json` explicitly forbids ChildLens pseudo-labels, aggregates, priors, calibration, or results from its ancestry. Therefore this calibration receipt cannot configure that experiment. It must remain a standalone triangulation artifact unless a separate future protocol, frozen before outcomes, explicitly permits it. The coordinator must fail closed if asked to export a binding for `nursery-synthetic-minimal-realism-extension`.

These are protocol clarifications, not reasons to weaken privacy controls or inspect predictions.

## Frozen inputs inspected

- `docs/nursery_program_convergence_v1/frozen_childlens_pseudo_calibration_protocol.json`
  - file SHA-256: `a39841818f7b4393e9dcc0f92ae282b7488780f70615a59a38c5bfdaf6fb9c2b`
  - status: `FROZEN_BEFORE_AGREEMENT_AGGREGATES`
- `docs/nursery_program_convergence_v1/frozen_minimal_realism_protocol.json`
  - file SHA-256: `662d8b8f988f2168b81bfce7d62a59855394398ac170eef052ab2303ddc1ec1c`
  - status: `FROZEN_FOR_IMPLEMENTATION_NOT_AUTHORIZED_FOR_SCIENTIFIC_OUTCOME`
- `scripts/childlens_local_inference_firewall_v1_3.py`
- `scripts/childlens_local_pseudo_adapter_v1_3.py`
- `scripts/run_childlens_model_assisted_pseudo_annotation_v1_3.py`

No historical namespace was edited and no historical output row was opened.

## Compatibility findings

### Directly reusable firewall services

The new coordinator should import these public, content-agnostic interfaces from `childlens_local_inference_firewall_v1_3.py`:

- `NetworkIsolationBackend.detect()` and `.command()`;
- `verify_network_isolation()`;
- `scrubbed_subprocess_environment()`;
- `validate_quarantine_root()`;
- `ResourceBudget(cpu_workers=2, mps_heavy_processes=1)`.

The existing MPS file-lock idea and owner-private atomic-write pattern should be reproduced in the new namespace, but private underscored functions are not a compatibility API. Every restricted inference subprocess must receive data and outputs through inherited descriptors, discard stdout/stderr, use the scrubbed offline environment, and run behind a fresh active socket-denial sentinel.

### Historical components that must remain untouched

- `FIXED_ADAPTER_PROFILES` contains Silero, Whisper turbo, the conservative role aid, and Qwen2-VL only. Its exact-set validation cannot accept Qwen3-ASR/aligner or Gemma 4.
- `build_aggregate_receipt()` intentionally emits only full-pilot completion/security counts; it does not accept prediction rows and cannot compute the new comparisons.
- `run_childlens_model_assisted_pseudo_annotation_v1_3.py` hard-codes 15 items, 912 candidate windows, and 135.25 speech minutes. Its restricted executor processes the full v1.3 pilot, not the frozen 900-second calibration slice.
- `childlens_local_pseudo_adapter_v1_3.py` emits the historical audio and Qwen2 referential schemas. The Qwen2 schema has referential status and noun/verb candidate lists, but not the new visibility, candidate-count, lexical-support, or lag bins.

Accordingly, the new coordinator may validate and reuse historical baseline files in place, read-only, inside quarantine. It may not update their checkpoint, rewrite their files, or present the challenger instruments as historical v1.3 profiles.

## Minimal two-phase coordinator

### Public phase

The zero-argument public phase performs no runtime-root discovery and cannot open quarantine. It must:

1. Verify the exact calibration-protocol file hash and exact schema.
2. Verify four fixed instrument receipts: Whisper turbo, Qwen3-ASR-1.7B plus forced aligner 0.6B, Qwen2-VL-2B, and Gemma-4-E4B-it 4-bit. Receipt values include authoritative license evidence, revisions, artifact hashes, adapter hashes, public-canary receipts, and `restricted_outputs_quarantine_only=true`; no path or URL enters the public receipt.
3. Refuse any automatic license click-through or missing public synthetic canary.
4. Detect network isolation and pass its active socket-denial sentinel.
5. Re-exec the same script with `--restricted-phase`, passing only a bounded sealed configuration through one inherited descriptor and accepting at most a small exact-schema aggregate on a result descriptor.
6. Validate the aggregate again, scan all string values for restricted-name/path/media/text patterns, then atomically write only `output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json`.

The public phase prints one fixed status token and no counts or metrics to stdout.

### Restricted phase

The restricted phase discovers the already approved runtime internally, validates its owner-private and out-of-repository controls, and creates a new owner-private `nursery_pseudo_calibration_v1` subnamespace. It must:

1. Validate the frozen sample-selection receipt digest and its structural declarations: 15 distinct frozen items, 900 total seconds, prediction-independent selection, no content/model field, and no overlap with the reserve.
2. Resolve the exact 15-minute intervals and opaque item linkage only from quarantine-local records. These values never enter argv, environment, logs, stdout/stderr, exceptions, or the public receipt.
3. Read existing Whisper and Qwen2 pseudo-label files read-only and verify their hashes against the existing checkpoint/receipt before reuse.
4. Run Qwen3-ASR then its forced aligner, and Gemma 4, sequentially under network denial. At most one MPS-heavy process runs at a time; CPU-only normalization/bootstrap uses two workers by default and no more than four.
5. Normalize all four paths into the frozen common schemas in quarantine.
6. Compute paired comparisons and calibration ranges in memory or owner-private temporary files, call the pure aggregate exporter, delete temporary decoded media, and return only the allowlisted aggregate through the descriptor.

There is one checkpoint row per `(sample_digest, instrument_receipt_digest, task_family, normalizer_version)`. Resume is hash-bound and may rerun only missing or failed rows. The single bounded engineering retry is recorded per path and may be triggered only by license/canary/network/schema/coverage failure, never disagreement, confidence, lexical content, visibility, or apparent quality.

Reserve activation is similarly structural: fewer than 12 primary items or fewer than 720 primary seconds must have decodable media and schema-valid output from both required paths in a task family. The exporter sees only a boolean reserve-activation receipt and aggregate eligible totals; it cannot inspect agreement to make this decision.

## Frozen sample/window crosswalk

The calibration protocol fixes the 15 speech minutes but does not specify how those minutes select historical referential windows. Before any challenger output is opened, the implementation receipt should freeze this content-independent crosswalk:

- speech comparison input is exactly the 15 preselected one-minute intervals;
- referential comparison units are existing frozen v1.3 candidate windows whose center lies in a selected half-open interval `[audit_start, audit_end)`;
- ties at the right boundary belong to neither adjacent interval unless that next interval is itself selected;
- no new content-, confidence-, or prediction-driven window is generated;
- both referential instruments receive the identical ordered window list and identical fixed five-frame offsets;
- window-set and sample-set digests remain quarantine-only; the repository receives only a digest of the sealed crosswalk algorithm/version, not item digests.

If this crosswalk is not accepted as a pre-output implementation clarification, the coordinator must stop. It may not choose a window rule after observing coverage or agreement.

## Common-schema bridges

### Speech

Both speech paths normalize to:

```text
utterance_intervals
word_intervals
source_language_code_hypothesis
transcript_hypothesis
abstain
```

Intervals and text remain restricted. Qwen forced alignment is valid only after the source language is verified as supported by the aligner. Whisper timing and Qwen timing are instrument hypotheses, not truth.

Boundary comparison uses maximum-cardinality bipartite one-to-one matching at the frozen 500-ms collar. Among maximum-cardinality matchings, use minimum total absolute onset-plus-offset error and then canonical index order as the deterministic tie-break. Precision, recall, and F1 are pairwise agreement measures.

For transcript disagreement, pair only matched, non-abstained German-source regions. Freeze normalization before execution: Unicode NFKC, Unicode casefold, punctuation-to-space, whitespace collapse, no translation, and preservation of German letters before casefold. Compute WER and CER symmetrically in both directions or report the mean of the two declared directional edit ratios; do not label either path a reference transcript. If the protocol owner does not approve this normalization/tie rule before outputs, transcript comparison remains blocked.

### Referential

Both paths normalize each frozen window to the exact bins in the calibration protocol. The historical Qwen2 output can directly bridge:

- `VISIBLE_CANDIDATE` → `visible_candidate`;
- `NULL_NOT_VISIBLE` or `IRRELEVANT` → `null_or_irrelevant`;
- `UNDECIDABLE` or `UNUSABLE` → `undecidable`;
- noun/verb candidate-list presence → coarse lexical support;
- union of de-duplicated noun/verb candidate strings → candidate-count bin.

It cannot validly supply `visibility_bin` or `lag_event_unit_bin`. Those fields must be `abstain`; they cannot be inferred from raw text, candidate count, or confidence. If this causes the frozen abstention/schema gate to fail, the one allowed engineering retry may run the same fixed Qwen2 path on the same 15-minute window set with a pre-frozen strict common-schema prompt. That retry is triggered by the known schema deficiency, not by model disagreement. It is still the Qwen2 path, not a third model vote. Gemma must emit the exact common schema directly and fail closed on extra keys or prose.

No majority vote, translation, confidence-weighting, lexical filtering, salience filtering, or model-as-human comparison is permitted.

## Pure aggregate exporter

The exporter is a pure function called only inside the restricted phase. It receives validated normalized rows plus fixed public protocol constants and returns an exact-key dictionary. It has no filesystem, subprocess, socket, logging, or model access.

It computes:

- speech boundary precision/recall/F1 at 500 ms;
- symmetric normalized WER/CER disagreement and language-ID agreement/abstention;
- raw agreement and Cohen kappa when defined for every referential field;
- per-path prevalence and abstention;
- item-cluster bootstrap intervals using 5,000 resamples and seed 730001;
- intersection prevalence where both paths emit the same non-abstain bin;
- union prevalence where either path emits a bin;
- disagreement as its own rate;
- the conservative minimum-lower/maximum-upper envelope only after grid resolutions are frozen.

Bootstrap clusters are the 15 selected items. All windows from a sampled item are resampled together. A fixed sorted metric registry controls RNG consumption so output is independent of dictionary iteration order.

The public schema contains only:

```text
schema_version
status
protocol_sha256
algorithm_receipt_sha256
instrument_receipt_sha256s
sample_declarations
path_gate_booleans
task_family_gate_booleans
agreement_metrics
calibration_dimensions
reserve_activation
privacy
scientific_boundary
consumer_compatibility
receipt_sha256
```

No instrument path, item/task row, text, language example, exact interval, filename, identifier, frame, audio, prediction, confidence, raw count, or failure message is allowed. Failures cross the boundary only as allowlisted fixed diagnostic codes.

For cell protection, the exporter does not publish exact item or event counts. It emits rounded proportions/intervals only when at least five item clusters contribute and uses a fixed `SUPPRESSED_K5` marker otherwise. If a published proportion plus a public denominator could reveal a 1–4 cell or its complement, suppress the complete categorical field, not just one bin. Zero and one prevalence are also represented by outward-rounded intervals rather than raw counts. Agreement and kappa are omitted when undefined rather than replaced with zero.

The current protocol does not give simulator-grid resolutions, so `calibration_dimensions.*.envelope` must remain `BLOCKED_GRID_RESOLUTION_UNFROZEN` and overall status must be `REVISE` until a signed immutable resolution supplement exists.

## Consumer boundary

The calibration output may describe provisional broad ranges for a future protocol that explicitly permits ChildLens aggregate calibration. It cannot be passed to a learner, tokenizer, checkpoint, primary evaluator, or simulator oracle.

It also cannot configure the current minimal-realism experiment. That protocol's ancestry firewall explicitly forbids ChildLens empirical data, pseudo-labels, aggregates, priors, calibration, and results, and its nonclaims disavow ChildLens calibration. The aggregate validator must therefore emit:

```text
consumer_compatibility.nursery_synthetic_minimal_realism_extension = false
```

and reject any request to write a minimal-realism config or calibration binding.

## Exact integration points

| Concern | Reuse | New coordinator responsibility |
|---|---|---|
| OS isolation | `NetworkIsolationBackend`, active sentinel | Re-exec restricted phase and recheck before every adapter family |
| Environment | `scrubbed_subprocess_environment` | Add only allowlisted offline/thread settings; no credentials or proxy inheritance |
| Quarantine | `validate_quarantine_root` | Explicit new child namespace, owner-only atomic files, no discovery in public phase |
| Resources | `ResourceBudget` and one-heavy-process policy | One MPS lock shared across Qwen3, Qwen2 retry, and Gemma; CPU 2–4 |
| Baselines | Historical Whisper/Qwen2 schemas and hashes | Read-only subset extraction and strict common-schema bridge |
| Challengers | Separately sealed offline executables | New exact receipt registry; never mutate v1.3 fixed profiles |
| Sample | Frozen selection receipt | Validate 15 items/900 seconds and pre-output crosswalk algorithm |
| Aggregates | v1.3 privacy philosophy only | New pure comparison/bootstrap/suppression exporter |
| Repository output | v1.3 bounded result-FD pattern | One new exact-schema aggregate under `output/nursery_program_convergence_v1` |

## Required synthetic-only tests

The single proposed test module should cover:

1. Exact calibration-protocol hash and schema; mutation fails closed.
2. Public phase cannot call runtime discovery, open quarantine, or accept paths in sealed configuration.
3. Restricted phase requires the active network sentinel; socket-capable and forged backends fail.
4. Scrubbed environment removes credentials, proxies, telemetry, and user-site loading.
5. Exact four-instrument registry, licenses, revisions, hashes, and public canaries; extra/missing profiles fail.
6. Historical profile registry and v1.3 files remain byte-identical.
7. Exactly 15 distinct items and 900 seconds; model/content-dependent selection fields fail.
8. Deterministic center-in-half-open-interval referential crosswalk, including boundary ties.
9. Existing baseline hashes are validated before reuse; coordinator never writes historical outputs/checkpoints.
10. Challenger rows and schema-extension retry write only in the new owner-private namespace.
11. Exactly one bounded retry per path and structural-only reserve activation; disagreement cannot trigger either.
12. Qwen3 aligner rejects an unsupported language rather than emitting invented timing.
13. Qwen2 bridge maps only supported fields and abstains on visibility/lag; no illicit derivation.
14. Strict Gemma/common schema rejects prose, extra keys, confidence, raw model response, and unknown bins.
15. Maximum-cardinality 500-ms matching with deterministic minimum-error tie-break on synthetic intervals.
16. German normalization/WER/CER is symmetric and never invokes translation.
17. Refer-ential raw agreement/kappa handles undefined marginals without fabrication.
18. Cluster bootstrap uses 15 item clusters, 5,000 fixed-seed resamples, and is order invariant.
19. Intersection, union, disagreement, and envelope calculations use no vote or human-truth language.
20. Complementary K=5 suppression prevents reconstruction of 1–4 cells.
21. Aggregate exact-schema validator rejects paths, identifiers, timestamps, media suffixes, transcript-like keys, examples, item rows, confidence, NaN, and free-form errors.
22. One MPS lock spans all heavy paths; CPU worker bounds are 2–4.
23. No instrument artifacts, tokenizer, vocabulary, embeddings, features, scores, or pseudo-labels enter learner/evaluation outputs.
24. Minimal-realism consumer binding is rejected because its frozen ancestry firewall forbids ChildLens calibration.
25. All tests operate on synthetic sentinel rows only and require no quarantine.

## Exact next action

Before implementation, freeze a short immutable clarification containing:

- simulator-grid resolution per calibration dimension, or an explicit statement that the receipt is descriptive and cannot form simulator envelopes;
- approval of the deterministic referential-window crosswalk;
- approval of the boundary-match tie-break and German normalization;
- confirmation that the output is standalone and must not configure the current minimal-realism protocol.

After that clarification, implement only the new `nursery_*` coordinator/test files, run public synthetic canaries, and validate the aggregate exporter with synthetic sentinels. Restricted inference remains a separate authorized execution step.
