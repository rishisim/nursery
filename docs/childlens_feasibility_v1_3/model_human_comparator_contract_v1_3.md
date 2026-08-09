# ChildLens v1.3 post-lock model–human comparator contract

Status: implementation contract, frozen protocol consumer  
Implementation: `scripts/compare_childlens_model_human_audit_v1_3.py`

## Purpose and scientific boundary

The comparator evaluates a locked, blinded `AUTHOR_AUDIT_A` record against the
fixed local v1.3 measurement-instrument hypotheses. It does not turn those
hypotheses into ground truth. The author-audited sample is the only
human-labeled ChildLens evidence; unaudited pseudo-labels remain limited to
nonidentifying aggregate simulator calibration and candidate generation. A
later causal evaluation must use simulator oracle labels as its primary truth.

The program contains no network, hosted-model, media-decoding, training, or
learner path. Its executable entry point takes zero arguments and discovers the
single initialized owner-private quarantine internally. It emits only a fixed
status code to stdout and the aggregate repository receipt
`output/childlens_feasibility_v1_3/author_audit_result_receipt.json`.

## Fail-closed ordering

The comparator performs these operations in order:

1. validate the v1.3 quarantine, SQLite integrity, frozen protocol digest, and
   primary sample digest;
2. require exactly one irreversible author-pass lock and validate the complete
   author record HMAC;
3. require all fifteen item records to be locked and every annotated item to
   carry a blind author competence disposition of native, fluent, or
   proficient;
4. prove from monotonically ordered audit-log event IDs that no prediction join
   occurred before the author-pass lock;
5. require the full-pilot local pseudo-annotation receipt and all fifteen exact
   audio/referential pseudo-output files;
6. construct a restricted prediction store keyed only by the pre-frozen opaque
   join keys and media digests, bind its exact SHA-256, and call the existing
   controller's post-lock-only attachment operation; and
7. recompute the author-record HMAC and event ordering before comparison.

An unlocked, incomplete, modified, unqualified, digest-mismatched, or prematurely
joined record is rejected with a payload-free error. The prediction store,
binding, raw hypotheses, author rows, and metric sufficient statistics remain
owner-private in quarantine.

## Metric implementation

All times are converted internally to integer milliseconds. Instrument
intervals are clipped to the frozen primary sample before scoring.

### Speech and utterance boundaries

ASR segments are the machine utterance hypotheses. Human/machine segments are
matched one-to-one by maximum cardinality at the declared 1,000 ms both-edge
tolerance; among maximum-cardinality matchings, total temporal intersection over
union is maximized. The strict metric is the fraction of 1,000 ms matches that
also admit a one-to-one both-edge match at 500 ms. Speech-time recall and
precision use the union of human linguistic utterances and the clipped Silero
VAD union.

### Source role

The frozen contrast is `NON_CHILD` versus `CHILD/OVERLAP`. The separately
generated visual-context role proposal is used only when the audio role aid
abstains. `UNCERTAIN`, an unmatched hypothesis, or an absent hypothesis remains
an abstention: it lowers decision coverage and class recall and is never
reassigned. Unmatched decisive machine hypotheses count as false positives.

### Non-child transcription

The comparator applies Unicode NFC, Unicode casefolding, removal of Unicode
punctuation/symbol categories, whitespace collapse, no translation, and no
external lexicon. CER is always computed on usable non-child author references.
WER is computed only for items whose author locked whitespace tokenization as
linguistically valid before reveal. Unmatched reference text is deletion error;
unmatched machine non-child text is insertion error. Usable coverage is the
fraction of human non-child speech duration with a nonempty temporally matched
ASR hypothesis.

### Coarse reference

Every human non-child utterance is retained. The four classes are visible,
null, irrelevant, and undecidable/unusable. The fixed machine window with
maximum temporal overlap supplies the prediction; absence maps to
undecidable/unusable. Decision coverage excludes machine
undecidable/unusable outputs but those rows remain in exact agreement and macro
F1.

### Noun/object and verb/action candidates

The unit is an independently recorded visible human mention opportunity. Every
nonempty local model proposal in a temporally overlapping non-child audit window
enters the precision denominator. A one-to-one match requires the same family,
the exact frozen-normalized author mention string in the temporally aligned ASR
hypothesis, and intersection of the model window with the human visible band
plus the frozen 1,000 ms collar. No embedding, external lexicon, translation,
confidence, or semantic fallback is used.

## Cluster uncertainty and dispositions

The selected item is the cluster. The deterministic seed is SHA-256 of the
frozen protocol digest concatenated with the restricted sample digest. Each of
10,000 replicates resamples fifteen whole items with replacement and recomputes
pooled numerator/denominator ratios. The comparator reports 90% percentile
bounds, requires at least 80% defined replicates, and applies every point,
bound, support, and hard-failure threshold from the frozen protocol without
editing it.

Leave-one-item-out sensitivity is exported only as the maximum absolute metric
change; no responsible item is named. A gate cannot pass if one item supplies
more than 30% of its eligible support. Primary support shortfalls or missed
pass/bound conditions are `BORDERLINE`; a frozen hard-failure crossing is
`HARD_FAIL` only with adequate human support. After the one reserve, support
shortfalls become `NOT_ESTIMABLE` and no further expansion is permitted.

## One-time reserve

If and only if the primary result contains at least one `BORDERLINE` gate and no
`HARD_FAIL`, the comparator finds the already pre-frozen disjoint 900-second
reserve by its public digest and writes one immutable restricted activation
seal. The seal records activation count one, maximum count one, the locked
primary author-record binding, and `BLINDED_AUTHOR_LOCK_PENDING`. It never mounts
reserve predictions. Re-running the comparator can only verify the byte-identical
seal; it cannot activate a second reserve. A hard failure or all-pass primary
result cannot activate the reserve.

The aggregate result reports that the reserve is pending. Human completion and
irreversible lock of that already activated reserve are required before a
pooled primary-plus-reserve comparison; absence of those human labels is never
filled by a model.

## Repository privacy

The public result contains six gate statuses, supported aggregate metric/bound
values, support-threshold dispositions, aggregate maximum cluster share,
aggregate maximum leave-one-out change, and scientific-boundary attestations.
Observed support below five is suppressed. It contains no item rows, author or
machine text, language string, identifiers, filenames, paths, exact times,
frames, media, model-label payloads, or small cells.

## Synthetic validation

`tests/test_compare_childlens_model_human_audit_v1_3.py` uses generated strings,
times, roles, and candidates only. It verifies:

- exact normalization and edit distance;
- maximum-cardinality/maximum-IoU temporal matching;
- all six passing gates under adequate synthetic support;
- supported hard failure cannot trigger the reserve;
- primary support shortfall is borderline and reserve-eligible;
- byte-identical one-time reserve sealing;
- refusal of an unlocked synthetic author database; and
- absence of network, subprocess, or hosted-model client imports.

These tests do not discover or access the ChildLens quarantine.
