# ChildLens v1.3 frozen model-assisted annotation and single-author audit protocol

Version: `childlens-model-assisted-author-audit-protocol-v1.3.0`  
Freeze state: **frozen before author labels or model–author comparison**  
Scientific learner outcome authorized: **no**

## What v1.3 changes—and what it does not

V1.3 replaces the operationally infeasible two-human validation requirement for this feasibility prototype with a fixed local pseudo-annotation pass and one blinded, qualified-author audit. It does not overwrite v1, v1.1, or v1.2. In particular, the v1.2 two-human design remains the historical gold-standard route for any future publication-grade annotation claim. V1.3 explicitly cannot estimate inter-human reliability and must report model–human agreement, not annotator reliability or machine accuracy against gold.

Machine records are measurement-instrument hypotheses. The locked author subset is the only human-labeled ChildLens evidence. Unreviewed pseudo-labels may support only permitted nonidentifying aggregate simulator calibration and candidate generation. They cannot be primary evaluation truth. The later synthetic causal evaluation must use simulator oracle labels.

This protocol was written without opening ChildLens media or inspecting any restricted frame, audio, text, identifier, filename, timestamp, annotation, or model output. Its frozen payload SHA-256 is recorded in [the machine-readable protocol](frozen_model_assisted_author_audit_protocol_v1_3.json).

## Prediction-independent 15-minute sample

The primary audit is exactly 900 seconds drawn from official speech-presence windows across the unchanged 15 participant-distinct selected items. Every item must contribute positive duration, and the preferred allocation is 60 seconds per item.

The authoritative restricted runtime sampler uses only the frozen selection binding, an opaque item key, content-addressed media binding, media duration, and official window intervals. It rejects every extra input field. It:

1. sorts and unions official windows within each item;
2. hash-orders items from the frozen v1.2 selection binding, fixed sampler version, allocation/placement purpose, and opaque item key;
3. initializes both primary and reserve to the smaller of 60 seconds or half the item's available unioned speech;
4. requires at least one second from every item in each sample;
5. allocates primary and then reserve deficits from remaining capacity using distinct frozen hash donor orders;
6. uses distinct hashes to choose leading slack, a middle gap, and primary/reserve orientation on the unioned speech-time measure; and
7. maps both blocks to physical official windows, verifies disjointness, and freezes exactly 900 seconds in each sample.

This rule is independent of predictions, confidence, transcript or lexical content, frames, visual salience, referential status, apparent success, and learner outcomes. Exact selections remain restricted. Repository output may contain only the sample digest, total duration, represented-item count, an observed “one minute per item” boolean, and permitted cell-suppressed aggregates.

Sampling fails closed unless both the primary and disjoint reserve can each supply exactly 900 seconds and every item can contribute at least one second to each. The preferred 60 seconds per item is reported as achieved only if it is actually achieved; deterministic redistribution is not disguised as equal allocation.

## Blinded author record and irreversible lock

The author uses raw local audio/video and sees no prediction, confidence, instrument identity, or prediction-derived ordering. The annotation app must not mount or query the prediction store while the route is in `AUTHOR_BLIND_OPEN`. The author records:

- source-language utterance boundaries and transcript text;
- `NON_CHILD`, `CHILD`, `OVERLAP`, `UNCERTAIN`, or `NONSPEECH`;
- `VISIBLE_CANDIDATE`, `NULL_NOT_VISIBLE`, `IRRELEVANT`, `UNDECIDABLE`, or `UNUSABLE` in the fixed ±5-second window; and
- independent noun/object and verb/action mention spans, candidate-count bands, and candidate time bands when visible.

The author must understand the encountered language well enough to correct source-language text and assign roles. Otherwise the author marks uncertainty/unusability and does not guess.

Comparison is disabled until a completeness receipt and cryptographic lock bind the protocol digest, restricted sample digest, language/normalization applicability choices, and every author row. Locked rows are append-only historical evidence; update or deletion invalidates the run. Only after `AUTHOR_LOCKED` may a separate local process join model hypotheses for aggregate agreement measurement.

V1.3 requires no second human and no adjudicator. Those roles cannot be simulated by Codex or another model.

## Frozen comparison definitions

Timing matches are one-to-one, maximizing cardinality and then temporal intersection without using text, role, or confidence. The strict boundary measure requires both onset and offset within 500 ms; segment F1 uses a 1,000 ms collar.

The primary source-role contrast is `NON_CHILD` versus `CHILD_OR_OVERLAP`. A machine `UNCERTAIN` is an abstention that lowers coverage and recall. Human `UNCERTAIN` rows remain in missingness reporting. `NONSPEECH` contributes to segmentation false-positive accounting.

CER is always calculated on usable non-child reference text. WER is gated only if the author locks whitespace tokenization as linguistically meaningful during the prediction-blind language stage. Text normalization uses Unicode NFC, Unicode case-folding, punctuation/symbol removal, and whitespace collapse; it never translates or imports an external lexicon. Unmatched human text counts as deletions and unmatched machine text as insertions.

Coarse reference has four comparison states: visible, null/not-visible, irrelevant, and undecidable/unusable. Null, ambiguity, uncertainty, and technical failure remain in the appropriate denominators. Noun/object and verb/action candidates match only when the independently annotated mention family agrees, the machine span maps to the locked human transcript, and the proposed visible time point/band intersects a human band with the frozen one-second collar. Matching is one-to-one.

## Frozen point and uncertainty thresholds

The pass thresholds are deliberately calibrated for candidate generation and aggregate simulator calibration, not publication-grade labels. Non-child precision is comparatively strict because child-speech leakage would directly distort the intended input ecology. Verb/action coverage is less demanding than noun/object coverage because dynamic events are temporally and linguistically sparser. None of these thresholds authorizes pseudo-labels as evaluation truth.

| Gate | Minimum audit support | Point-estimate pass | 90% cluster-bound floor |
| --- | --- | --- | --- |
| Boundary | 30 linguistic utterances, 10 items, 300 speech seconds | segment F1@1s ≥ .75; both edges@.5s ≥ .70; speech recall ≥ .80; precision ≥ .75 | .65; .60; .70; .65 |
| Source role | 30 decidable utterances, 10 items, at least 10 per primary class | non-child precision ≥ .85; recall ≥ .70; binary macro-F1 ≥ .75; coverage ≥ .80 | .75; .60; .65; .70 |
| Non-child text | 500 reference characters, 10 items, 180 speech seconds; 100 words when WER applies | CER ≤ .25; WER ≤ .40 when applicable; usable coverage ≥ .75 | CER upper bound ≤ .35; WER upper bound ≤ .50; coverage lower bound ≥ .65 |
| Coarse reference | 20 auditable non-child utterances, 8 items, 8 human-visible cases | exact ≥ .65; macro-F1 ≥ .60; visible precision ≥ .75; visible recall ≥ .60; coverage ≥ .70 | .55; .50; .65; .50; .60 |
| Noun/object candidate | 20 human-visible candidates over 8 items | precision ≥ .75; coverage ≥ .50 | .60; .35 |
| Verb/action candidate | 15 human-visible candidates over 6 items | precision ≥ .70; coverage ≥ .40 | .55; .25 |

The machine-readable protocol also freezes hard-failure thresholds. A point between a pass threshold and its hard-failure threshold, a missed uncertainty floor, or support recoverable from the reserve is `BORDERLINE`. A point beyond a hard-failure threshold with adequate human support is `HARD_FAIL`. Lack of support after the reserve is `NOT_ESTIMABLE`, not evidence that the corpus structurally lacks the phenomenon.

Uncertainty resamples whole selected items, not utterances, for 10,000 deterministic percentile-cluster bootstrap replicates and 90% intervals. Ratio metrics recompute pooled numerators and denominators within each resample. At least 80% of replicates must be defined. The audit also reports leave-one-item-out influence, and a pass is prohibited if one item supplies more than 30% of a gate's eligible support. These intervals are descriptive small-sample checks, not population or publication inference.

## Exactly one bounded escalation

The exact 900-second disjoint reserve is frozen at the same time as the primary sample. It can be activated once, and only when the primary result is borderline with no hard failure. Activation cannot depend on lexical richness, confidence, visual salience, apparent success, or a learner result.

The pooled primary-plus-reserve audit uses the unchanged metrics. There is no second expansion. After the reserve:

- every gate passing yields `CHILDLENS_MODEL_ASSISTED_FEASIBILITY_GO`;
- a remaining borderline or not-estimable gate yields `CHILDLENS_MODEL_ASSISTED_FEASIBILITY_REVISE`; and
- an adequately supported essential hard failure yields `CHILDLENS_MODEL_ASSISTED_FEASIBILITY_STOP`.

Before author completion, `CHILDLENS_AUTHOR_AUDIT_READY` is permitted only when full-pilot local pseudo-annotations, the frozen sample, and the blinded local app are complete. A technical or licensing issue with a bounded local correction yields `REVISE`; it is never repaired by weakening a scientific threshold.

## Small-sample limitations

Fifteen participant-distinct items and 15–30 speech minutes cannot establish population prevalence, full-release lexical support, inter-human reliability, or publication-grade annotation accuracy. Rare languages, roles, referential statuses, noun categories, or actions may be not estimable. Agreement is confounded with one author's judgment and error. The protocol therefore supports only the narrow question of whether fixed offline instruments are good enough for aggregate calibration and candidate generation under the stated boundaries.

The authoritative restricted sampler is implemented in [prepare_childlens_author_audit_packet_v1_3.py](../../scripts/prepare_childlens_author_audit_packet_v1_3.py). Generic freeze checks are implemented in [childlens_author_audit_protocol_v1_3.py](../../scripts/childlens_author_audit_protocol_v1_3.py). Their tests use synthetic intervals only and do not access a ChildLens runtime.
