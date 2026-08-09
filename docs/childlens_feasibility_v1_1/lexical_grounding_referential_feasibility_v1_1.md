# ChildLens v1.1 lexical-grounding and referential-annotation feasibility

Version: `childlens-lexical-referential-feasibility-v1.1.0`  
Audit date: 2026-07-21  
Scope: frozen pilot readiness and human-validation handoff only  
Scientific learner outcome authorized: **no**

## Updated finding

V1.1 establishes a content-blind 15-item preselection and an identifier-free procedural receipt. It does not establish lexical-grounding feasibility from corpus content. No selected video or audio was acquired, no actual language was identified, no utterance was transcribed or role-coded, no visible object/action candidate was judged, and no lexical support or split support was counted. Genuine human referential validation has not started.

The current gate dispositions are therefore:

- `G5_INPUT_LEXICON = UNRESOLVED`
- `G6_REFERENTIAL_ANNOTATION = UNRESOLVED`
- `G7_HELDOUT_EVALUATION = UNRESOLVED`
- `G8_BASELINE_CEILING = CONDITIONAL`

These are absence-of-evidence dispositions, not structural failure findings. They preserve the frozen v1 thresholds and do not overwrite the v1 `REVISE` history.

## V1.1 pilot evidence—and its limit

The preselection receipt records 15 selected items from 15 distinct participant groups, with a maximum of one selected item per participant. Selection used only coarse activity, speech-presence, an annotation-derived duration proxy tertile, and location when available. It was blind to lexical text, ASR output, audio content, visual content, apparent groundability, and learner results.

The human-validation skeleton has SHA-256 `ae8503fc5c21fc6df0b08c763202eab074b2526f83ef847d2faca3ae4eb217a5`. It contains null review fields and no media. Its digest binds the assignment skeleton, not ChildLens language, timing, lexical recurrence, referent prevalence, agreement, or quality.

Exact remote video bytes remain unresolved, the canonical final restricted manifest is incomplete, and acquisition has not started. The fixed 15 items cannot be substituted because a different item seems richer, clearer, more visible, easier to transcribe, or more likely to pass. Exact-byte/resource failure may contract the sample only within the frozen 12–18 range and only under the already allowed access/size/empty-stratum rules, before content inspection.

## Frozen unit and annotation window

The unit is a human-validated `NON_CHILD` utterance with an accepted source-language transcript and onset/offset, paired with the fixed ±5-second RGB context around utterance onset. `CHILD`, `OVERLAP`, `UNCERTAIN`, and `NONSPEECH` records are retained for role and missingness accounting but cannot silently count as non-child input.

Annotators first decide whether the window is technically usable, then apply the fixed six-state utterance ontology:

1. `UNUSABLE`
2. `UNDECIDABLE`
3. `IRRELEVANT`
4. `NULL_NOT_VISIBLE`
5. `VISIBLE_SINGLE`
6. `VISIBLE_MULTIPLE`

Null, ambiguous, undecidable, and unusable cases remain in their applicable denominators and reports. No annotator or adjudicator may force one target from a genuinely ambiguous set.

An eligible noun/object mention denotes a directly visible, discriminable object category. An eligible verb/action mention denotes a directly visible dynamic action with annotatable temporal boundaries. Proper names, pronouns, grammatical function words, purely mental/state predicates, and off-camera-only evidence are ineligible for the primary endpoint. Mixed utterances retain per-mention statuses so an on-camera noun does not make an off-camera verb visible.

## Referential timing and ambiguity

The fixed signed lag is `utterance onset minus nearest adjudicated visible candidate boundary`.

- Action candidates use the nearest adjudicated action onset or offset; onset wins an exact tie.
- Object candidates use the nearest continuous-visibility onset or offset; onset wins an exact tie.
- Left- and right-censored candidates are kept as censored categories rather than treated as exact lags.
- `VISIBLE_MULTIPLE` retains every plausible candidate and an utterance-level minimum absolute lag; it does not manufacture a unique target.
- Nonvisible, irrelevant, undecidable, and unusable cases receive no imputed lag.

Exact internal times remain restricted. Reportable lag bins remain −5 to −2 seconds, −2 to −0.5, −0.5 to 0.5, 0.5 to 2, and 2 to 5, with censoring separate and small-cell suppression applied.

## Genuine human coding and reliability

At least 20% of referential items must be independently double-coded, selected by release-bound hash order and stratified by participant and fixed speech/activity metadata. The set is fixed before annotation. Annotators remain blind to peer labels, lexical-frequency tables, split assignment, machine predictions, model design, and learner results. Codex cannot serve as a referential annotator or adjudicator.

Before adjudication, report:

- nominal Krippendorff alpha or a prespecified equivalent on the six statuses, with point estimate at least `0.67` for `G6`;
- participant-cluster uncertainty;
- agreement on visible versus other status;
- noun/object versus verb/action mention-family agreement;
- candidate-count-band agreement; and
- matched uncensored boundary agreement.

At least 80% of independently matched, uncensored boundaries must differ by no more than 1 second for the strong-alignment ceiling to be operationally usable. This is a `G8` construction requirement, not a substitute for the `G6` alpha threshold.

One bounded remediation cycle is allowed: clarify wording with non-study examples, retrain, and score a disjoint preselected block without changing the ontology, ±5-second window, lag anchor, or thresholds. Adjudication never overwrites pre-adjudication reliability.

## Corpus-local lexical feasibility thresholds

Language-specific normalization must be frozen after human language identification and before frequency/support tables are examined. No external vocabulary, translation list, BabyView/AEA artifact, pretrained tokenizer, or public benchmark lexicon may contribute.

Each prototype-eligible lexical type must satisfy all of the following in both the participant-disjoint primary split and session-disjoint sensitivity split:

1. At least 10 validated non-child training occurrences across at least 3 training participant groups.
2. At least 5 training occurrences with an annotatable visible mention-level link across at least 3 training participant groups.
3. At least 3 `VISIBLE_SINGLE` training links spanning at least 2 training participant groups.
4. At least one validated visible-link occurrence in each of at least 3 held-out participant groups.
5. No supporting occurrence is `UNCERTAIN`, uncorrected `OVERLAP`, `UNDECIDABLE`, or `UNUSABLE`.

`G5` passes only with at least 8 noun/object types and 6 verb/action types meeting every condition. The current packet contains no transcripts or lexical counts, so none of these conditions has been tested. Vocabulary strings should remain restricted by default; report only cell-suppressed type and support aggregates if terms permit.

## Leakage-resistant held-out construction

Before lexical counts are examined, all records sharing an internal participant key must be assigned indivisibly to the primary train/development/test split using metadata-only deterministic balancing, targeting 70%/15%/15% of usable annotated duration and at least 3 participant groups in both development and test. The sensitivity split similarly keeps every session indivisible. No participant fragment may cross the primary split and no session may cross either split.

The 15-item feasibility pilot is too small to stand in for the later full support audit. Its participant-distinct selection tests whether the annotation protocol can be executed without content-driven sampling; it cannot by itself prove that the full release supports the required held-out lexical inventory. `G7` passes only after the stable grouping, frozen split manifests, and minimum 8 noun/6 verb inventories all survive both split constructions.

## Weak baseline and strong-alignment ceiling

No control manifest or learner is built in this task. A later construction proof must generate both controls from one canonical ChildLens-derived inventory with identical raw RGB/text membership, multiplicity, split, tokenizer, initialization, example order, update count, optimizer, and compute. The only permitted difference is the frozen hidden alignment-target/readout transformation.

The weak baseline uses natural utterance-centered chronology and receives no referential status, candidate identity, boundary, detector output, or side cue. The strong ceiling uses adjudicated hidden annotations only to define its frozen alignment target within the same frame sequence. Null and ambiguous cases are retained under identical inclusion rules. Referential labels and exact timing are absent from every evaluation model call.

External instrument features, embeddings, labels, tokenizers, scores, or vocabulary never enter either control. IMU, contact/touch, proprioception, and motor streams remain simulator-defined counterfactual modalities; nothing in referential coding makes them ChildLens-measured.

`G8` is conditional because the equal-exposure control construction and no-training checks are frozen, but it cannot pass until a no-training validator proves equality of the canonical inputs and schedules, minimum visible-single lexical support, 80%-within-1-second boundary reliability, retention of null/ambiguity, and complete separation of hidden targets.

## Prohibited adaptations

The following would invalidate v1.1 feasibility evidence:

- replacing or expanding selected media because of words, speech density, visual salience, referent yield, agreement, or model performance;
- assuming a language, translating in an external vocabulary, or using another child corpus;
- treating ASR, diarization, a detector, or a vision-language system as annotation truth;
- dropping null, ambiguous, uncertain, undecidable, or unusable items to improve coverage;
- moving participant/session groups after lexical counts are visible;
- changing the ontology, window, lag anchor, support thresholds, or endpoint inventory in response to a learner result; or
- training, probing, or comparing any scientific learner or causal arm during this feasibility task.

## Gate evidence still required

After exact-byte acquisition admission, qualified humans must establish source language, timing, source-language text, speaker role, visible/null/ambiguous status, candidate boundaries, and pre-adjudication reliability. Only then can cell-suppressed aggregates test:

- recurring validated non-child lexical material (`G5`);
- at least 15% visible-candidate coverage and status alpha at least 0.67 (`G6`);
- participant/session-disjoint exposure and held-out support (`G7`); and
- equal-input baseline/ceiling construction with usable boundary reliability (`G8`).

Until that evidence exists, a ChildLens-only lexical prototype is plausible but empirically unproven. The exact next task is the bounded human-validation run described in [human_validation_packet_handoff.md](human_validation_packet_handoff.md), after the exact-byte/resource and quarantine gates pass. No learner training follows automatically from that handoff.
