# ChildLens lexical-grounding and referential-annotation feasibility protocol

Version: `childlens-lexical-referential-feasibility-v1.0.0`  
Stage: feasibility/governance only  
Empirical corpus boundary: currently accessible ChildLens Keeper release only  
Scientific learner training or acquisition-effect execution authorized: **no**  
Pilot status in this artifact: **not run; no media inspected**  

## Scope and evidential status

This document defines how to determine whether ChildLens can support a bounded lexical-grounding prototype without selecting examples or changing criteria in response to apparent grounding quality. It does not claim that the release passes any feasibility gate, report a pilot statistic, select vocabulary items, train a learner, or issue a terminal decision.

The protocol is subordinate to the frozen [feasibility rubric](./frozen_feasibility_rubric_v1.json), [pilot protocol](./frozen_pilot_protocol_v1.json), and [scientific data-boundary contract](../child_only_prototype_v1/scientific_data_boundary_contract_v1.md). Measurement definitions come from the [ChildLens measurement specification](../child_only_prototype_v1/calibration/childlens_measurement_spec_v1.json). The local EgoBabyVLM README was consulted only for public method interfaces—multi-frame/utterance manifests, scratch language and CLIP-style baselines, and feature-extractor separation. No BabyView data, empirical values, vocabulary, tokenizer, weights, checkpoints, or results are used here.

This protocol addresses gates `G5_INPUT_LEXICON`, `G6_REFERENTIAL_ANNOTATION`, `G7_HELDOUT_EVALUATION`, and `G8_BASELINE_CEILING`. It depends on, but cannot establish, release/terms, grouping, audio, and transcript-role gates `G1`–`G4`.

## Binding prerequisites

No referential content inspection may begin until all of the following are true:

1. `G1_TERMS` is `PASS`, including permission for local derived transcripts and referential annotations in a restricted workspace.
2. The exact accessible release is pinned by its release manifest digest.
3. Participant and session grouping keys are available internally and can be used without export.
4. The content-blind sample has been selected exactly under `childlens-lexical-feasibility-pilot-v1.0.0`: target 15 videos, allowable range 12–18, metadata-stratified allocation, and release/protocol-bound hash ordering.
5. Selected media, human-corrected transcripts, timestamps, links, and internal grouping keys remain in the restricted namespace. None may be copied into this report or the machine-readable public/report artifact.
6. Language identification and the human-corrected speaker-role pipeline are sufficiently established to distinguish `NON_CHILD` speech from `CHILD`, `OVERLAP`, and `UNCERTAIN` speech. Raw ASR is never treated as gold.

If any prerequisite is missing, the relevant lexical gate remains `UNRESOLVED`; lack of permission is not evidence that the scientific construct fails.

## Unit of analysis and eligibility

The primary annotation unit is one human-validated utterance whose onset, offset, speaker role, and transcript are retained only in the restricted workspace. The referential context is the usable video interval from 5 seconds before to 5 seconds after utterance onset, clipped to the episode bounds. Clipping and unusable intervals are recorded explicitly. An utterance is eligible for the primary input analysis only when its speaker role is `NON_CHILD` and its timing and transcript are valid after human correction.

- `CHILD` speech is measured separately and does not count toward the non-child input lexicon.
- `OVERLAP` counts only if the non-child channel can be human-separated and recoded as a valid `NON_CHILD` segment before referential annotation.
- `UNCERTAIN` speech never counts toward lexical or grounding support.
- `NONSPEECH` is outside the utterance denominator.
- Technical failure is `UNUSABLE`, not a null referent.

Every valid non-child utterance is retained in the audit denominator, including utterances with no visible candidate, irrelevant speech, ambiguity, or undecidable content. Lexical support and visual-link support are reported as different quantities; a word occurrence is never silently promoted to a grounded occurrence.

## Restricted annotation record

The restricted record uses opaque internal keys and contains no learner-visible oracle fields. It minimally records:

- utterance validity, human speaker role, language routing, and timing-censoring flags;
- zero or more corpus-local lexical mentions, each with human-adjudicated lemma/construction key, part-of-speech family, and endpoint eligibility status;
- the utterance-level referent status;
- zero or more candidate event configurations and their object/action roles;
- candidate visibility interval, event interval, boundary type, occlusion/uncertainty flags, and candidate plausibility;
- per-mention link status, so a visible noun and a nonvisible verb in the same utterance are not conflated;
- double-code block, annotator receipt, adjudication state, and versioned codebook digest.

Exact text, lexical strings, frames, paths, timestamps, participant/session keys, candidate tracks, and per-item annotations never enter report artifacts. Hidden annotations must remain physically separate from model-visible records, consistent with the existing episode-schema separation.

## Referential ontology

### Candidate definition

A visible candidate is a directly observable object or action configuration in the fixed context window that could plausibly ground at least one content-bearing mention in the utterance. “Plausible” is semantic and perceptual, not causal: the annotator need not infer the speaker’s true intention. A candidate cannot be inferred solely from audio, off-camera context, cultural knowledge, an earlier/later scene outside the window, or a detector label.

Candidate configurations may contain:

- one visible object entity for a concrete noun mention;
- one temporally bounded visible action for a dynamic verb mention;
- an action plus its participant object(s) when the utterance expresses both;
- multiple alternative configurations when the scene or language does not support a unique link.

Separate clear arguments within one action configuration do not by themselves create ambiguity. For example, an action with an unambiguous actor and object can remain one configuration. `VISIBLE_MULTIPLE` is reserved for two or more plausible alternative link configurations, not merely a sentence containing several unambiguous content words.

Object visibility requires enough directly visible image evidence for a human to distinguish a stable entity candidate at the intended ontology level. Partial visibility may qualify and is flagged; an entirely occluded or merely inferred object does not. Action visibility requires a directly observable state change or motion interval. Static possession, mental state, intention, or an acoustically inferred action does not qualify as a visible action.

### Utterance-level status decision order

The frozen status set is applied in this order so the categories are mutually exclusive:

1. `UNUSABLE`: the fixed window or speech record is technically unusable for the judgment.
2. `UNDECIDABLE`: the media are usable, but transcript meaning, occlusion, or candidate identity prevents a defensible classification after applying the codebook.
3. `IRRELEVANT`: there is no concrete object/action-grounding proposition to annotate, such as purely social, discourse-management, or grammatical material.
4. `NULL_NOT_VISIBLE`: there is an eligible concrete object/action mention but no plausible candidate is visible within the fixed window.
5. `VISIBLE_SINGLE`: at least one eligible mention has one plausible visible candidate configuration and no eligible mention has alternative visible configurations. Mention-level nulls are retained separately.
6. `VISIBLE_MULTIPLE`: at least one eligible mention has two or more plausible alternative visible candidate configurations.

Mixed utterances are not simplified away. An utterance can have an utterance-level visible status while one of its lexical mentions is `NULL_NOT_VISIBLE`; endpoint support is always computed from mention-level links.

### Candidate count and ambiguity

For each eligible mention, annotators record the number of plausible candidate configurations, capped for coding at `5_PLUS`. Exported summaries use only cell-suppressed count bands `0`, `1`, `2`, `3_4`, and `5_PLUS`. The primary ambiguity statistic is the proportion of usable eligible utterances with `VISIBLE_MULTIPLE`. Candidate multiplicity, lexical multiplicity, and technical uncertainty are reported separately.

No adjudicator selects a “true” referent from a genuinely ambiguous set. All plausible alternatives remain in the hidden annotation; a later learner can receive a bag/multiple-positive target only if that rule is frozen before training.

## Event boundaries and utterance–event lag

The signed lag is defined by the existing measurement specification: `utterance onset minus nearest adjudicated visible candidate boundary`.

- For an action, candidate boundaries are the adjudicated action onset and offset; the boundary with the smallest absolute distance to utterance onset is selected, with onset winning an exact tie.
- For a pure object mention, boundaries are the onset and offset of a continuous visibility segment; the nearest boundary is selected, with visibility onset winning an exact tie.
- If the object/action is already active or visible at the left window edge, the onset is left-censored. If it continues through the right edge, the offset is right-censored. A censored boundary is not treated as an exact lag.
- `VISIBLE_MULTIPLE` yields one lag record per plausible utterance–candidate pair plus an utterance-level minimum absolute lag. It never yields an invented unique target.
- `NULL_NOT_VISIBLE`, `IRRELEVANT`, `UNDECIDABLE`, and `UNUSABLE` have no event lag; they are not imputed.

The restricted workspace may retain exact internal timing. Reports contain only coarse lag bins and distribution summaries with no exact timestamps, example text, or example frames. Bins, frozen before coding, are −5 to −2 seconds, −2 to −0.5, −0.5 to 0.5, 0.5 to 2, and 2 to 5. Left- and right-censored cases are separate categories and never placed into an exact lag bin.

## Annotation procedure and reliability

Annotators are trained on abstract examples or a preselected calibration block from the authorized ChildLens pilot. They remain blind to the other annotator’s labels, lexical frequency tables, split membership, model design decisions, and any learner result. Automated detector or vision-language suggestions are not shown during the independent pass. A fixed pretrained tool may assist only as a quarantined measurement instrument after the annotation-instrument amendment is approved; its predictions and embeddings never enter the learner.

All timing/speaker-role validation items follow the separate transcription protocol. At least 20% of referential items are double coded, selected by the same release-bound hash order and stratified by participant group and speech/activity stratum. The double-code set is fixed before annotation and cannot be enriched with difficult or easy items after inspection.

Agreement is scored before adjudication:

- nominal Krippendorff alpha on the six utterance statuses, with point estimate at least `0.67` for `G6`;
- participant-cluster bootstrap interval for that alpha;
- agreement on visible versus nonvisible/irrelevant/undecidable/unusable collapse;
- mention-family agreement (`NOUN_OBJECT`, `VERB_ACTION`, or neither);
- candidate-count-band agreement;
- for independently matched candidate boundaries, the proportion with absolute disagreement no greater than 1 second, reported with participant clustering.

The last four are diagnostics, not substitutes for the frozen `0.67` status-alpha gate. The lag/ceiling annotation is considered operationally usable only if at least 80% of independently matched, uncensored boundaries differ by no more than 1 second. This is a predeclared `G8` construction requirement, not an added condition on the already frozen `G6` gate: if it fails, `G6` may still satisfy its rubric, but the strong-alignment construction remains `CONDITIONAL`.

One bounded remediation cycle is allowed and must be declared before scoring the production reliability block: clarify wording and add non-study examples without changing the six statuses, ±5-second window, candidate definition, lag anchor, or thresholds; retrain annotators; then score a disjoint, preselected double-code block. A second ontology change or threshold change requires a new versioned feasibility protocol and yields `CONDITIONAL`, not a retroactive pass. Adjudication creates the production label but does not overwrite pre-adjudication reliability evidence.

## Corpus-local lexical eligibility

Language-specific normalization must be defined after actual language identification and before any frequency/support table is examined. It may include human-reviewed case, inflection, clitic, and multiword-expression rules. The scientific tokenizer is separate: it is trained from scratch on the final training partition only and may not inherit an external vocabulary or weights.

Candidate lexical types are limited to validated `NON_CHILD` input:

- noun/object candidates are common-noun lemmas or fixed corpus-local expressions denoting directly visible, discriminable object categories;
- verb/action candidates are dynamic verb lemmas or fixed corpus-local expressions denoting directly visible actions with annotatable temporal boundaries;
- proper names, pronouns, auxiliaries, copulas, modals, discourse markers, purely mental/state predicates, and labels whose only evidence is off-camera are ineligible for primary endpoints;
- code-switched material is eligible only under a frozen language-routing and normalization rule and must meet the same support thresholds without translating in vocabulary from an external corpus.

For an individual type to be **prototype-eligible**, all of these conditions must hold in both the participant-disjoint primary split and the session-disjoint sensitivity split:

1. At least 10 validated non-child training occurrences across at least 3 training participant groups.
2. At least 5 training occurrences with an annotatable visible mention-level link across at least 3 training participant groups.
3. At least 3 of those training links are `VISIBLE_SINGLE` and span at least 2 training participant groups, so strong alignment is not defined entirely by ambiguous bags.
4. At least one validated occurrence with an annotatable visible link in each of at least 3 held-out participant groups.
5. No occurrence used to satisfy support is `UNCERTAIN`, uncorrected `OVERLAP`, `UNDECIDABLE`, or `UNUSABLE`.

These are feasibility/support thresholds, not claims that ten occurrences are scientifically sufficient for learning. The later outcome protocol must freeze a train-only exposure threshold at least this strict, based on resource and precision planning, before training. It may raise the threshold; it may not lower it after an outcome is inspected.

`G5` can pass only if at least 8 noun/object and 6 verb/action types are prototype-eligible. More eligible types may be retained. If an endpoint inventory must be capped, it is selected before training by descending minimum support across the two splits, then by training participant-group coverage, then by visible-link count, with ties broken by a release/protocol-bound keyed hash. Token identity and any learner score are prohibited tie-breakers.

No token strings are exported unless the release terms explicitly permit a nonidentifying vocabulary export. Even when permitted, cells below 5 are suppressed and complementary cells are coarsened. Default reports include only the number of eligible types by endpoint family, support bands, coverage, and uncertainty.

## Leakage-resistant grouping and support audit

### Primary participant-disjoint split

All fragments sharing an internal participant key are an indivisible group. Before any transcript text or lexical counts are consulted, groups are assigned to train/development/test using a deterministic greedy balance over metadata-only annotated duration and available coarse activity, speech-presence, and location strata. Targets are 70%/15%/15% by usable annotated duration, with at least 3 participant groups in development and at least 3 in test. Ties use `SHA-256(release_manifest_digest || split_protocol_id || internal_participant_key)`.

No participant, session, or recording fragment may cross a primary split. If stable participant grouping is unavailable, `G7` fails rather than falling back to file-level splitting.

### Session-disjoint sensitivity split

All files/fragments sharing an internal session key are indivisible. Sessions are assigned 70%/15%/15% by the same metadata-only deterministic procedure and tie-break rule, replacing participant key with session key. Participant overlap across sensitivity splits is allowed and must be reported as an aggregate because the purpose is to test sensitivity to the stricter participant boundary. No session may cross a split. If session grouping is unavailable, `G7` cannot pass.

### Content-blind audit sequence

1. Freeze the two split manifests and their digests using only grouping and coarse metadata.
2. Apply human-corrected role and language filters.
3. Build training-only corpus normalization/token counts.
4. Determine training-supported candidate types without looking at development/test learner performance.
5. Evaluate the fixed candidates against development/test participant and visible-link support.
6. Freeze the eligible endpoint inventory and digest before any learner initialization or outcome run.

The support audit may inspect aggregate lexical counts only after split assignment. It may not reassign groups, add examples, alter normalization, or substitute types to improve a later model result. A failure caused by content-blind splitting is evidence about feasibility, not permission to optimize the split lexically.

Report only cell-suppressed aggregate matrices: number of qualifying noun/object and verb/action types, distribution of participant-group coverage, distribution of training occurrence/support bands, and whether both split constructions meet the thresholds. Internal lexical hashes are not exported because low-frequency hashes can be dictionary attacked.

## Strong-alignment ceiling and weak baseline

The audit does not instantiate or train either control. It establishes a construction proof over one canonical ChildLens-derived inventory.

Each learner record must contain the same opaque record key, human-corrected training-partition text, fixed ±5-second usable RGB window, frame/sample-validity mask, and record weight. The raw record multiset, transcript multiset, frame multiset, tokenizer, splits, model initialization, architecture, example order, update count, batch/padding shapes, optimizer, stopping rule, and compute budget are identical between the two control builds.

- **Weak VL baseline:** preserves the natural utterance-centered chronology and uses the frozen utterance-centered pooling/pairing rule. It receives no candidate identity, event boundary, referent status, detector output, or side cue. Null and ambiguous utterances remain in the canonical inventory under the same content-blind inclusion rule.
- **Strong-alignment ceiling:** uses the hidden adjudicated annotation only to choose the text-to-visual readout mask/target within the same fixed frame sequence. `VISIBLE_SINGLE` uses its candidate interval; `VISIBLE_MULTIPLE` uses the frozen multiple-positive bag; `NULL_NOT_VISIBLE` uses the explicit null target; `IRRELEVANT` has no lexical alignment target but retains identical MLM/visual-self-supervision exposure; `UNDECIDABLE` and `UNUSABLE` use the same predeclared validity masking in both builds. No additional crop, frame, text, or repeated update is allowed.

The only permitted control difference is the frozen alignment-target/readout transformation and its declared target mask. A manifest validator must prove equal record IDs, raw payload digests, per-objective eligibility masks other than the declared alignment target, example order, and compute receipt. The strong annotations are hidden-oracle/offline artifacts and cannot appear in primary evaluation model calls.

This construction is a ceiling, not a ChildLens-measured sensor condition and not evidence for the later synchronized-side hypothesis. IMU, contact, proprioception, and motor streams absent from ChildLens remain simulator-defined counterfactual modalities. They cannot be inferred from the referential annotations or calibrated with another corpus.

### `G8` construction-pass conditions

`G8_BASELINE_CEILING` is `PASS` at protocol level only if a validator can establish all of the following without training:

1. a single canonical inventory generates both control manifests;
2. exact equality of raw RGB/text record membership, multiplicity, split, and update schedule;
3. the only manifest difference is a named, frozen alignment transformation;
4. each prototype-eligible noun and verb has at least the visible-single support required above;
5. at least 80% of independently matched, uncensored candidate boundaries agree within 1 second;
6. null and ambiguous cases are retained under fixed rules;
7. strong targets and all referential annotations remain outside the weak learner and every evaluation model call;
8. no model result, cue effect, or endpoint score was inspected in constructing either manifest.

If equality requires adding/removing examples, borrowing an external vocabulary, or choosing alignment windows after seeing outcomes, `G8` fails.

## Held-out evaluation construction

The primary later evaluation remains vision/text only and corpus grounded. Endpoint words must be observed and supported in the ChildLens training partition; object/action instances, target objects, world assets, backgrounds, and render seeds are held out according to the existing evaluation specification. Trial labels and referential annotations remain in a physically separate offline builder.

- Noun/object trials use isolated corpus-local forms or training-attested minimal frames and score concept-macro balanced retrieval/minimal pairs on held-out object instances.
- Verb/action trials use corpus-local forms or training-attested minimal frames and score concept-macro balanced retrieval/minimal pairs on held-out action instances and target objects.
- Any grammatical carrier/template must be derived from the ChildLens training partition under the same one-corpus boundary; unsupported forms are not imported.
- Candidate types are fixed before learner outcomes. Equal trial counts per concept are sampled by a protocol-bound hash; no concept is dropped because it performs poorly.
- Side cues, candidate masks, event timing, participant/session metadata, and strong-alignment annotations are absent from model calls.

The participant-disjoint audit is evidence that the vocabulary is recurrent across children, not a license to expose held-out transcripts or identifiers to the learner. Machine-DevBench remains secondary and coverage-gated; its public interface does not authorize BabyView-derived vocabulary or trials in a ChildLens instance.

## Exact feasibility judgments for this workstream

No terminal study decision is assigned here. When authorized evidence exists, the coordinator should apply these status rules without changing them:

- `G5_INPUT_LEXICON = PASS` only when at least 8 noun/object and 6 verb/action types meet every prototype-eligibility condition using validated non-child input. It is `FAIL` when a complete authorized audit shows the minimum inventory cannot be met and no bounded ChildLens-only correction exists. Otherwise it is `CONDITIONAL` or `UNRESOLVED` according to the frozen rubric.
- `G6_REFERENTIAL_ANNOTATION = PASS` only when at least 15% of validated non-child utterances in usable windows are `VISIBLE_SINGLE` or `VISIBLE_MULTIPLE`, nominal pre-adjudication status alpha is at least 0.67, a bounded adjudication path exists, and the participant-cluster intervals and coverage are reported. Null, ambiguous, undecidable, and unusable cases must remain in denominators/categories. No lower confidence-bound threshold is added to the frozen 15% rule; the interval is reported for calibration and limitations.
- `G7_HELDOUT_EVALUATION = PASS` only when stable participant and session keys support the two predeclared disjoint splits and at least the minimum 8 noun/object and 6 verb/action fixed candidates remain prototype-eligible under both. Split manifests and endpoint inventory must be frozen before outcome training.
- `G8_BASELINE_CEILING = PASS` only when the eight construction conditions above validate on protocol/manifests without learner training.

Pilot sample size may change within 12–18 only for access, byte-size, or empty-stratum constraints identified before content inspection. It may not change because speech is sparse, vocabulary is weak, visibility is low, or agreement is disappointing. Annotation effort may cover every eligible utterance in already selected videos, but new content cannot be selected using transcript, visual, lexical, or learner information.

## Prohibited adaptations

The following invalidate this workstream’s feasibility evidence:

- selecting or replacing videos based on ASR text, lexical richness, visible-referent yield, salience, or apparent alignment;
- assuming English, translating external vocabulary into the corpus, or using BabyView/AEA vocabulary, tokenizers, measurements, weights, checkpoints, or results;
- treating raw ASR, diarization, a pretrained detector, or a vision-language model as annotation truth;
- deleting null, ambiguous, undecidable, or technically unusable cases from the applicable denominator without reporting coverage;
- moving participant/session groups after lexical counts are visible;
- choosing endpoint types or alignment windows using learner outcomes;
- passing measurement-model embeddings/features, referential labels, exact timing, or side cues to the primary evaluation model;
- representing simulator-only physical streams as measured by ChildLens.

## Required workstream outputs after an authorized pilot

Only nonidentifying, terms-permitted, cell-suppressed aggregates may be reported:

- usable utterance and video coverage with participant-cluster uncertainty;
- referent-status proportions and ambiguity bands;
- lag-bin summaries and boundary reliability, without exact times;
- pre-adjudication agreement and adjudication rate;
- counts of eligible noun/object and verb/action types, never raw vocabulary by default;
- pass/conditional/fail/unresolved evidence for `G5`–`G8` with source manifest/codebook/split digests;
- reasons for exclusion and missingness bands;
- an explicit statement that no learner or acquisition-effect comparison ran.

At the time of writing, all empirical result fields remain `NOT_RUN`. This artifact provides protocol evidence only.
