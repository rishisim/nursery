# ChildLens v1.2 operational human-validation codebook

Version: `childlens-human-validation-codebook-v1.2.0`  
Applies to: the authenticated 15-item frozen v1.1 sample only  
Machine proposals visible to reference coders: **none**

## Training and qualification

Each human completes orientation with non-study examples before receiving a pass. The training record, language competence, start/end time, and any clarification stay in quarantine. Study items cannot be used as training examples. A coder may begin only when they can:

1. distinguish audible speech from vocalization and environmental sound;
2. transcribe the established source language without translating;
3. apply all five speaker-role values to non-study examples;
4. apply all six referential values without forcing a referent; and
5. mark relative candidate boundaries within a fixed ten-second window.

The coordinator assigns each human exactly one pre-generated pass token per relevant A/B pair. A person cannot act as both A and B. The adjudicator cannot coach a coder after seeing the peer record. One remediation cycle may clarify written rules with new non-study examples; it cannot change the ontology, window, sample, threshold, or prior locked records.

## Language and audio integrity

Listen broadly enough to judge the item, without reading a machine proposal.

- Use a lowercase ISO 639 code only when the coder is competent in that language and the evidence is clear.
- Use `MIXED_OR_CODE_SWITCHED` when intelligible speech materially alternates between languages within the item. A borrowed proper name alone is not code switching.
- Use `UNDECIDABLE` when the speech is too sparse, masked, or outside the coder’s competence. Do not infer language from geography, metadata, a person’s appearance, or release documentation.
- `USABLE` means the speech-bearing intervals required for the pilot are audible and technically decodable.
- `PARTIAL` means some intervals are corrupted or masked but a meaningful subset remains codable.
- `UNUSABLE` means technical or acoustic failure prevents the required judgments.
- `UNDECIDABLE` means the coder cannot separate a content problem from a technical problem.

Mark speech present only for linguistic vocal activity. Mark overlap when two or more people speak simultaneously for a nontrivial interval. The adjudicator resolves language, audio integrity, speech presence, and overlap while preserving both independent records.

## Utterance segmentation and text

An utterance is one continuous contribution by one speaker role. Start at the first audible linguistic segment and end at the final audible linguistic segment. Keep natural pauses within a syntactically or prosodically continuous contribution; split at a clear turn change, a sustained pause that begins a new contribution, or a role change. Do not extend a segment merely to match an annotation window.

Record times to the nearest discernible boundary. Onset must be nonnegative, offset must follow onset, and both must remain within the item. For overlap, include the interval in which voices overlap and choose `OVERLAP`; do not assign the louder voice. For unintelligible material, use the project’s explicit unintelligible marker instead of guessing. Preserve audible filled pauses and repetitions when they are part of the source-language speech. Do not translate, standardize dialect, supply omitted words, or copy a machine proposal.

`NONSPEECH` is retained for false speech candidates or intervals containing only environmental sound, music, breath, laughter without linguistic content, or other nonlinguistic vocalization. It does not count toward the 300-utterance stopping total or 30 speech minutes.

## Speaker-role decision tree

Apply the first rule that fits:

1. No linguistic speech: `NONSPEECH`.
2. Two or more simultaneous linguistic speakers: `OVERLAP`.
3. The enrolled/vest-wearing child is the audible speaker: `CHILD`.
4. A different person is the audible speaker: `NON_CHILD`.
5. Speech is present but role cannot be determined reliably: `UNCERTAIN`.

Age inferred from voice alone is insufficient when identity cannot be established from the local context. A nearby child who is not the enrolled child is `NON_CHILD`. Do not force `CHILD` or `NON_CHILD` through uncertainty.

All timing/text/role items are independently double-coded. After both passes close an item, the adjudicator matches corresponding segments, retains unmatched insertions/deletions, and locks accepted records. An item-level completion lock removes it from the queue so later items cannot starve.

The controller stops acceptance after the first non-`NONSPEECH` record that reaches either 300 accepted utterances or 30 accepted speech minutes. No further timing record can be accepted before the referential inventory is frozen.

## Referential unit and fixed window

Only an adjudicated `NON_CHILD` utterance is eligible. The coder sees the accepted source-language text and exactly five seconds before through five seconds after utterance onset. Evidence outside that window cannot be used.

First decide technical usability, then relevance, then visibility and ambiguity:

1. `UNUSABLE`: decoding, severe occlusion, missing frames, or another technical failure prevents a visual judgment.
2. `UNDECIDABLE`: the window is technically usable, but the utterance or scene is too uncertain to decide relevance/visibility.
3. `IRRELEVANT`: the utterance has no eligible concrete noun/object or dynamic verb/action mention for this pilot.
4. `NULL_NOT_VISIBLE`: an eligible mention is understood, but no corresponding candidate is visible in the fixed window.
5. `VISIBLE_SINGLE`: exactly one plausible visible candidate satisfies the mention.
6. `VISIBLE_MULTIPLE`: two or more plausible visible candidates remain; never choose one to simplify the record.

The mention family is:

- `NOUN_OBJECT` for directly visible discriminable object-category mentions;
- `VERB_ACTION` for directly visible dynamic actions with a temporal extent;
- `BOTH` when at least one eligible mention of each family occurs; or
- `NEITHER` when no eligible mention exists.

Proper names, pronouns, function words, purely mental/state predicates, and off-camera-only evidence are ineligible for the primary endpoint. A mixed utterance retains the actual mention-level ambiguity; one visible noun cannot make an off-camera action visible.

Candidate bands are `ZERO`, `ONE`, `TWO`, `THREE_PLUS`, or `UNKNOWN`. They must be consistent with the six-state judgment. Visible boundaries are relative to utterance onset and must stay within −5 to +5 seconds. Mark censoring when the candidate begins before or ends after the window. Nonvisible statuses cannot receive a boundary.

## Double coding and blinding

Pass A codes every eligible utterance. Pass B receives at least 20% within each opaque metadata stratum. The order is a deterministic hash of the frozen selection binding, the authenticated opaque item key, adjudicated timing boundaries, and the stable A/B source-segment display pair. The source-pair term prevents identical-boundary collisions without using transcript content. It uses no transcript string, lexical frequency, referential label, visual salience, machine result, or apparent grounding quality. Once frozen, no utterance can be added or substituted.

Both coders remain blind to peer records, split assignment, lexical tables, machine hypotheses, model design, and learner outcomes. The local app implements the fully human-hidden reference condition only. Any future proposal-visible production correction requires a separate approved workflow and cannot overwrite these references.

## Pre-adjudication reliability and completion

Compute reliability before adjudication from locked A/B records:

- exact language agreement;
- matched timing coverage, unmatched A/B counts, and the fraction of all accepted timing records with both independently supplied boundaries within 500 ms (unmatched records count as failures);
- speaker-role exact agreement, nominal alpha, and four-class macro-F1 over `NON_CHILD`, `CHILD`, `OVERLAP`, and `UNCERTAIN`;
- paired source-text Unicode-codepoint error rate as a language-neutral preflight, with a language-specific token/grapheme metric frozen later if required;
- six-status referential exact agreement and nominal Krippendorff alpha or equivalent; and
- fraction of matched uncensored candidate boundaries with both edges within one second.

The frozen reliability thresholds are at least 0.70 for both timing edges within 500 ms, 0.80 four-class role macro-F1, 0.67 six-status referential alpha, and 0.80 matched uncensored boundaries within one second. Report classwise errors, missingness, disagreement counts, and adjudication fraction in the restricted analysis. Repository output receives only permitted cell-suppressed aggregates.

If a threshold fails, one prespecified remediation cycle may use new non-study examples and a disjoint preselected block. Original records remain immutable and continue to be reported. The adjudicator resolves only after independent locking and cannot retroactively improve pre-adjudication reliability.

## Fail-closed rules

Stop and notify the coordinator if the app shows a source filename/path/identifier, a peer label before both locks, a window outside ±5 seconds, a machine proposal, a mutable locked record, or a task not in the assigned pass. Do not copy text, screenshots, audio, frames, times, or labels out of the quarantine. Do not use a general browser profile; the fixed launcher opens a dedicated browser profile and cache under the restricted root and authenticates the session with an ephemeral launch token.
