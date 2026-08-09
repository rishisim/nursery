# ChildLens v1.1 annotation-schema and modality audit

Version: `childlens-annotation-structural-audit-v1.1.0`  
Date: 2026-07-21  
Scope: aggregate structural evidence only; no media, transcript, or lexical inspection

## Result

The locally summarized annotation representation is structurally useful for metadata stratification and leakage-resistant grouping, but it does not supply the lexical or speaker-role supervision required by the grounding pilot. Across 192 extracted annotation files, the aggregate parser counted 27,167 annotation rows. A location key was present in 191 files, and all 192 annotation files contained at least one speech/talk/speak/vocal-like string value. This is file-level coverage: it must not be interpreted as 192 utterances, speakers, or speech intervals.

The participant table has 192 rows, 58 participant groups, 141 participant-plus-recording groups, and complete participant-plus-recording grouping fields for all 192 rows. That supports content-blind participant/session grouping in schema. It does not establish that a held-out-child/session split retains enough recurring nouns and actions; that is a later lexical pilot gate.

No lexical text, word/token field, utterance transcript, speaker-role field, recorded-language field, or age field was found in the aggregate schema receipt. ASR text and child-versus-other speech roles therefore remain measurement tasks requiring fixed instruments plus human validation/correction. No language may be assumed from the repository location, paper language, labels, or user interface.

## Structural inventory

| Structural fact | Aggregate evidence | Feasibility implication |
| --- | --- | --- |
| Annotation files | 192 extracted files | canonical-stem audit linked all 192 annotation/media references structurally; byte-checksum linkage remains unproven |
| Annotation rows | 27,167 | sufficient volume for coarse structural sampling; not lexical examples |
| Files with location key | 191 | location can stratify most files; one-file missingness must be retained explicitly |
| Files with a speech-like string value | 192 | file-level speech-related coverage; not an utterance or interval count |
| Participant-table rows | 192 | candidate row-level join surface |
| Participant groups | 58 | participant-disjoint grouping is structurally available |
| Participant-plus-recording groups | 141 | finer session/date grouping is structurally available |
| Complete grouping rows | 192 of 192 | no aggregate grouping-field missingness |
| Lexical/transcript field | absent | timed lexical grounding cannot use official annotation as text truth |
| Speaker-role field | absent | child/non-child input must be separately measured and human validated |
| Language field | absent | actual language(s) must be established from authorized audio by humans |
| Age field | absent | age cannot be a release-local pilot stratum from this representation |

A restricted canonical-stem audit validated all 192 annotation/media references and all 192 participant-table joins. This is sufficient for the operational structural linkage used in metadata-only preselection, while exact media bytes and checksum-level linkage remain unproven. The final restricted manifest must preserve missingness and export only a digest plus cell-suppressed aggregates.

## Annotation coverage and missingness

The 27,167 rows are annotation records, not necessarily mutually exclusive time intervals. They cannot be summed as annotated duration without an interval-union calculation inside quarantine. Likewise, file-level presence of a speech-like string value is not a timed lexical transcript and does not identify the speaker. Null, overlapping, ambiguous, or non-speech cases must remain in the later validation population rather than being filtered for apparent lexical richness.

The observed location-key coverage is `191/192`. The one missing-key file must be represented as a fixed missing-metadata stratum; it cannot be silently discarded or imputed after content inspection. The aggregate receipt does not establish activity-key completeness, interval timing validity, audio-channel integrity, or annotation/media duration consistency.

## Modalities and learner boundary

The release inventory establishes egocentric video-container objects and an annotation representation. The existing ChildLens audit describes synchronized video/audio as the intended raw empirical modalities, but this aggregate v1.1 structural pass did not decode containers or test audio usability. Accordingly:

- visual media are release-inventory evidence, not a validated referential signal;
- audio presence/usability, language, utterance timing, and speaker role remain pilot measurements;
- official annotations provide coarse structural strata only, not lexical gold or referent gold;
- absent IMU, touch/contact, proprioception, and motor streams remain simulator-defined counterfactual modalities, never ChildLens-measured;
- fixed pretrained annotation instruments may create quarantined proposals only; their features, embeddings, weights, tokenizers, vocabularies, scores, and unvalidated hypotheses cannot enter learner data;
- no learner was trained and no acquisition effect was inspected.

## Grouping and held-out evaluation

Complete participant-plus-recording grouping fields satisfy only the structural half of leakage control. A future restricted manifest can freeze participant/session-disjoint partitions before text inspection. However, G7 cannot pass from this receipt: exposure recurrence, noun/object support, verb/action support, and test-train lexical overlap have not been measured after grouping.

The participant total of 58 is specific to this accessible table. It must not be replaced with another publication-level participant count. The 141 participant-plus-recording groups are operational group units, not guaranteed independent sessions for every downstream purpose until the restricted join and date/session definition validate.

## What the frozen pilot must still measure

Without changing the v1 thresholds, the restricted 12–18-video pilot must establish:

1. decodable audio and usable speech timing;
2. actual language(s), determined from authorized audio rather than metadata assumptions;
3. human-corrected text and five-way speaker roles, including non-child input;
4. visible object/action candidates while retaining null and ambiguous cases;
5. recurring corpus-grounded noun/object and verb/action candidates after participant/session separation; and
6. inter-annotator reliability and timing error under the frozen protocol.

Until those measurements exist, the aggregate annotation volume cannot be interpreted as lexical-grounding feasibility evidence.

## Digests and preselection limitation

- immutable public snapshot commit: `4856662653b2fa183e53268d088cffba02a33443`
- reportable release-binding receipt SHA-256: `2aa1dcf53fc5592965019afe5df97b51ccc2fe05e70acb2d51aa396c73843805`
- frozen v1 pilot protocol SHA-256: `63be49d470c79e94f26240e73b314cdc42f2a6e6a57e96c897340d856f561ecb`
- canonical restricted manifest: `PENDING_EXACT_REMOTE_BYTES_AND_LOCAL_CHECKSUMS`
- placeholder manifest SHA-256: `cc2b669e811ff78a045565be031537d5a93bfece40f27de3482768350adc1deb`
- content-blind preselection: `FROZEN_15_ITEMS_15_DISTINCT_PARTICIPANT_GROUPS`
- restricted preselection input SHA-256: `81282e6a4d8178b1559a7473d22c70a36e44450467b6b5e2ff70ad9b7e9f049a`
- restricted download-plan SHA-256: `bb39c32139310e30ac15095a15b3d4b95508ab217694f591845aac38e00364b8`

The placeholder sizes are not scientific evidence or transfer admission. No media content or lexical signal was inspected, and the preselection cannot be changed in response to later yield.

This report contains no restricted row values or identifiers and makes no GO/REVISE/STOP decision.
