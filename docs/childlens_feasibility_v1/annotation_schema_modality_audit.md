# ChildLens annotation schema and modality audit

Version: `childlens-annotation-audit-v1.1.0`  
Audit date: 2026-07-21  
Scope: feasibility and governance only; no scientific learner training or acquisition-effect analysis  
Release-specific status: sanitized authenticated live-view counts/schema observed; immutable live revision, full manifest, and linkage coverage pending

## Finding

The public ChildLens record supports metadata-first sampling and shows that the corpus contains synchronized egocentric video and audio plus manually timed, exhaustive multi-label activity/location annotations. It does **not** establish a timed lexical transcript, word-to-referent labels, fine-grained noun/object or verb/action labels, or any physical side-sensor stream. The public speech labels are timed speech **bouts**, not utterance transcripts: audio labels separated by two seconds or less were merged into one event. They are therefore useful for stratification and coarse speaker-role validation, but they cannot directly supply the utterance boundaries, lexical content, or referential alignment required by the proposed grounding prototype.

The public paper also establishes that participant-disjoint splitting was possible for the authors' voice-type benchmark. A sanitized read-only Keeper receipt now confirms that the authenticated live view uses the public DOI library UUID and exposes file-to-anonymized-child and recording-date metadata fields, so participant-disjoint and date/session-disjoint construction are supported **in schema**. It does not establish complete linkage for every eligible file, an immutable revision of the live view, full annotation-schema coverage, recorded language, or lexical support after splitting. This report makes no terminal feasibility decision.

## Evidence boundary

This workstream used only:

- the ChildLens version-of-record paper and its public tables;
- the official Max Planck ChildLens project page and DOI pointer; and
- the official ChildLens code repository's public README/benchmark documentation at observed commit `4923ca1411e9622ec91367d03e23627ed70ef19a`; and
- a coordinator-supplied, sanitized receipt from read-only authenticated Keeper inspection on 2026-07-21. The receipt contained aggregate object counts and field names only, not values.

No video, audio, frame, transcript, participant/session identifier value, or exact media timestamp was inspected by this workstream or included in the sanitized receipt. No media was downloaded during the authenticated inspection. No AEA or BabyView empirical artifact was used. A temporary checkout of the public ChildLens code repository was used to read documentation; auxiliary data files were not opened, and the checkout was moved to Trash immediately after the documentation audit.

## Public-paper snapshot versus the accessible release

The public release identity found by the governance workstream is the tuple `(DOI 10.17617/4.fe, Keeper library c8ed0104-b793-4c35-817e-302afd4e036b, snapshot commit 4856662653b2fa183e53268d088cffba02a33443)`. This is a public repository/snapshot identifier, not a participant identifier. The DOI snapshot link requires login. DataCite records no semantic version and no rights entry, so the tuple pins a repository state but says nothing by itself about processing permission.

The Keeper DOI landing describes 109 total hours, 62 children, and 54 annotated hours, without file counts. The version-of-record paper was published on 2026-04-13 and reports a corpus-level snapshot of 354 video files, 108.58 hours, and 62 children aged 3–5. It reports that 192 videos—51%, approximately 55 hours—were annotated at the paper snapshot, and explicitly says annotation was ongoing and the repository would be updated regularly. The total-hour values are compatible after rounding; 54 versus approximately 55 annotated hours may reflect rounding, version description, or a coverage-definition difference and must be reconciled rather than collapsed. These values do not automatically describe the exact authenticated/live Keeper view accessible on 2026-07-21.

The sanitized authenticated receipt observed the same Keeper library UUID as the public DOI and counted 192 MP4 objects, 192 `final` JSON annotation objects, and 192 platform-format JSON annotation objects in the current UI. The two JSON families must be treated as candidate alternate serializations of the same annotations—not 384 independently annotated videos—until content hashes and one-to-one linkage are verified. The 192 MP4 count equals the paper's count of annotated videos and is therefore consistent with an annotated-subset package, but that is an inference, not a verified package definition. Relative to the paper's 354 total video files, the accessible UI exposes 162 fewer MP4 objects under these counting units.

The task input's earlier estimate of about 343 videos is not supported as the current MP4 count by the sanitized receipt. It remains an unverified description that may have used a different folder, asset class, version, or counting unit; it must not be mixed with the observed 192 MP4 count.

The paper's corpus totals and annotation totals must not be silently combined with release-specific counts. In particular:

- the current Keeper release must be pinned by a release/share identity and a canonical nonidentifying manifest;
- `video file`, `annotation file`, `annotated video`, and `usable linked video` must be counted separately;
- annotation coverage must be computed as the union of annotated time within each file, because concurrent multi-label intervals overlap and summing class durations double-counts time;
- paper-versus-release differences must be classified as version changes, exclusions, duplicate/alternate encodings, link failures, or genuinely missing assets; and
- participant and session counts must come from the release manifest, not be inherited from the paper.

The official project page calls the DOI target the "latest version," while the paper warns that updates are ongoing. The live library UUID matches the public DOI library, but the UI did not expose an immutable live revision/hash, so content-equivalence to the pinned DOI snapshot commit is not established. A canonical release manifest and digest remain required.

## Raw and derived modalities

| Modality or signal | Public evidence | Audit interpretation |
| --- | --- | --- |
| Egocentric RGB video | Child-worn vest camera; forward field of view; paper reports 1920×1080 at 30 fps and a 140° lens; live UI contains 192 MP4 objects | Reported raw empirical modality and release object class. Exact stream/container properties, corruption, duration, and linkage require metadata/pilot audit. The camera is vest/chest-mounted, not a measured head-pose stream. |
| Audio | The vest camera includes a microphone and the release description claims video plus audio | Reported raw empirical modality. No audio stream was decoded in this audit; exact presence per MP4, codec/channels/sample rate, and usable timing require metadata/pilot audit. |
| Still images | Preliminary release descriptions mention images, but this workstream did not inspect the release | Treat as release assets or previews, not an additional sensor modality; exact provenance must be recorded by the release manifest. |
| Location annotations | Five coarse location classes, timed by video association | Manual annotation modality suitable for sampling strata; not a positioning/GPS stream. |
| Activity/speech annotations | Fourteen exhaustive, concurrent interval classes plus social/person attributes; live UI contains 192 final JSON and 192 platform-format JSON objects | Manual interval annotations. The JSON families are presumed alternate serializations pending hashes/linkage. Coarse activities and speech bouts are not lexical or fine-grained action truth. |
| Voice type | Public paper derives key-child/other-child/adult/unspecified speech classes from activity plus person attributes for a benchmark; no dedicated lexical transcript/speaker-role field was observed in the inspected JSON prefix | Potential measurement target, not an official utterance-level speaker-role annotation. Overheard speech lacks age/gender information in the paper mapping. |
| Lexical transcript | Not reported publicly and no lexical transcript field was observed in the inspected release JSON prefix | Unavailable on inspected evidence, while full-schema absence remains unproven. Any transcript must be a permitted, quarantined derived annotation and human-validated; ASR output is never gold. |
| Object identity, action token, referent, gaze, hand/contact | Not reported as released annotation streams | Unavailable on public evidence. Visible candidates/null/ambiguity require a separately frozen manual annotation protocol. |
| IMU, touch/contact sensor, proprioception, motor command | Not reported as raw ChildLens streams | Must remain `UNAVAILABLE_UNLESS_RELEASE_MANIFEST_PROVES_OTHERWISE`. They may only be simulator-defined counterfactual modalities and can never be described as ChildLens-measured. |

An absence from the public paper is not by itself proof that a file does not exist in Keeper. Conversely, a hardware capability must never be treated as a released signal without a manifest entry and usable stream audit.

## Published annotation schema

The paper describes one or more annotations associated with each video file. The conceptual fields are:

| Conceptual field | Published meaning | Feasibility caution |
| --- | --- | --- |
| Media association | Annotation belongs to a video file | Exact foreign key and one-to-one/one-to-many linkage must be audited in the release without exporting the value. |
| Start and end | Temporal boundaries of each activity | Serialized unit, precision, time base, out-of-range behavior, and missing values are not established publicly. Exact values remain restricted. |
| Activity class | One of fourteen coarse classes | Multi-label: simultaneous activities are represented as separate annotations. |
| Location | Current location, one of five coarse classes | Whether location is stored on every activity row, a separate interval track, or only some records must be checked in the release. |
| Social engagement | Child engaged alone or with somebody else | Domain values and applicability by class must be checked. This is not speaker diarization. |
| Person age class | Age class for each person involved in an activity | Cardinality and missing/not-applicable encoding are not published. The VTC mapping uses at least child/adult distinctions for `other person talking`. |
| Person gender | Gender for each person involved in an activity | Domain values, multiplicity, missingness, and consent-sensitive handling require release audit. Do not use this attribute for scientific endpoints. |

The paper calls the scheme exhaustive and multi-label: when activities overlap, each activity gets its own annotation. For all audio-based classes, gaps of two seconds or less are merged into the same event and longer gaps split events. Thus `child talking`, `other person talking`, and `overheard speech` are coarse interval labels, not utterance rows.

The public code documentation says its benchmark preparation expects individual JSON annotation files and later creates combined and split files. That is evidence about the authors' benchmark interface, not proof that the currently accessible release uses the same serialization or preserves every conceptual field.

### Authenticated live-view schema receipt

The current UI exposed 192 MP4 objects and two annotation representations with 192 JSON objects each: a `final` representation and a platform-format representation. A representative visible prefix of one final JSON showed:

- top-level `video_name`/`videoName`, `duration`, and `annotations` fields;
- event `startTime`, `endTime`, `time`, `category_id`/`categoryId`, `eventId`, `type`, and `fields`; and
- contextual `Type of Location`, `Alone?`, and up to six repeated other-person age-group/gender slots.

This is a prefix-level schema observation, not a complete schema export. It establishes that timed events and contextual social/person attributes are serialized in the accessible release. It does not establish the time unit/base, optionality, category codebook, full field inventory, or cross-file consistency. No lexical transcript or dedicated utterance-level speaker-role field was observed in the prefix; absence from the uninspected remainder is not claimed.

The participant table schema includes a media-link field, anonymized child grouping, recording date, birthday/age derivation, gender, and free-text comments. Only schema names—not any values—entered this audit. The anonymized child grouping and recording date can support participant- and date/session-disjoint construction inside the restricted workspace, provided linkage completeness is verified. Birthday, recording date, and comments are sensitive; exact values must remain quarantined, comments should not enter learner/calibration paths, and only coarse derived age or cell-suppressed grouping summaries may leave the restricted namespace if terms permit.

## Label inventory

### Locations

- `livingroom`
- `playroom`
- `bathroom`
- `hallway`
- `other` (the public aggregate table renders this as `Other Room`)

The spelling/casing difference between prose and the aggregate table must be resolved against the release's literal codebook rather than normalized silently.

### Audio-based activity/speech presence

- `child talking`: the wearer talks to themself or someone else;
- `other person talking`: another person talks to the wearer;
- `overheard speech`: a conversation is audible but does not involve the wearer;
- `singing/humming`; and
- `listening to music/audiobook`.

The distinctions are valuable for a ChildLens-only language-input audit: `other person talking` is the closest published proxy for child-directed non-child input, while `overheard speech` is available but not necessarily directed to the wearer. Neither provides words. `Other person` may be a child or adult and must not be collapsed to adult speech without checking the person attribute and manual role validation.

### Visual activity

- `watching something`
- `drawing`
- `crafting things`
- `dancing`
- `playing with object`

### Multimodal activity

- `playing without object`
- `reading a book`
- `making music`
- `pretend play`

These are coarse activity categories. `Playing with object` does not identify the object, and the multimodal/visual labels do not identify a lexical verb, its argument, or a visible candidate boundary suitable for lexical evaluation.

There is a public categorization inconsistency: the paper prose lists `playing with object` as visual-based, while the public aggregate Table 4 places it under the `Multimodal` heading. The fourteen class labels themselves are stable across those presentations, but the higher-level modality family must be taken from the exact release codebook or recorded as audit-defined; it must not be silently chosen from one paper rendering.

## Coverage, imbalance, and missingness

The paper's published aggregate table covers 192 annotated videos, and the authenticated live UI exposes 192 MP4 objects plus two 192-object JSON families. That count agreement supports—but does not prove—the interpretation that the live view is an annotated subset. The paper reports pronounced imbalance across both activity instance counts and total class duration; its benchmark consequently used only three high-duration visual/multimodal classes. Sparse public categories are not repeated here as small numeric cells. Public or UI-level aggregates do not establish per-participant/session support or complete linkage.

Release-specific missingness must be reported at these distinct levels:

1. **Asset inventory:** expected versus accessible media, annotation, metadata, and participant rows.
2. **Link integrity:** orphan annotations, unannotated media, duplicate links, and media without a participant group.
3. **Temporal coverage:** usable media duration; union duration covered by any annotation; location coverage; and speech-label coverage. Do not sum overlapping class durations.
4. **Attribute coverage:** social-engagement, person-age, and person-gender present/missing/not-applicable rates by coarsened class family.
5. **Technical usability:** decode failures, absent/short audio tracks, duration disagreement, discontinuities, and timestamp-base mismatch.
6. **Scientific usability:** corrected utterance timing, language routing, speaker-role confidence, lexical recurrence, visible-candidate status, and annotation reliability.

All participant-linked or lexical cells below five must be suppressed or coarsened without complementary reconstruction. Counts published at the whole-corpus level may be recorded, but public-paper status does not make a release-specific small cell safe to export.

## Language evidence

The recorded language or languages are **not established** by the public sources or sanitized authenticated schema receipt reviewed. The paper says that families lived in a mid-sized German city and that native German speakers performed the manual activity labeling. Those facts are contextual and suggest a German-compatible annotation workforce; they do not prove that all, most, or any particular recording is German, nor do they rule out multilingual households, dialects, media speech, or code-switching.

The audit must therefore keep language as `UNKNOWN_PENDING_RELEASE_OR_PERMITTED_PILOT`. ASR selection, tokenizer construction, vocabulary thresholds, and interpretation cannot assume English or German. Language must be identified from explicit release documentation if available; otherwise it needs a terms-permitted, quarantined manual/automatic language audit with an uncertain/multilingual route before transcription.

## Participant and session grouping

The paper reports 62 children, each sampled at one age within a single participation instance. Recordings for a child occurred over one to two weeks: some children contributed one day and others multiple days. The paper's voice-type benchmark used a child-disjoint 38/10/10 participant split, demonstrating that the authors could group that benchmark subset by child. The resulting 58 groups do not equal the paper's 62-child corpus total; the public paper does not explain this difference as a release schema fact, so it must be treated as benchmark-subset coverage rather than a current release count.

Implications:

- **Participant-disjoint primary split:** supported in schema by the media-link and anonymized-child fields, but still conditional on complete, one-to-one linkage for all eligible MP4/annotation objects. Values remain quarantined while deterministic HMAC group tokens and aggregate support are used internally.
- **Session/day-disjoint sensitivity split:** supported in schema by recording date, but still conditional on complete linkage and a frozen rule for multiple files on the same date. Exact dates never leave quarantine. File modification time is not a substitute.
- **Household/co-speaker leakage:** not addressed publicly. If multiple enrolled wearers could share a household or recurring speakers, the broadest permitted household/family grouping should dominate the split. If unavailable, this remains a limitation.
- **Paper benchmark split:** the video benchmark split clips 80/10/10 and is not documented as participant-disjoint. It must not be reused as evidence of leakage resistance for lexical grounding.

Even with grouping, held-out lexical evaluation remains conditional on corrected non-child speech and candidate noun/action types retaining train exposure and support across the required held-out groups. Public class totals cannot answer that question.

## Suitability of existing labels for the frozen pilot

Existing labels can safely define a content-blind sampling frame after release-link validation:

- coarse activity family;
- speech-presence category;
- coarsened duration stratum; and
- location, after collapsing sparse strata and enforcing cell suppression.

They cannot preselect on ASR words, visual salience, apparent referentiality, or expected grounding quality. The frozen 12–18-video pilot still requires utterance transcription/timing, speaker-role correction, and independent visible-candidate/null/ambiguity annotation. Existing activity boundaries may be displayed as context to an annotator only if the frozen protocol explicitly permits that; they are not gold action boundaries for noun/verb grounding.

## Gate implications for the coordinator

| Feasibility gate | Workstream-B evidence | Status from this audit alone |
| --- | --- | --- |
| G2 release/grouping | Live library UUID matches the DOI library; current UI counts and participant/date grouping fields are observed; immutable live revision, canonical hashes, and linkage completeness are absent | `CONDITIONAL` |
| G3 audio/timing | Raw synchronized audio and coarse timed speech bouts are reported | `UNRESOLVED`; utterance timing and decode coverage need pilot evidence |
| G4 transcript/role | Coarse child/other/overheard distinctions and person attributes exist; no lexical transcript/dedicated speaker-role field was observed in the inspected JSON prefix | `UNRESOLVED` |
| G5 input lexicon | Non-child-directed and overheard speech categories exist | `UNRESOLVED`; recurrence and language require corrected transcripts |
| G6 referential annotation | Broad object/activity contexts exist | `UNRESOLVED`; no referential labels or visibility reliability evidence |
| G7 held-out evaluation | Participant and recording-date split keys exist in schema; linkage completeness and lexical/action support after splitting are unproven | `UNRESOLVED` |
| G9 simulator calibration | Publicly reported raw streams are video/audio; no physical side streams are reported | Physical streams must remain unavailable/counterfactual unless the exact manifest proves otherwise |

These are evidence statuses, not a terminal scientific decision.

## Required release checks

The coordinator can complete this audit without exposing restricted values by providing a signed/sanitized receipt containing:

1. immutable live-view revision if exposed, observation date, and canonical manifest digest;
2. aggregate bytes, distinct-content hashes, and parseable/linkable counts for the observed MP4 and two JSON families;
3. literal annotation field names, data types, category codebook, time unit/time base, and missing/not-applicable encodings;
4. aggregate foreign-key integrity and annotation-to-media linkage rates;
5. number of participant groups and files lacking a group, with small-cell suppression;
6. participant and recording-date linkage coverage for every eligible MP4/annotation object;
7. union annotation coverage and usable-duration computation rules, not raw timestamps;
8. explicit presence/absence of transcript, object/referent, IMU, touch, proprioception, motor, gaze, or pose streams; and
9. explicit language documentation, or `UNKNOWN` if absent.

## Sources

- Suffo, Martin, Suffo, Haun, and Bohn, [ChildLens: An egocentric video dataset for activity analysis in children](https://link.springer.com/article/10.3758/s13428-026-02982-6), *Behavior Research Methods* 58, article 115 (2026), DOI `10.3758/s13428-026-02982-6`.
- Public [activity-class definitions, Table 1](https://link.springer.com/article/10.3758/s13428-026-02982-6/tables/1).
- Public [paper-snapshot class/location aggregates, Table 4](https://link.springer.com/article/10.3758/s13428-026-02982-6/tables/4). Sparse values were not reproduced in this report.
- [ChildLens dataset DOI](https://doi.org/10.17617/4.fe), its [pinned Keeper snapshot landing](https://keeper.mpdl.mpg.de/doi/libs/c8ed0104-b793-4c35-817e-302afd4e036b/4856662653b2fa183e53268d088cffba02a33443/), and the [DataCite registration](https://api.datacite.org/dois/10.17617/4.fe).
- Official Max Planck [ChildLens project page](https://www.eva.mpg.de/comparative-cultural-psychology/technical-development/childlens/).
- Official [ChildLens code repository](https://github.com/neleSuffo/ChildLens/tree/4923ca1411e9622ec91367d03e23627ed70ef19a) and [BMN benchmark documentation](https://github.com/neleSuffo/ChildLens/blob/4923ca1411e9622ec91367d03e23627ed70ef19a/docs/bmn_benchmark.md).
- Sanitized authenticated Keeper live-view receipt supplied by the coordinator, observed read-only on 2026-07-21; aggregate counts and schema field names only, with no media, values, identifiers, or exact timestamps exported.
