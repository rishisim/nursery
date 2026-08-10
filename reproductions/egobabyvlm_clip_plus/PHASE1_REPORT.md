# Phase 1 source/data audit report

## Status: blocked by exact usable-subset evidence

The exact BabyView 2025.1 subset used for the published BabyView CLIP+ row is
not yet fully confirmed. The official candidate list, official BabyView mapping,
and authorized Databrary exports have now been copied to owner-only governed
storage, checksum-recorded, and compared there.
The Databrary release metadata is reported as 894 hours; the pinned EgoBabyVLM
README describes approximately 863 hours. An official BabyView project analysis
of release 2025.1 separately reports 868 analyzed hours (and 3,163,675 frames
sampled at 1 Hz). Public evidence does not define what differs among these
accounting totals. The pinned preprocessing implementation
recursively accepts all media supplied to it, but publishes neither an
inclusion list nor a release-to-training selection rule. Therefore the roughly
31-hour difference—and the intervening 868-hour figure—is **not reconciled**,
and no exclusion causes are inferred.

## How the 2025.1 inventory was located

In simple terms:

1. The current Databrary volume is no longer the frozen paper dataset. It now
   presents release 2025.2 and reports a substantially larger corpus.
2. Databrary's From/To controls filter recording or session dates. They do not
   filter by release-upload date, which is why entering the volume and metadata
   upload dates returns no useful 2025.1 rows.
3. The official BabyView website source states that release 2025.1 totals 894
   hours and release 2025.2 totals 1,428 hours.
4. The official BabyView GitHub organization contains a February-2025 inclusion
   list with 5,566 entries. It was committed in April 2025, before the August
   2025 release, and its plausibly dated entries end on December 15, 2024.
5. The list was compared in memory with the authorized 2025.2 metadata visible
   in Databrary. No controlled metadata was downloaded or saved.
6. The current metadata contains 8,576 recordings. Restricting it to dates on
   or before December 15, 2024 gives 5,626 recordings, 60 more than the official
   candidate list. This proves that a date cutoff alone is not exact.
7. A privacy-safe participant-date multiplicity comparison matched 5,469 of
   5,511 plausibly dated candidate entries (about 99.2%). The remaining entries
   require filename- or record-level matching inside governed storage.

These results make the official February-2025 list the best available candidate
for frozen 2025.1 membership. They do not yet prove its total media duration or
that every listed file remains downloadable.

## Governed exact-match results

The official February-2025 candidate list contains 5,566 entries. The official
BabyView analysis mapping resolves 5,498 of them and leaves 68 without a mapping.
Its mapped durations split as follows:

- `BV-main`: 4,610 recordings and 868.4602 hours;
- legacy single-child cohort: 888 recordings and 69.6941 hours;
- mapped total: 5,498 recordings and 938.1543 hours.

Of the 5,498 mapped records, 5,456 occur in the current authorized BabyView
metadata and 42 do not. In a separate direct comparison of all 5,566 candidate
names to current Databrary asset filenames, 1,639 resolved uniquely and 3,927
require an authoritative rename/supersession crosswalk. No ambiguous exact
matches were found. Detailed rows, filenames, identifiers, references, paths,
and hashes remain only in governed storage.

## What the three hour totals mean so far

- **894 hours:** the official total for the complete 2025.1 release.
- **868 hours:** the total used by an official BabyView analysis of the main
  cohort.
- **approximately 863 hours:** the video total reported by EgoBabyVLM.

The exact official mapping establishes that the 868-hour figure is the
`BV-main` cohort: 868.4602 hours. EgoBabyVLM's approximately 863 hours is thus
consistent with using `BV-main` followed by about 5.46 hours of usability loss,
and is inconsistent with using the full mapped candidate list. This is strong
aggregate evidence for `BV-main`, but it is not an exact paper manifest. Neither
the public code nor the governed exports identify which recordings account for
the remaining approximately 5.46 hours or why they were excluded.

The official 894-hour release total also does not equal the 938.1543 hours in
the candidate mapping. The sources evidently use different release/accounting
boundaries, but no record-level official crosswalk currently explains that
difference. It would be scientifically incorrect to force 894, 868.4602, and
approximately 863 into a single inferred ledger.

## Readiness gate before downloading video

The frozen dataset is ready for download only after all of the following pass:

1. Copy the official 5,566-entry candidate inclusion list into governed storage
   and record its Git commit and SHA-256. Do not commit the populated list.
2. Obtain an authorized Databrary CSV/export containing the current asset
   references and source filenames directly into governed storage.
3. Match candidate entries to Databrary assets using exact filename/record
   mapping, not only dates. Resolve every unmatched, duplicate, renamed, or
   superseded entry.
4. Confirm that the matched inventory contains exactly the intended 2025.1
   membership and determine whether both `BV-main` and `Ego-SingleChild` belong
   in the EgoBabyVLM training corpus.
5. Read media duration and stream metadata without downloading the videos where
   Databrary exposes it. Otherwise, download only a tiny bounded sample to
   governed scratch storage to verify naming and media access.
6. Sum video duration by cohort and availability status. The totals should
   explain 894, 868, and approximately 863 hours rather than merely resemble
   them.
7. Produce a privacy-safe aggregate record containing counts, durations,
   unmatched totals, exclusion reasons, manifest hashes, and the confirmed
   download selection. Only then start the bulk video download.

## Governed readiness run

The official candidate list has been copied to the owner-only durable manifest
area without writing it to the worktree. Its governed provenance record contains:

- 5,566 entries;
- repository head `94b1dffb5e9a488dc2c08a84ce7967cdcae272ab`;
- file-introducing/last-modifying commit
  `59cc646e6d73a54a37e0c0820f9389f2e4737595`;
- SHA-256 `0e79d853b1c78d0c0272391ba1109b3525790f5a59a8421bc4df6b2f42153d40`.

The checksum was verified after transfer. No candidate rows or identifiers are
tracked in Git.

The authorized full-volume asset export, session sample, current BabyView
metadata export, and official BabyView duration/mapping table were transferred
to governed storage and checksum-recorded. A governed private match ledger and
aggregate companion record were produced. The task-created local metadata
download was then deleted. Intermittent SSH authentication remains an
operational concern but did not prevent the completed governed comparison.

The minimum external evidence is the authors' exact privacy-safe training
inclusion manifest (or its digest plus a governed matchable list), including
record-level exclusion reasons, together with the frozen 2025.1
release-to-asset rename/supersession crosswalk. The frozen 2025.1 Airtable
release export would also resolve the release/accounting boundary. Official
pipeline code confirms that release membership is represented as an explicit
Airtable list, not a date rule.

## Aggregate audit

- Candidate 2025.1 membership list located and checksum-verified in governed
  storage: yes (5,566 entries)
- Candidate/current aggregate comparison completed: yes, in governed storage
- Official candidate-to-duration mapping: 5,498 matched; 68 unmapped
- Exact current asset filename match: 1,639 matched; 3,927 need an authoritative
  rename/supersession crosswalk
- Complete authorized 2025.1 inventory present: not yet confirmed
- Official mapped recordings and duration: 5,498 and 938.1543 hours
- `BV-main` recordings and duration: 4,610 and 868.4602 hours
- Legacy single-child recordings and duration: 888 and 69.6941 hours
- Included/excluded duration: not assessed
- Missing, corrupt, or audio-unusable totals: not assessed
- Release membership evidence: official BabyView candidate inclusion list plus
  aggregate comparison against authorized current metadata
- 894 → ~863 reconciled: no; `BV-main` is strongly supported, but the exact
  approximately 5.46-hour usability exclusion and 894-hour boundary are unknown
- Video downloads performed: two bounded exact-match files; bulk download
  remains gated on a frozen manifest and the unresolved 489-record rename set

## Rolling-transfer and ZIP workflow check

With explicit authorization, a one-file rolling transfer was tested: one exact
`BV-main` video was downloaded through the authorized browser, streamed to
governed Juno, byte-count and SHA-256 verified, and deleted from the laptop.
The governed queue checkpoint records two completed exact files at this stage;
no bulk run has been started.

The Databrary `Download all as ZIP` control was then tested with one-day date
ranges. The session table remained empty for the tested recording/upload-date
windows, the ZIP request produced no downloadable archive, and no ZIP was left
on the laptop. The control therefore cannot currently be treated as a viable
filtered 2025.1/`BV-main` workflow. It also provides no cohort selector, so even
a successful date-filtered ZIP would not establish `BV-main` membership.

## Primary evidence inspected

- BabyView 2025.1 controlled release: <https://databrary.org/volume/1882>
- Pinned EgoBabyVLM source and ≈863-hour statement:
  <https://github.com/facebookresearch/egobabyvlm/tree/224621caf0628270b6115845ac75a65b984234a3>
- Pinned preprocessing documentation and implementation:
  <https://github.com/facebookresearch/egobabyvlm/tree/224621caf0628270b6115845ac75a65b984234a3/apps/data_preprocessing>
- Official BabyView project release-2025.1 analysis reporting 868 hours:
  <https://babyview-project.github.io/assets/pdf/BV_Objects_CCN_2025.pdf>
- Official BabyView dataset source stating the 2025.1 and 2025.2 totals:
  <https://github.com/babyview-project/babyview-project.github.io/blob/master/_pages/dataset.md>
- Official BabyView dataset repository containing the candidate inclusion list:
  <https://github.com/babyview-project/babyview-dataset>
- Official BabyView pipeline showing explicit release membership:
  <https://github.com/babyview-project/babyview-pipeline>
- EgoBabyVLM paper v1: <https://arxiv.org/abs/2605.19130v1>

Only this aggregate status is tracked. Detailed ledger rows, source paths,
references, and hashes are restricted to governed storage.
