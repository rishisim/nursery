# Phase 1 source/data audit report

## Status: blocked by governed metadata transfer

The exact BabyView 2025.1 subset used for the published BabyView CLIP+ row is
not yet fully confirmed, but official public sources now provide a strong
candidate inclusion list that can be checked against authorized Databrary data.
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

## What the three hour totals may mean

- **894 hours:** the official total for the complete 2025.1 release.
- **868 hours:** the total used by an official BabyView analysis of the main
  cohort.
- **approximately 863 hours:** the video total reported by EgoBabyVLM.

The current metadata distinguishes `BV-main` and `Ego-SingleChild`. Within the
candidate date range it contains 4,733 main-cohort records and 893 single-child
records. This makes cohort selection a plausible explanation for much of the
894-to-868-hour difference, but it remains a hypothesis until durations are
summed. The remaining approximately five hours may reflect unusable or excluded
media, but no exclusion cause should be asserted without evidence.

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

The current Databrary metadata file was confirmed in the authorized web session,
but was not exported: the browser authorization cannot be safely delegated to a
Juno-side downloader, local controlled-data staging is forbidden, and Juno SSH
authentication became intermittent after the candidate transfer. The minimum
handoff is stable Juno SSH plus either (a) a Juno-side authorized Databrary
download mechanism or (b) a manual export of the current metadata CSV directly
into the governed manifest area. Exact matching, duration totals, and the bounded
media check remain gated on that file.

If exact matching cannot be completed, request the frozen
`babyview-videos-metadata-2025.1.csv` or the 2025.1 Airtable release export from
the BabyView maintainers. Official pipeline code confirms that release
membership is represented as an explicit Airtable list, not a date rule.

## Aggregate audit

- Candidate 2025.1 membership list located and checksum-verified in governed
  storage: yes (5,566 entries)
- Candidate/current aggregate comparison completed: yes, in memory
- Exact asset-level match: not completed; governed comparison required
- Complete authorized 2025.1 inventory present: not yet confirmed
- Total recordings and duration: not assessed
- Included/excluded duration: not assessed
- Missing, corrupt, or audio-unusable totals: not assessed
- Release membership evidence: official BabyView candidate inclusion list plus
  aggregate comparison against authorized current metadata
- 894 → ~863 reconciled: no
- Video downloads performed: none

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
