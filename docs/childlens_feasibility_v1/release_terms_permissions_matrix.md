# ChildLens release, terms, and permissions audit

Date: 2026-07-21  
Scope: governance and release identification only  
Outcome training authorized: no  
Authenticated Keeper inspection performed by this workstream: no

Sanitized authenticated Keeper receipt integrated: yes, supplied by the coordinator on 2026-07-21; no restricted identifiers, names, paths, or timestamps were received

## Bottom line for the feasibility gates

`G1_TERMS` is **UNRESOLVED**. Public sources say that ChildLens is available to researchers for research purposes after review, but they do not state whether an authorized recipient may make local copies, generate restricted transcripts or referential annotations, run local model training, or export nonidentifying aggregates. DataCite has no rights statement or version field, and the coordinator's read-only receipt found that the accessible library's archive-metadata `License` field is blank. Those permissions must therefore come from a recipient- and release-specific DUA/access letter or another explicit instrument. No license should be accepted or inferred from this report.

`G2_RELEASE_GROUPING` is **CONDITIONAL**. The accessible read-only library UUID matches the DOI library ID, and its README documents a participant table linking anonymized child grouping, age, gender, and recording date to media. However, the current live view did not expose an immutable revision hash, so equivalence to the DOI's fixed snapshot commit is not established. A canonical inventory digest and row-level grouping/linkage audit are still required.

The article's CC BY 4.0 license applies to the article and covered article material. It is not evidence that the restricted child recordings or their derivatives are CC BY 4.0.

## Exact public release identity

| Field | Public evidence | Audit interpretation |
| --- | --- | --- |
| Dataset DOI | `10.17617/4.fe` | Primary persistent identifier for the ChildLens data release. |
| Title | *ChildLens: An Egocentric Video Dataset for Activity Analysis in Children* | Matches the project, official code repository, and 2026 article. |
| DOI registration | Issued in 2025; DataCite record created and last updated 2025-07-23; resource type `Dataset` / `Library`; publisher is MPDL Keeper Service | Registration metadata, not a semantic dataset version. DataCite supplies neither `version` nor a rights entry. |
| Keeper library identity | Public DOI target contains Keeper library ID `c8ed0104-b793-4c35-817e-302afd4e036b` | This is a public repository identifier, not a participant or session identifier. |
| Keeper snapshot identity | The DOI landing's dataset link is a snapshot at commit `4856662653b2fa183e53268d088cffba02a33443` | This fixed commit is the strongest public release pin found. The link is marked “login required.” |
| Accessible live library | Coordinator's sanitized read-only receipt reports the same library UUID as the DOI target | Establishes repository identity, but not byte-for-byte equality to the DOI snapshot. |
| Accessible live revision | No immutable revision hash was exposed in the sanitized receipt | Record as `UNPINNED_LIVE_REVISION`; do not substitute the DOI commit. |
| Certification marker | The DOI landing displays Keeper's Cared marker | Archival/stewardship evidence only; see below. |
| Public project page | The MPI-EVA ChildLens page calls the DOI the “latest version” and instructs prospective users to request access | A discovery/access statement; it does not override the snapshot commit or grant processing rights. |

Canonical public landing: [ChildLens data DOI](https://doi.org/10.17617/4.fe). The DOI currently resolves to the [Keeper DOI landing](https://keeper.mpdl.mpg.de/doi/libs/c8ed0104-b793-4c35-817e-302afd4e036b/4856662653b2fa183e53268d088cffba02a33443/). DataCite registration is available through the [DataCite DOI API](https://api.datacite.org/dois/10.17617/4.fe).

This report treats the tuple below as the public release identity:

```text
(doi=10.17617/4.fe,
 keeper_library=c8ed0104-b793-4c35-817e-302afd4e036b,
 keeper_snapshot_commit=4856662653b2fa183e53268d088cffba02a33443)
```

The user's current read-only library and the DOI target share a library UUID, but only the DOI target exposes the fixed commit above. The current view must therefore be recorded as the same repository at an unpinned live revision. The audit must not silently claim that the current bytes equal the DOI snapshot.

## Public claims versus release observations

| Quantity or property | 2025 Keeper DOI landing | 2026 version-of-record article | Current read-only live view, sanitized receipt | Release-specific conclusion |
| --- | ---: | ---: | ---: | --- |
| Total video duration | 109 h, rounded | 108.58 h | Archive description says 109 h; no container-duration sum performed | Description is compatible with paper rounding, but not a release measurement. |
| Participants | 62 children | 62 children | Archive description says 62; README documents a participant-to-file mapping | Grouping is documented as available; completeness must be audited without exporting identifiers. |
| Video files | Not stated | 354 | 192 MP4 rows | The accessible live inventory differs from the paper's full-corpus count by 162 files. |
| Video bytes | Not stated | Not stated | 191 displayed sizes sum to approximately 201.33 GB; one row has no displayed size | 201.33 GB is a lower bound, not a complete library or video-byte total. |
| Annotated duration | 54 h | 51%, around 55 h | Archive description says 54 h; no interval-union computation performed | Compute release-local union duration under one definition after permission. |
| Annotated videos | Not stated | 192 | 192 final-format JSON rows and 192 platform-format JSON rows | The two folders appear to be parallel representations; they must not be counted as 384 annotated videos. |
| Root inventory classes | Not stated | Not stated | Two annotation-format folders, images, videos, archive metadata, archival certificate, participant table, and README | Supports a metadata-first manifest; only aggregate class names are reported. |
| Raw modalities | Video plus audio from a vest/chest-height camera | 1920 x 1080, 30 fps video plus microphone audio | MP4 objects present; streams/codecs not opened | Actual audio-stream coverage remains unverified. |
| Annotations | Five location classes and fourteen activity classes | Timed, exhaustive multi-label activities; activity, location, alone/with-others, and person age-class/gender fields described | Two 192-row JSON representations present; content not opened by this receipt | Schema semantics and foreign-key completeness still require permitted inspection. |
| License | Not stated | Article is CC BY 4.0; restricted data access described separately | Archive-metadata `License` field is blank | No dataset license can be inferred from article licensing or technical access. |
| Access | Dataset link requires login | Research access is restricted and reviewed | Current library is read-only | Access does not itself establish downstream-use permissions. |

The article states that annotation was ongoing and reports the full collected corpus as 354 video files and 108.58 hours, with 192 currently annotated videos and roughly 55 annotated hours. The current read-only live view instead contains 192 MP4 rows, exactly matching the article's annotated-video count, plus two annotation representations of 192 JSON rows each. This pattern is consistent with the accessible library containing the annotated subset rather than all 354 collected files, but that is an **inference**, not a proven mapping, until the release README/schema and foreign keys are audited. The former preliminary claim of about 343 videos is contradicted by the current sanitized UI receipt and is retired; it must not appear in planning denominators or reconciliation results.

Primary public sources: the [version-of-record article](https://doi.org/10.3758/s13428-026-02982-6), the [MPI-EVA ChildLens project page](https://www.eva.mpg.de/comparative-cultural-psychology/technical-development/childlens/), and the [official ChildLens repository](https://github.com/neleSuffo/ChildLens).

## Count reconciliation procedure

Run this procedure only after release-specific terms permit metadata/annotation processing. Keep the row-level manifest restricted.

1. Pin the source as either the DOI snapshot tuple above or a separately named live-library commit/share. Record the observation date and source role; never overwrite an earlier manifest.
2. Inventory every object without opening media content. For each object, assign exactly one top-level class: `VIDEO`, `ANNOTATION`, `IMAGE`, `METADATA`, `PARTICIPANT_TABLE`, `README_TERMS`, `CERTIFICATE`, or `OTHER`.
3. Report three video counts separately: filesystem video objects, distinct byte-identical video objects by SHA-256, and technically parseable video containers. Do not conflate them.
4. Parse annotation foreign keys only inside the restricted workspace. Report the size of the media-linked annotation set, the annotation-only/missing-media set, and the media-without-annotation set as cell-suppressed aggregates.
5. Compute media duration from container headers and compute annotated duration as the union of valid annotation intervals per media object. Also retain the naïve sum to detect overlapping multi-label intervals, but never report it as unique annotated hours.
6. Test for duplicate bytes, alternate encodes, unsupported extensions, zero-byte/corrupt objects, hidden files, and non-video auxiliaries. These checks often explain count differences without implying missing participants.
7. Audit whether participant and session grouping fields link to every eligible media object. Use release-local, keyed hashes internally; export only field-availability flags, cell-suppressed group totals, and the deterministic grouping algorithm.
8. Produce a reconciliation table with separate columns for DOI landing, article, current unpinned live view, canonical release manifest, and annotation linkage. Every difference receives a reason code: `ROUNDING`, `VERSION_DIFFERENCE`, `ANNOTATED_SUBSET`, `COUNTING_UNIT`, `DUPLICATE`, `UNLINKED_ANNOTATION`, `MISSING_MEDIA`, `CORRUPT`, or `UNRESOLVED`.

The canonical release digest is SHA-256 over sorted, canonical JSON records from the restricted manifest. Only the digest and permitted aggregate inventory leave the restricted workspace.

## Manifest schema and privacy boundary

The reportable manifest contains no filenames, paths, participant/session identifiers, exact recording timestamps, per-file durations, transcript text, or small cells.

```json
{
  "manifest_schema_version": "childlens-release-manifest-v1.0.0",
  "release_identity": {
    "doi": "10.17617/4.fe",
    "keeper_library_id": "PUBLIC_RELEASE_ID",
    "keeper_snapshot_commit": "PUBLIC_RELEASE_COMMIT",
    "source_role": "DOI_SNAPSHOT_OR_SEPARATELY_NAMED_LIVE_VIEW"
  },
  "canonical_restricted_manifest_sha256": "HEX_DIGEST",
  "inventory_summary": [
    {
      "top_level_class": "VIDEO_OR_OTHER_CLASS",
      "file_count": 0,
      "distinct_content_count": 0,
      "byte_total": 0,
      "availability": "PRESENT_OR_ABSENT_OR_UNRESOLVED"
    }
  ],
  "annotation_linkage_summary": {
    "linked_media_count": 0,
    "unlinked_annotation_count": 0,
    "media_without_annotation_count": 0,
    "unique_annotated_duration_seconds": 0,
    "cell_suppression_applied": true
  },
  "grouping_summary": {
    "participant_field_documented": true,
    "recording_date_field_documented": true,
    "session_grouping_construct_validated": false,
    "complete_linkage": false,
    "reportable_group_counts": "SUPPRESSED_OR_PERMITTED_AGGREGATE"
  }
}
```

The restricted manifest may additionally contain relative paths, original names, remote object version/ETag, bytes, SHA-256, media stream/container/codec metadata, exact per-file duration, annotation linkage, and keyed internal participant/session/media identifiers. These fields stay in the quarantined workspace. Internal keys should be `HMAC-SHA256(audit_secret, source_identifier)` with the secret stored separately; unsalted source hashes are vulnerable to dictionary reconstruction. Exact recording dates/times remain restricted even in derived reports. Any participant-linked or lexical cell below five must be suppressed or coarsened, without complementary-cell reconstruction.

## Terms and permissions matrix

| Proposed action | Public evidence | Status for this audit | Required release-specific evidence |
| --- | --- | --- | --- |
| Researchers may request access for research | Article and project page explicitly describe reviewed/restricted research access | `SUPPORTED_FOR_ACCESS_ONLY` | Access approval tied to the current recipient and release |
| Read release README, certificate, annotations, and metadata in place | Public sources do not state downstream terms; current task authorizes read-only inspection | `UNRESOLVED_TERMS` | Applicable DUA/access letter or release README clause |
| Selectively download a small pilot to local restricted storage | Not stated | `UNRESOLVED_TERMS` | Explicit local-copy and security/retention permission |
| Bulk-download the full archive | Not stated; scientifically unnecessary and does not fit the host policy | `PROHIBITED_BY_AUDIT_POLICY` | Not sought |
| Decode audio/video locally | Not stated | `UNRESOLVED_TERMS` | Explicit research-processing permission covering local decoding |
| Create ASR transcripts, corrected transcripts, diarization, speaker roles, or referential annotations | Not stated | `UNRESOLVED_TERMS` | Explicit permission for restricted derived personal-data artifacts, storage, retention, and deletion |
| Use fixed pretrained systems as quarantined measurement instruments | Not stated by data terms; learner-boundary amendment is separate | `UNRESOLVED_TERMS` | Processing permission plus approved instrument/data-flow controls |
| Train a ChildLens-only learner on permitted local derivatives | Research purpose is compatible in the abstract, but training permission is not explicit | `UNRESOLVED_TERMS` | Explicit permission for model training and treatment of checkpoints/weights |
| Export nonidentifying, cell-suppressed aggregates | Not stated | `UNRESOLVED_TERMS` | Explicit aggregate-publication/export permission and disclosure rules |
| Export raw/derived transcripts, frames, identifiers, paths, or exact timestamps | Sensitive restricted data; prohibited by this audit contract | `PROHIBITED` | Not sought |
| Redistribute media or restricted derivatives | No grant found | `PROHIBITED_UNLESS_EXPLICITLY_GRANTED` | Explicit redistribution license; outside current scope |
| Modify the Keeper share, upload, or invite users | Not authorized by this audit | `PROHIBITED` | Not sought |
| Accept a license/DUA | User-only legal/governance action | `DO_NOT_ACCEPT` | User instruction after reviewing the actual instrument |

No row becomes permitted merely because the user can technically see or download an object.

## Certificate implications

Keeper's public [Cared Data Commitment](https://keeper.mpdl.mpg.de/f/1b0bfceac2/) commits the service/depositor relationship to long-term access for entitled persons (at least ten years), checksum-based integrity, exportability, backups, retrievability, and depositor-controlled access. Keeper's public [service overview](https://keeper.mpdl.mpg.de/f/d17ecbb967/) describes the marker as evidence of long-term compliant archiving.

Therefore the Cared marker is positive evidence for archival stewardship and snapshot integrity expectations. It is **not** a dataset license, DUA, ethics approval for a new purpose, or permission for a recipient to download, transcribe, annotate, train, publish derivatives, redistribute, or export aggregates. “Exportability” in the commitment is a storage-service promise that filed data can be exported without loss; it is not a grant of legal reuse or public export rights. The depositor's retained access control reinforces the need for recipient-specific terms.

The Keeper DOI footer link labeled “[Terms of Services](https://keeper.mpdl.mpg.de/f/2206ad0c0a8346cb8f9e/)” currently resolves to a provider-identification/website data-protection notice, not to ChildLens-specific reuse terms. It cannot satisfy `G1_TERMS`.

## Storage and sharding policy

Current host preflight on 2026-07-21 showed about 122.3 GiB available on the data volume. The frozen pilot requires at least 30 GiB free reserve. The companion resource plan adopts a stricter 50 GiB operating floor, so the current project-peak headroom is about 72.3 GiB and controls admission. Free space must be rechecked immediately before each shard because this value is not stable.

The 191 displayed MP4 sizes sum to approximately 201.33 decimal GB, or about 187.5 GiB, and one MP4 row has no displayed size. Thus 201.33 GB is only a lower bound for video bytes. Even that lower bound would require more than 217.5 GiB free to retain the videos plus the mandatory 30 GiB reserve, exceeding current availability by more than 95 GiB; the stricter operating floor would require more than 237.5 GiB. This confirms the metadata-first, selective-only policy. Do not impute the missing size from an average.

1. Acquire only terms/README/certificate and annotation/metadata tables first, after terms permit it.
2. Resolve the 12–18-video pilot from metadata strata and the frozen hash order before watching or listening. Never choose files by lexical content, visual salience, ASR results, or apparent grounding quality.
3. Issue metadata-only `HEAD` or equivalent preflight requests for selected objects. Sum source bytes plus conservative derivative and temporary-space multipliers before acquiring any body. Cap raw pilot payload at 20 GiB and total projected ChildLens-feasibility peak at 73 GiB, subject to the stricter post-peak free-space test.
4. Stage selected media in deterministic batches of three to five videos. Verify source version/ETag where exposed and local SHA-256 before processing. One active shard is safest on this host.
5. Store raw media, derived audio, transcripts, and row-level manifests only in the restricted quarantine. Reports receive only permitted aggregates and digests.
6. Do not make a second full-resolution copy for frame extraction. Decode/stream bounded windows into ephemeral caches; cap caches by bytes and remove them only under the approved retention/deletion policy after integrity and measurement receipts are complete.
7. Abort acquisition if projected post-peak free space is below 50 GiB, if the release version changes, if a selected object would require share mutation/license acceptance, or if any byte is routed to an external service. The 50 GiB operating floor is deliberately stricter than the frozen protocol's 30 GiB minimum reserve.
8. No upload, cloud ASR, email, share change, commit, push, or publication is part of this policy.

## Evidence limitations and exact next governance check

This workstream did not authenticate to Keeper. It integrated a coordinator-supplied sanitized receipt from the user's read-only view: repository identity, aggregate root classes/counts/sizes, the blank license field, and README-level grouping availability only. No media, annotations, participant rows, identifiers, paths, recording dates, or exact timestamps were transferred into this workstream. The receipt establishes the current UI inventory described above, but not an immutable live revision, byte-level snapshot equivalence, media stream usability, row-level annotation linkage, or downstream processing permission.

The next check is read-only and content-minimal: compare the recipient's access letter/DUA and release README/certificate against every row of the permission matrix; record document names, versions/dates, and clause-level paraphrases without reproducing sensitive text. If accepting or choosing any legal instrument is required, stop and ask the user. Only after `G1_TERMS` passes may the canonical metadata manifest and selective pilot proceed.
