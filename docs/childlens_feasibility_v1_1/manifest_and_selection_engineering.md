# ChildLens v1.1 manifest and pilot-selection engineering

Version: `childlens-manifest-selection-engineering-v1.1.0`  
Date: 2026-07-21  
Scope: local metadata engineering only; no Keeper access, acquisition, content inspection, or learner outcome

## Result

`scripts/build_childlens_restricted_manifest_v1_1.py` now provides a fail-closed, two-output boundary for the frozen pilot:

1. a canonical restricted manifest with row-level operational metadata and the selected internal records, written only to quarantine outside the repository; and
2. an allowlisted reportable receipt containing release-level digests, broad inventory summaries, linkage/grouping status, cell-suppressed counts, resource admission, and an identifier-free pilot summary.

The implementation consumes only a caller-supplied local JSON metadata inventory. It has no network client and performs no download, media decode, transcript processing, annotation, training, evaluation, or scientific decision. The v1 protocol and all v1 audit artifacts remain unchanged.

## Fail-closed boundary

The restricted output path is resolved before input processing. The tool refuses a path equal to or below the repository root, including an existing symlink alias into the repository. It also rejects a collision between restricted and reportable outputs. The restricted file is created atomically with mode `0600`; the reportable receipt is created atomically only after the restricted artifact succeeds. If the public write fails, the newly written restricted artifact is removed rather than leaving an unreceipted result.

All validation failures use constant error codes. A malformed filename, key, locator, annotation link, or metadata value is never copied into an exception or command-line diagnostic. Unknown fields fail schema validation, so lexical strings, transcripts, ASR output, detector output, visual-salience fields, and other content-derived selection features cannot be silently added.

The input attestations require all internal media, object, participant, session, and annotation keys to be release-local HMAC-SHA-256 values. Only the key-derivation receipt digest is stored; the HMAC secret must be held separately from both the repository and restricted manifest.

## Restricted input contract

The exact schema identifier is `childlens-restricted-manifest-input-v1.1.0`. Its top-level records are:

- `release`: the public DOI and library identity, public DOI snapshot commit, source role, optional observed immutable revision, observation receipt digest, optional independent DOI-snapshot inventory comparison receipt, and the fixed `PARTICIPANT_PLUS_RECORDING_DATE` session definition;
- `attestations`: HMAC-key use, metadata freeze before content inspection, absence of content fields, and the key-derivation receipt digest;
- `objects`: opaque object key, restricted source locator, one fixed top-level class, byte size when known, source/local SHA-256 when available, remote version/ETag when exposed, selective-copy availability, and optional container-parse status;
- `annotations`: opaque annotation key, annotation-object link, optional media foreign key, and one of the fixed representation classes; and
- `media`: opaque media/object/participant/session keys plus duration in integer milliseconds and only the four frozen metadata strata—coarse activity, speech-presence bin, duration source, and location when available.

Duration is converted to `LOW`, `MIDDLE`, or `HIGH` tertiles inside quarantine. Exact durations never enter the reportable receipt. Object keys, source locators, remote versions, per-object checksums, participant/session keys, annotation foreign keys, selected row keys, and full stratum values remain restricted.

The schema accepts incomplete object sizes/checksums for inventory reconciliation but excludes objects with unknown or zero byte size from the pilot. Any annotation link to a nonexistent media record, media link to a non-video object, annotation link to a non-annotation object, duplicate operational key, duplicate locator, partial grouping, or session spanning more than one participant fails. Byte-identical objects are not silently removed: duplicate checksums remain separate object records, while the reportable inventory distinguishes file count from known distinct-content count.

## Canonical digest and release binding

The canonical restricted release payload is encoded as UTF-8 JSON with sorted keys, no insignificant whitespace, no NaN values, and deterministic record ordering. Its SHA-256 is the `canonical_restricted_manifest_sha256`. The snapshot-comparison assertion fields and pilot selection are excluded from this release digest, avoiding self-reference. The frozen selector binds to this digest.

Release binding is explicit:

| Accessible source | Binding recorded | DOI-snapshot equivalence |
| --- | --- | --- |
| Public DOI snapshot at commit `4856662653b2fa183e53268d088cffba02a33443` | DOI commit plus restricted manifest digest | Source is bound to the public DOI snapshot commit; complete checksums are still reported separately. |
| A separately named immutable live commit | Named live commit plus restricted manifest digest | Not inferred merely because the library identity matches. |
| An unpinned live view | Restricted manifest digest plus observation-receipt digest | Explicitly `NOT_PROVEN`; the receipt states that only the local observation is bound. |
| Live inventory with an independent DOI-snapshot size/SHA-256 multiset receipt | Compare sorted `{size_bytes, sha256}` records, retaining duplicates | `PROVEN_BY_COMPLETE_SIZE_AND_SHA256_MULTISET_DIGEST` only when every object has size and SHA-256 and both multiset digests match. A mismatch fails. |

The identity-neutral object-inventory digest permits a live revision and DOI snapshot to be compared without forcing their source-role or observation fields to match. When either inventory lacks complete object SHA-256 values, the tool records `NOT_PROVEN_CURRENT_OBJECT_CHECKSUMS_INCOMPLETE`; it does not promote a metadata match, common library identifier, matching object count, or technical access into a byte-equivalence claim.

The public DOI and library identifier are reportable release identifiers. An observed private/live revision value, observation receipt digest, and comparison-receipt digest stay inside the canonical restricted payload; the public receipt exposes only binding booleans/statuses and the release-level manifest/object-inventory digests.

## Frozen selection algorithm

The selector verifies that the on-disk v1 protocol still names:

- protocol `childlens-lexical-feasibility-pilot-v1.0.0`;
- target 15 and allowable range 12–18;
- the exact frozen SHA-256 ordering formula;
- no full-archive download; and
- no learner training.

A mismatch fails before selection. The protocol file's byte digest is included in both outputs.

Eligibility is determined before media content inspection: a media row must have a valid annotation foreign key, complete participant/session grouping, known positive object bytes, known duration metadata, and selective-copy availability. The complete annotated set must have participant/session grouping; the tool does not quietly drop grouped failures. Location may be unavailable and is then a fixed missing-metadata stratum.

The selection procedure is deterministic:

1. Sort eligible duration metadata and assign deterministic tertiles.
2. Form combined strata from coarse activity, speech-presence, duration tertile, and location availability/value.
3. Within each stratum, order candidates by `SHA-256(release_manifest_digest || protocol_id || internal_episode_key)` with literal string concatenation, exactly as frozen.
4. Repeatedly select from a least-filled stratum, with deterministic stratum and candidate hash tie-breaks.
5. Complete a one-video-per-participant pass before allowing a second video from any participant. Never exceed two.
6. Reject any candidate that would exceed the caller-supplied raw cap. Do not substitute based on lexical content, ASR, visual salience, apparent groundability, or learner behavior.

The default target is 15. A caller may request only 12–18. If access/eligible-pool or raw-byte constraints prevent the requested target, the result may contract only to at least 12 and records a fixed reason code; fewer than 12 fails. Selection retains duplicate-byte records as distinct episodes because file deduplication is an inventory finding, not authorization to change the frozen episode unit.

## Resource admission

The caller must supply exact integers for the raw pilot cap, projected namespace peak, and currently free volume bytes. The tool enforces all three v1.1 ceilings before selection:

- raw pilot payload at most 20 GiB;
- projected ChildLens-feasibility namespace peak at most 73 GiB; and
- projected post-peak free space at least 50 GiB.

The selected aggregate byte total must also remain within the caller's possibly stricter raw cap. These checks are admission controls only. They do not reserve space or acquire data, so free space must be measured again immediately before every future shard.

## Reportable receipt and suppression

The reportable schema is `childlens-manifest-selection-aggregate-receipt-v1.1.0`. It contains only:

- public release identity, binding mode/status, release-level digests, and value-free limitations;
- a fixed eight-class inventory with availability, cell-suppressed counts/totals, checksum-completeness flags, and known distinct-content aggregates;
- cell-suppressed annotation/media linkage partitions;
- grouping availability/completeness and group totals subject to suppression;
- frozen protocol identity/digest, aggregate selected count/bytes/group count, maximum reuse, content-blind field names, aggregate stratum coverage, selection digest, and fixed adjustment reason codes;
- resource admission totals and hard ceilings; and
- explicit restricted-payload absence flags.

Positive cells below five are suppressed. For a partition containing a positive small cell, the largest nonsmall cell is also suppressed to prevent direct complementary reconstruction. Per-class byte and distinct-content totals are suppressed whenever that class's file count is suppressed. Zero cells may be reported as zero; a zero contains no person or lexical example. No stratum label values are exported—only the number of distinct selected values, itself suppressed below five.

The pilot-selection digest is computed over restricted, HMAC-keyed selection rows and functions as a procedural receipt. The rows themselves and their hashes never enter the reportable artifact.

## Invocation shape

After permissions pass and a restricted input has been created outside the repository, invoke the tool with absolute paths supplied by the coordinator:

```text
python3 scripts/build_childlens_restricted_manifest_v1_1.py \
  --input RESTRICTED_INPUT_PATH \
  --restricted-output RESTRICTED_OUTPUT_PATH \
  --reportable-output REPORTABLE_OUTPUT_PATH \
  --raw-cap-bytes RAW_CAP_INTEGER \
  --projected-namespace-peak-bytes PROJECTED_PEAK_INTEGER \
  --volume-free-before-bytes CURRENT_FREE_INTEGER \
  --target 15
```

No restricted path or value is printed. Success prints only the canonical restricted manifest digest and selected count. Failure prints only a constant error code.

## Synthetic verification

Command run:

```text
pytest -q tests/test_childlens_restricted_manifest_v1_1.py
```

Result at creation: `22 passed`.

The tests use synthetic HMAC-shaped keys, locators, metadata, and checksums. They cover deterministic canonicalization under reordered input; the exact selection formula; metadata-stratum balance; distinct-participant preference and the two-video cap; target contraction and hard resource caps; duplicate keys/locators; byte duplicates; foreign-key integrity; partial and cross-participant grouping; rejection of a transcript field; complementary cell suppression; unpinned-live and DOI-snapshot binding; complete/incomplete/mismatched inventory comparisons; restricted output refusal inside the repository and through a symlink alias; atomic outside-repository writing and restricted file mode; reportable-payload sentinel absence; and continued conformance of the immutable v1 protocol file.

## Limits and handoff

This engineering artifact is not evidence that G1 permissions pass, that the current Keeper bytes match the DOI snapshot, that the live release contains any particular count, that participant/date linkage is complete, or that a pilot is scientifically feasible. Input attestations and receipts must be supplied from actual authorized inspection; the tool validates structure and deterministic handling, not the truth or legal sufficiency of the source evidence.

If the accessible source has no immutable revision, the admissible fallback is the deterministic restricted manifest digest plus an observation receipt, with DOI byte equivalence explicitly unresolved. A later checksum-complete independent snapshot comparison may strengthen that statement but must create a new observation/receipt rather than overwrite the earlier one.

No acquisition should occur unless the permission workstream passes every required clause and the coordinator verifies the reportable receipt and restricted quarantine boundary. This workstream makes no GO, REVISE, or STOP decision.
