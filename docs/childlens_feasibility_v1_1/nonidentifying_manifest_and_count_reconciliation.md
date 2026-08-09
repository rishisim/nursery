# ChildLens v1.1 nonidentifying manifest and count reconciliation

Version: `childlens-nonidentifying-manifest-reconciliation-v1.1.0`  
Date: 2026-07-21  
Scope: aggregate release binding and structural reconciliation only

## Result

The public ChildLens DOI snapshot is bound to Keeper snapshot commit `4856662653b2fa183e53268d088cffba02a33443`. The accessible live view is separately bound by the v1.1 redacted observation receipt. Byte-for-byte equivalence between those two sources is **not proven**: the live view exposed no immutable revision and the audit did not obtain a complete size-and-SHA-256 multiset for both sources.

The live-view aggregate is internally coherent at the counting level: 192 video objects, 192 final annotation objects, and a 192-row participant table. Every participant-table row has the fields needed for participant-plus-recording grouping; the table yields 58 participant groups and 141 participant-plus-recording groups. These are release-view aggregates, not public-paper corpus totals.

The authorized restricted inventory subsequently supported a deterministic, content-blind 15-item preselection. Canonical-stem structural joins cover all 192 annotation/media references and all 192 participant-table rows. That proves the operational structural linkage used for preselection, but not byte identity: exact selected remote video sizes, per-object media checksums, and the final canonical restricted-manifest digest remain unresolved. The frozen placeholder manifest therefore binds selection order only and is not acquisition admission or scientific evidence about media content.

## Bound identities and receipts

| Item | Bound value or status |
| --- | --- |
| Public DOI | `10.17617/4.fe` |
| Public Keeper library | `c8ed0104-b793-4c35-817e-302afd4e036b` |
| Public DOI snapshot commit | `4856662653b2fa183e53268d088cffba02a33443` |
| Accessible pilot source | live read-only view |
| Immutable live revision observed | no |
| Live-to-DOI byte equivalence | `NOT_PROVEN` |
| Release-binding receipt SHA-256 | `2aa1dcf53fc5592965019afe5df97b51ccc2fe05e70acb2d51aa396c73843805` |
| Frozen v1 pilot protocol SHA-256 | `63be49d470c79e94f26240e73b314cdc42f2a6e6a57e96c897340d856f561ecb` |
| Preserved v1 artifact-set SHA-256 | `35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf` |

The release-binding receipt digest identifies the reportable observation receipt, not the underlying media bytes. The public snapshot commit identifies an immutable repository snapshot, but it does not prove that the separately accessible live objects are the same bytes.

## Aggregate reconciliation

| Inventory or linkage unit | Aggregate | Interpretation |
| --- | ---: | --- |
| Live video objects | 192 | UI display count |
| Live video displayed-size parse count | 192 | all displayed video rows yielded a size-like value |
| Live video displayed-size parsed total | 201,584,000,000 bytes | approximate display-derived total; **not exact object bytes** |
| Live exact video-size attributes | 0 | prevents an exact byte manifest from this receipt |
| Live final annotation objects | 192 | UI object count |
| Live final annotation displayed known bytes | 18,295,700 bytes | display-derived known-byte total |
| Local annotation archive | 1,021,655 bytes | local package size; its operational structural joins were validated, but byte equivalence to the live display is not proven |
| Local extracted annotations | 192 files; 18,236 KiB | local extraction summary; not a canonical byte multiset |
| Participant-table rows | 192 | aggregate row count |
| Participant groups | 58 | restricted values were not exported |
| Participant-plus-recording groups | 141 | restricted values were not exported |
| Rows with complete participant-plus-recording grouping | 192 of 192 | grouping-field completeness, not split adequacy |

Beyond the matching counts, a restricted canonical-stem audit validated all 192 annotation/media references and all 192 participant-table joins. This establishes the operational structural linkage used by the frozen preselection without exporting names or keys. It does not prove byte identity between the live view and the local archive, nor media/annotation checksum equivalence. The local archive's compressed size and the extracted filesystem summary are not directly comparable to the UI's displayed annotation-byte total. Compression, allocation/reporting units, or source-version differences could explain the mismatch; none is selected as the explanation without checksum evidence.

## Count discrepancy interpretation

The accessible live view's 192 video objects must be treated as its own release-view inventory. It cannot be silently substituted for a public-paper total or interpreted as the full ChildLens corpus merely because 192 is also a known annotated-video count. The public DOI snapshot is immutable, while the live view remains mutable/unversioned from this audit's perspective.

The only defensible binding for a live-view pilot is therefore:

```text
public DOI/library/snapshot identity
+ redacted live observation-receipt digest
+ frozen placeholder-manifest and selection receipts
+ pending final exact-byte canonical restricted-manifest digest
```

This composite binding must continue to state `live_to_public_snapshot_byte_equivalence = NOT_PROVEN` unless a complete identity-neutral size-and-SHA-256 multiset comparison later passes.

## Frozen pilot preselection status

The frozen protocol permits 12–18 videos, targets 15, requires participant-distinct metadata-stratified hash ordering, and forbids content-driven selection. A later restricted run froze 15 items from the 192-item pool, each from a distinct participant group, using coarse activity, speech presence, an annotation-derived duration-proxy tertile, location when available, and protocol-bound hash ordering. It used no lexical text, audio or visual content, ASR output, referential judgment, or learner result.

The nonidentifying receipt binds:

- restricted preselection input SHA-256: `81282e6a4d8178b1559a7473d22c70a36e44450467b6b5e2ff70ad9b7e9f049a`
- placeholder manifest SHA-256: `cc2b669e811ff78a045565be031537d5a93bfece40f27de3482768350adc1deb`
- restricted download-plan SHA-256: `bb39c32139310e30ac15095a15b3d4b95508ab217694f591845aac38e00364b8`
- empty human-validation packet SHA-256: `ae8503fc5c21fc6df0b08c763202eab074b2526f83ef847d2faca3ae4eb217a5`

The placeholder used non-evidentiary size values solely to freeze deterministic selection. `canonical_restricted_manifest_sha256` and the exact selected raw-byte total remain pending until exact remote sizes and local checksums are resolved content-blindly. The 15 items may not be replaced or expanded based on later content or yield.

## Privacy disposition

This artifact contains only public release identifiers, aggregate counts, approximate release-level byte totals, digests of reportable/frozen artifacts, and fixed reason codes. It contains no filenames, paths, participant/session values, timestamps, transcripts, annotation rows, selected episode keys, or small identifiable cells. No AEA or BabyView empirical artifact contributes to this reconciliation.
