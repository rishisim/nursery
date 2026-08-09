# ChildLens permission, release-binding, and lexical-feasibility continuation v1.1

Date: 2026-07-21  
Terminal decision: `CHILDLENS_FEASIBILITY_REVISE`  
Scientific learner outcome authorized or run: no

## Decision

ChildLens remains a plausible sole empirical anchor for a naturalistic weak-alignment lexical-grounding prototype, but the prototype is not yet launch-ready. Version 1.1 resolves the principal governance uncertainty, binds the public DOI snapshot identity, safely reconciles the accessible live view, and freezes a content-blind 15-video preselection. It does not establish exact selected media bytes, live-to-snapshot byte equivalence, audio/language quality, corrected speaker roles, lexical recurrence, referential reliability, or held-out endpoint support. Those are essential gates, so `GO` is not justified.

No fundamental corpus failure was established. The remaining correction is bounded: complete the exact-size restricted manifest and selective acquisition, then obtain genuine blinded human auditory, language, speaker-role, and referential validation on the already frozen sample. Accordingly, `STOP` is also not justified.

This is a feasibility decision only. It does not support or refute the later synchronized-cue hypothesis, and no learner, causal arm, tokenizer, checkpoint, acquisition effect, or scientific outcome was trained or inspected.

## What changed from v1

Version 1 remains immutable and retains its historical `CHILDLENS_FEASIBILITY_REVISE` decision. Its 23-artifact path-and-SHA-256 set baseline is `35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf`.

Version 1.1 adds five evidence advances without changing the frozen thresholds or selection rules:

1. **Permission resolved from actual evidence.** A signed 2026-07-15 dataset access agreement, the dataset manager's 2026-07-16 approval, and the 2026-07-21 Keeper invitation/access completion support `G1_TERMS = PASS_WITH_CONTROLLING_CONSTRAINTS` for the named noncommercial academic project.
2. **Authenticated inspection made fail-closed.** The page-local extractor was synthetically sentinel-tested before use. It exports only fixed public release identities, broad counts/bytes, schema booleans, and constant errors; its local test suite passed 9/9 cases.
3. **Immutable public identity bound.** DOI `10.17617/4.fe`, Keeper library identity `c8ed0104-b793-4c35-817e-302afd4e036b`, and public snapshot commit `4856662653b2fa183e53268d088cffba02a33443` were observed. The accessible live view is separately receipted, but its exact byte equivalence to that snapshot is not proven.
4. **Pilot preselection frozen.** Fifteen videos were content-blindly preselected from the 192-item annotated source pool, one from each of 15 participant groups, using only the frozen metadata strata and hash ordering. This is a preselection receipt, not acquisition admission or empirical pilot evidence.
5. **Engineering readiness improved.** Restricted-manifest tooling, storage admission rules, local decode/UI preflight, and a blinded human-validation packet specification now exist. The metadata quarantine was admitted with owner/indexing/backup controls. Exact selected media bytes, model installation, language-matched staffing, and genuine human judgments remain outstanding.

## Permission actually established

The signed, approved use covers selective local copying and local decoding for the named project. Restricted transcripts and referential annotations, fixed local measurement instruments, and learner/checkpoint handling are permitted only as internal, secure, noncommercial, nonshared project materials. Nonidentifying aggregate reporting is permitted with the required ChildLens citation. The fail-closed retention review/deletion date is 2027-07-31 unless renewed in writing.

The receipt does not authorize cloud or vendor processing, third-party access, public checkpoints or derivatives, commercial use or commercial model training, a changed project purpose, or retention beyond the approved period. Each measurement instrument still needs its own compatible license. No agreement was accepted or modified in this task.

## Release and preselection evidence

The accessible live view contains 192 videos, 192 final annotation objects, complete participant-plus-recording grouping for 192 table rows, 58 participant groups, and 141 participant-plus-recording groups. The release receipt binds that observation while explicitly recording `live_to_public_snapshot_byte_equivalence = NOT_PROVEN`; no complete size-and-SHA-256 multiset comparison exists.

The content-blind preselection receipt records:

- restricted input digest: `81282e6a4d8178b1559a7473d22c70a36e44450467b6b5e2ff70ad9b7e9f049a`;
- placeholder manifest digest: `cc2b669e811ff78a045565be031537d5a93bfece40f27de3482768350adc1deb`;
- restricted download-plan digest: `bb39c32139310e30ac15095a15b3d4b95508ab217694f591845aac38e00364b8`; and
- human-validation packet digest: `ae8503fc5c21fc6df0b08c763202eab074b2526f83ef847d2faca3ae4eb217a5`.

The placeholder manifest is expressly not the final canonical manifest: exact remote video bytes are unresolved, its placeholder video sizes are not scientific evidence, no selected video was acquired, and no audio or visual content was inspected.

## Gate readout

| Gate | v1.1 status | Evidence and remaining requirement |
| --- | --- | --- |
| G1 terms | PASS WITH CONTROLLING CONSTRAINTS | Signed approved project evidence covers all required local actions under secure, noncommercial, nonsharing, aggregate-only, citation, and time limits. |
| G2 exact release and grouping | CONDITIONAL | The immutable DOI snapshot identity and a live observation are bound; participant/date grouping is structurally complete. Live-to-snapshot byte equivalence, exact selected remote sizes, media/annotation byte checksums, and the final restricted canonical manifest remain incomplete. |
| G3 audio and speech timing | UNRESOLVED | Local FFmpeg decode is reproducibly preflighted on synthetic audio, but no selected ChildLens media was decoded and no human timing coverage was measured. |
| G4 transcript and speaker role | UNRESOLVED | A human-only route and fixed ASR/alignment acquisition specifications exist, but actual language, corrected text/timing quality, and five-way role reliability are unmeasured. |
| G5 non-child input lexicon | UNRESOLVED | No human-dispositioned non-child transcript or corpus-local recurrence audit exists; the frozen 8 noun/object and 6 verb/action minimum has not been tested. |
| G6 referential annotation | UNRESOLVED | The ontology, ±5-second window, double-coding rule, and packet are fixed, but no genuine human visible/null/ambiguity/lag judgments or reliability estimates exist. |
| G7 held-out evaluation | UNRESOLVED | Participant/session grouping can support deterministic splitting, but lexical/action train exposure and support across at least three held-out participant groups are unmeasured. |
| G8 strong ceiling and weak baseline | CONDITIONAL | The construction protocol remains frozen, but release-specific timing/referential targets and canonical paired manifests have not been validated. No control learner was run. |
| G9 simulator calibration | PASS WITH LIMITATIONS | The permission receipt allows nonidentifying ChildLens video/audio aggregates. IMU, touch/contact, proprioception, and motor streams remain unavailable and simulator-defined counterfactual modalities only. |
| G10 storage, labor, and compute | CONDITIONAL | The low storage envelope passes, two CPU workers are admitted, and the metadata quarantine's owner/indexing/backup controls were established; the base envelope is narrow and the high case fails. Exact selected media bytes, human labor, and later compute remain unresolved. |

Essential `CONDITIONAL` and `UNRESOLVED` gates prevent `GO`. No essential unrepairable `FAIL` was observed, so the rubric selects `REVISE`.

## Privacy and provenance disposition

Version 1.1 report artifacts contain no raw media, decoded audio, frame, transcript, participant/session identifier, row-level manifest, exact media timestamp, or small lexical cell. Restricted inputs are represented only by permitted aggregate receipts and cryptographic digests. The validated extractor returned no raw text, rows, identifiers, or exact timestamps. No restricted payload was uploaded or sent to an external service.

One participant-table object and the final-annotation class were retained only in the owner-restricted, untracked quarantine after its exact root was excluded from content indexing and Time Machine. No video or audio was retained. Invalid HTML staging responses were rejected and unlinked rather than admitted as dataset content.

The historical v1 browser-output incident remains recorded and is not erased by this continuation. The new fail-closed extractor is the required remediation for subsequent inspection. No AEA or BabyView empirical data, aggregate, vocabulary, tokenizer, weight, checkpoint, result, or other empirical artifact is scientific ancestry for v1.1. A resource-preflight search did overmatch two nonempirical environment fields in an AEA provenance receipt; the values were unused and host facts were independently re-probed. This remains a disclosed strict-scope procedural exception, not empirical ancestry. Public method interfaces remain methodological references only.

No Keeper share was changed; no license was accepted; no message was sent; and nothing was committed, pushed, or published. The v1.1 manifest/preselection work did not train or expose a learner outcome.

## Validation

The fail-closed extractor passed 9 synthetic sentinel tests, the restricted-manifest builder passed 22 tests, the immutable v1 validator passed, and its 17-test suite passed. Synthetic FFmpeg conversion was byte-repeatable, and the local Streamlit/SQLite runtime imported successfully without restricted input. The v1.1 decision record parses as JSON, enumerates all 10 gates in order, preserves the four preselection digests, and passes report-only restricted-pattern checks. These are engineering/privacy checks; none substitutes for the missing ChildLens media and genuine human evidence.

## Exact next task

Run **ChildLens v1.1 restricted pilot acquisition and genuine blinded human validation**, without changing the 15-item selection or any frozen threshold:

1. Resolve exact remote bytes for the preselected objects without content inspection; build the final restricted canonical manifest and linkage/checksum receipt while continuing to state that DOI/live byte equivalence is unproven unless a complete comparison succeeds.
2. Revalidate the admitted hidden quarantine's ownership, retention, backup/indexing exclusions, and free-space controls immediately before media transfer. Enforce the 20 GiB raw-pilot cap, 73 GiB namespace peak cap, and 50 GiB projected post-peak free-space floor before every shard.
3. Selectively acquire only the frozen 15 items, establish actual language with qualified humans, and populate the blinded timing/text/role/referential packet. Codex may build and audit the packet but must not impersonate a human judge.
4. Apply G3-G8 and G10 exactly as frozen, export only cell-suppressed aggregates and digests, and issue a new decision. Do not train the CLIP+ learner or run any causal arm in that task.
