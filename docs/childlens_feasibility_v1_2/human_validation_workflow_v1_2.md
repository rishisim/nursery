# ChildLens v1.2 blinded local human-validation workflow

Version: `childlens-human-validation-workflow-v1.2.0`  
Status: implementation complete; restricted packet population and human coding are separate runtime steps  
Scientific learner outcome authorized: **no**

## Purpose and scope

This workflow implements the human-only judgments that automated measurement cannot establish: language, utterance timing and source-language text, speaker role, visible/null/ambiguous reference in the frozen ±5-second window, independent reliability, and adjudication. It is a local annotation instrument, not a scientific learner and not a source of acquisition-effect evidence.

The implementation was developed and tested without reading ChildLens media, annotations, identifiers, transcripts, filenames, paths, or exact timestamps. Runtime row data and autosaves are confined to the already authorized restricted quarantine. Repository reports may receive only permitted nonidentifying aggregates and digests after a separate export review.

## Components

- `scripts/childlens_human_validation_v1_2.py` is the fail-closed SQLite controller, validator, and minimal command-line interface.
- `scripts/childlens_human_validation_app_v1_2.py` is the local Streamlit interface.
- `scripts/launch_childlens_human_validation_v1_2.py` is the required zero-argument launcher. It discovers exactly one initialized owner-only runtime without serializing its location, binds Streamlit to `127.0.0.1`, enables XSRF protection, disables telemetry and file watching, opens an ephemeral-token-authenticated app window in a dedicated browser profile/cache inside quarantine, and suppresses verbose logs.
- `tests/test_childlens_human_validation_v1_2.py` exercises quarantine placement, immutable packet binding, independent-coder separation, locked records, ontology rejection, deterministic double-coding, the fixed window, progress logic, and adjudication.

The SQLite file, WAL files, HMAC key, audit trail, human text, labels, timing, and adjudications all remain inside the restricted root with owner-only modes. No repository path is accepted as a store.

## Runtime admission contract

The application fails closed unless all of the following are true:

1. Administrative initialization retains the explicit path-plus-`CHILDLENS_V12_QUARANTINE_ROOT` double opt-in. The human-app launcher takes no arguments and independently requires exactly one initialized runtime in an owner-only, ChildLens-labelled hidden sibling namespace; the app receives only a process-local authorization, never a root path in argv or its environment.
2. The resolved root is outside this repository, exists, is owned by the current user, has no group/other access, and contains no symlink component.
3. The root contains the Spotlight exclusion sentinel.
4. An owner-only v1.2 quarantine policy receipt attests git exclusion, indexing exclusion, backup exclusion or encrypted local-only equivalence, local-only operation, and the controlling `2027-07-31` retention deadline.
5. The packet manifest is inside that root and has exactly 15 rows.
6. The restricted v1.1 skeleton is also inside the root, has canonical digest `ae8503fc5c21fc6df0b08c763202eab074b2526f83ef847d2faca3ae4eb217a5`, and authenticates the frozen selection digest and exact 15-key set.
7. The restricted v1.1 canonical input has canonical digest `81282e6a4d8178b1559a7473d22c70a36e44450467b6b5e2ff70ad9b7e9f049a` and fixes each selected key’s video-object link, duration, annotation linkage, and metadata stratum.
8. A self-sealed native-transfer receipt is `COMPLETE`, carries the same pilot selection digest, and binds every video object to its content-addressed local SHA-256 path.
9. The v1.2 manifest contains exactly those cross-bound key→object→content-path, duration, annotation-linkage, and stratum values; it cannot permute media between opaque display keys.

The policy bootstrap command is not an automated claim that the controls exist. The coordinator may run it only after separately verifying every attested control:

```bash
export CHILDLENS_V12_QUARANTINE_ROOT=/absolute/restricted/root
python3 scripts/childlens_human_validation_v1_2.py bootstrap-policy \
  --root "$CHILDLENS_V12_QUARANTINE_ROOT" \
  --attest-all-controls
```

The example path is a placeholder and must never be copied into a report. The packet is then initialized once:

```bash
python3 scripts/childlens_human_validation_v1_2.py init \
  --root "$CHILDLENS_V12_QUARANTINE_ROOT" \
  --packet "$CHILDLENS_V12_QUARANTINE_ROOT/restricted_packet_v1_2.json" \
  --frozen-v1-1-skeleton "$CHILDLENS_V12_QUARANTINE_ROOT/frozen_v1_1_packet.json" \
  --frozen-v1-1-restricted-input "$CHILDLENS_V12_QUARANTINE_ROOT/frozen_restricted_input.json" \
  --native-transfer-receipt "$CHILDLENS_V12_QUARANTINE_ROOT/native_transfer_receipt.json"
```

Initialization authenticates the skeleton and restricted-input digests, selection digest, exact key set, sealed native-transfer receipt, key→object→content linkage, duration, annotation linkage, frozen stratum, actual media SHA-256, and owner-only media. It stores a canonical packet digest and refuses any later packet change. It also generates distinct high-entropy tokens for every coder/adjudicator slot and saves them in one owner-only coordinator packet inside the workflow directory. Tokens are distributed offline and cannot be self-provisioned through the app. Initialization does not acquire or discover media.

## Restricted packet schema

The packet is generated only after selective acquisition and content-addressed storage. Its top-level fields are:

- `schema_version`: exactly `childlens-human-validation-workflow-v1.2.0`;
- `opaque_key_attestation`: `true` only after the coordinator verifies that no source identifier or name survives;
- `frozen_selection_digest`: the unchanged v1.1 selection binding; and
- `items`: exactly 15 allowlisted records matching the authenticated skeleton.

Each record contains only `internal_key`, `display_key`, `media_relpath`, `stratum_key`, `duration_ms`, `batch_number`, `source_object_key`, `media_sha256`, and `annotation_linkage_sha256`. Display keys follow `HV-001` form. Internal and stratum keys are release-bound opaque values stored only in quarantine. Media paths must be the native receipt’s content-addressed `raw_v1_2/<SHA-256>.bin` entries (or an independently sealed future opaque preparation binding). Byte-identical selected objects may share that one path, but their source-object keys must remain distinct and each key→object→SHA/path relation must independently match the frozen input and sealed transfer receipt. Participant/session keys, source filenames, source paths, transcripts, exact timestamps, and recording dates are rejected at import.

## Human stages

### 1. Language and audio integrity

Two qualified humans independently complete `LANGUAGE_A` and `LANGUAGE_B`. They hear the same content-blind item allocation but never see the peer record or a machine language proposal. Each records:

- an ISO 639 code, `MIXED_OR_CODE_SWITCHED`, or `UNDECIDABLE`;
- language competence;
- `USABLE`, `PARTIAL`, `UNUSABLE`, or `UNDECIDABLE` audio integrity;
- speech and overlap presence; and
- an irreversible lock.

Only after both locks does the adjudicator see both records and establish the final language route. The original records remain immutable.

### 2. Timing, text, and speaker role

`TIMING_A` and `TIMING_B` independently segment every validation item, transcribe intelligible source-language speech without translation, and choose exactly one of:

- `NON_CHILD`
- `CHILD`
- `OVERLAP`
- `UNCERTAIN`
- `NONSPEECH`

The controller rejects every other role. Each pass autosaves drafts and then closes an item irreversibly. The peer pass remains hidden until both passes close. A qualified adjudicator then matches or preserves unmatched segments and locks accepted utterances while retaining both independent sources.

The validation block ends at the first of 300 adjudicated linguistic utterances or 30 adjudicated speech minutes. `NONSPEECH` never contributes. All records contributing to that block have two closed independent passes; a segment absent from one pass remains an observable disagreement rather than being silently dropped. Each adjudicated item is closed explicitly and then leaves the queue, preventing early items from starving later items. Once the first threshold-crossing record is accepted, further timing adjudication is rejected until the referential inventory is frozen.

### 3. Referential coding

Only adjudicated `NON_CHILD` utterances are eligible. After the frozen minimum and item completion, the adjudicator freezes the assignment once. Pass A receives every item. Pass B receives at least 20% selected by deterministic hash order within each opaque metadata stratum, bound to the frozen selection, authenticated opaque item key, and adjudicated timing boundaries. Because selection occurs before referential labels and uses no transcript string, word, visible content, machine result, or apparent groundability, it cannot adapt to referential yield.

Both coders see only the fixed ±5-second window, the accepted source-language utterance, and an opaque key. The application has no machine-proposal display path; therefore the reference condition is fully ASR/VLM hidden, exceeding the frozen minimum 20% ASR-hidden reference requirement. Every coder selects exactly one frozen status:

- `VISIBLE_SINGLE`
- `VISIBLE_MULTIPLE`
- `NULL_NOT_VISIBLE`
- `IRRELEVANT`
- `UNDECIDABLE`
- `UNUSABLE`

They also record `NOUN_OBJECT`, `VERB_ACTION`, `BOTH`, or `NEITHER`; a candidate-count band; and optional candidate boundaries constrained to the fixed window. Boundaries attached to a nonvisible status or outside the window are rejected. Null and ambiguous cases remain first-class records.

The adjudicator sees A and B only after both records lock, preserves both, and stores a separate immutable resolution. Reliability must be calculated from the pre-adjudication records.

## Coder independence and blinding

Each stage has fixed A/B slots. Initialization generates distinct high-entropy tokens, stores only their HMACs in SQLite, and writes plaintext once to an owner-only coordinator packet in quarantine. The app authenticates preassigned tokens and cannot claim a slot. Ordinary task APIs return only opaque display keys, batch number, and the current coder’s task state. Timing coders can retrieve only their own segments. Peer records become queryable solely through the adjudicator functions after the required locks.

A qualified user may complete a first pass, but the second pass must be performed by a genuinely independent authorized human. The first user cannot simulate the second human by using another token. Codex cannot occupy a coder or adjudicator slot.

## Small resumable batches and autosave

The UI serves up to 20 pending records at a time (10 media items for timing adjudication). Every form submission commits immediately with full SQLite synchronization. Locked records cannot be updated. Opaque batch numbers are established in the restricted packet, and progress is reported without lexical text, item keys, identifiers, or paths.

The app reports:

- locked language items per pass;
- closed timing items per pass;
- adjudicated utterance count and speech minutes;
- whether the 300-utterance-or-30-minute threshold is reached;
- referential inventory size and frozen double-code count; and
- locked referential counts per pass.

## Starting the app

Run only the fixed launcher from the local machine:

```bash
python3 scripts/launch_childlens_human_validation_v1_2.py
```

The launcher rejects every argument and opens the only authorized browser window automatically. It places the browser profile, disk/media cache, save-file directory, and download directory under the restricted root while passing only relative store names to the browser from a quarantine-confined working directory. The restricted root is absent from launcher argv, child environments, and the shell command. An ephemeral URL token is checked by the app, and synchronization/background services are disabled. Discovery fails closed if no initialized runtime or more than one candidate exists. Restricted tables are rendered without dataframe download controls. Do not reopen the URL in a general browser, set the legacy root environment variable for app launch, change the bind address, tunnel the port, run through a remote notebook, or enable telemetry. End the app window before another coder session. Authorized coders receive a pass name and pre-generated token through an approved offline route; tokens must not be written in repository artifacts.

## Human effort estimate

The runtime packet provides the actual durations needed for an estimate. Before handoff, calculate:

- language discovery: two independent passes at roughly `media duration × 0.25–0.5` plus adjudication;
- timing/text/role: two independent passes at roughly `speech minutes × 4–8`, depending on language and intelligibility, plus adjudication;
- referential pass A: approximately 1–2 minutes per accepted `NON_CHILD` utterance;
- referential pass B: the frozen double-code count at the same rate; and
- adjudication: roughly 0.5–1.5 minutes per disagreement.

These are scheduling ranges, not evidence. The coordinator must replace them with an actual estimate from the populated packet’s duration, utterance count, and a short non-study training calibration before asking the user to begin. The handoff must clearly state which first pass the user can perform, which second pass requires another independent qualified human, and the expected time for each.

## Validation and prohibited uses

Run:

```bash
python3 scripts/childlens_human_validation_v1_2.py validate \
  --root "$CHILDLENS_V12_QUARANTINE_ROOT"
python3 scripts/childlens_human_validation_v1_2.py readiness \
  --root "$CHILDLENS_V12_QUARANTINE_ROOT"
python3 -m pytest -q tests/test_childlens_human_validation_v1_2.py
python3 -m pytest -q tests/test_launch_childlens_human_validation_v1_2.py
```

The structural validator returns `STRUCTURAL_PASS` only for storage/schema integrity; it is never presented as human completion. The separate readiness validator distinguishes initial `LANGUAGE_PASS_READY`, coding in progress, completed genuine human evidence, and frozen reliability-threshold results. Reliability uses locked pre-adjudication records and reports no text or row keys. Neither result alone satisfies a scientific gate. The operational decisions and examples are frozen in [human_validation_codebook_v1_2.md](human_validation_codebook_v1_2.md).

No instrument code, weight, checkpoint, tokenizer, vocabulary, prompt, feature, embedding, hidden state, score, confidence, raw machine proposal, or unreviewed record may enter learner data. The app does not create a learner store, tokenize text, compute an acquisition result, expose a causal arm, or authorize scientific training. Any later machine-proposal interface requires a separate frozen, licensed amendment and an independent human-hidden reference route; it cannot be switched on inside this base workflow.
