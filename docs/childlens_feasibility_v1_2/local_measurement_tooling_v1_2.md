# ChildLens v1.2 local structural-measurement tooling

Version: `childlens-local-measurement-tooling-v1.2.1`  
Audit date: 2026-07-21  
Scope: local restricted container/audio/decode and speech-window structural checks only  
Scientific learner or causal outcome authorized: **no**

## Result

The local FFmpeg/FFprobe structural route is implemented and synthetic-sentinel tested. The auditor can consume the restricted, frozen-pilot measurement manifest inside the existing quarantine, verify every file against its expected byte count and SHA-256, probe container/video/audio structure, perform a full error-fatal decode, compare observed and annotation-derived duration, and validate the bounds of speech-presence windows. It returns only a fixed aggregate receipt with suppression of every nonzero cell below five.

This tooling artifact itself did not read ChildLens media, annotations, manifests, identifiers, or lexical content. It is engineering readiness, not empirical evidence and not a pass for audio, transcription, speaker-role, or referential gates. The coordinator may run it only after native acquisition and restricted manifest finalization.

The implementation is [audit_childlens_local_media_v1_2.py](../../scripts/audit_childlens_local_media_v1_2.py). The local-instrument inventory is [audit_childlens_local_instruments_v1_2.py](../../scripts/audit_childlens_local_instruments_v1_2.py), and the aggregate readiness receipt is [measurement_tooling_readiness_receipt.json](../../output/childlens_feasibility_v1_2/measurement_tooling_readiness_receipt.json).

## Restricted input contract

The manifest must remain owner-only inside the restricted quarantine and must use exactly this schema:

```json
{
  "schema_version": "childlens-restricted-measurement-manifest-v1.2.1",
  "pilot_selection_sha256": "<64-lowercase-or-uppercase-hex-characters>",
  "items": [
    {
      "blinded_item_key": "<restricted-opaque-key>",
      "media_relpath": "<restricted-relative-path>",
      "expected_size_bytes": 1,
      "expected_media_sha256": "<64-hex-characters>",
      "reference_duration_seconds": 1.0,
      "reference_duration_basis": "AUTHORITATIVE_MEDIA_DURATION",
      "speech_presence_expectation": "PRESENT",
      "speech_windows": [
        {"start_seconds": 0.0, "end_seconds": 1.0}
      ]
    }
  ]
}
```

The displayed values above are schema placeholders, not ChildLens observations. The production manifest must contain exactly the frozen 12–18 item sample; v1.1 fixes 15 items. `speech_presence_expectation` must be exactly `PRESENT` or `ABSENT`, as fixed by the restricted annotation linkage: `PRESENT` requires at least one structurally valid window, while `ABSENT` requires an empty window list. Unknown fields fail closed. Transcript text, speaker labels, participant/session identifiers, source locators, URLs, cookies, credentials, acquisition-effect values, and instrument output are not accepted.

The auditor requires:

- a real owner-only quarantine directory outside the repository;
- an owner-only manifest and owner-only regular media files;
- no symlink in the manifest or media path chain;
- relative paths contained by the quarantine;
- exact local byte count and SHA-256 agreement before a media object is opened;
- the frozen pilot-selection digest; and
- no more than 18 objects.

A mismatch is aggregated as an integrity failure and the mismatched object is not probed or decoded.

## Fixed structural measurements

Each integrity-matched object is inspected sequentially with locally pinned FFprobe and FFmpeg. FFmpeg protocols are restricted to local/embedded file mechanisms; network protocols are absent from the allowlist. Standard error is discarded, FFprobe JSON is parsed only in memory, and no subprocess output or command path is copied into report artifacts.

The fixed measures are:

1. manifest byte-count and SHA-256 agreement;
2. successful FFprobe parse;
3. presence of at least one video stream;
4. presence of at least one audio stream;
5. duration-reference validity: container duration within the larger of two seconds or 2% of an authoritative duration, or containment of an annotation-derived maximum endpoint within the container;
6. audio-stream duration within the larger of two seconds or 2% of container duration when stream duration metadata is exposed;
7. complete error-fatal video/audio decode to the null muxer; and
8. satisfaction of each selected object's explicit speech-presence expectation: `PRESENT` requires at least one valid window and `ABSENT` requires zero windows; every supplied start must be nonnegative, every end must be greater than its start, and every end must remain within the probed media duration plus a fixed boundary tolerance.

The full decode does not write a second media copy, decoded audio, frame, thumbnail, waveform, transcript, or log. It does not infer speech, language, speaker role, or visible reference. Annotation-derived speech windows remain restricted input and their exact boundaries are never exported.

## Aggregate export and cell suppression

The public receipt schema is fixed. It may contain the already permitted pilot-selection digest, a digest of the exact restricted measurement manifest, total selected-media count, tool binary digests, aggregate metric states, and Boolean boundary assertions.

For a binary metric, all-pass and none-pass counts may be reported because the complementary cell is exactly zero. If both outcomes occur and either nonzero cell is below five, both counts become `null` with `MIXED_SMALL_CELL_SUPPRESSED`. Window totals below five are suppressed. No item-level result, path, filename, identifier, exact timestamp, duration, file size, hash, transcript, codec inventory, frame, audio, or small nonzero cell can enter the receipt.

The duration basis must be exactly `AUTHORITATIVE_MEDIA_DURATION` or `ANNOTATION_MAX_END`. The latter establishes only that the annotation timeline fits inside the media; it is not full recording-duration agreement. The receipt separately reports aggregate authoritative-reference coverage.

The receipt status `ALL_STRUCTURAL_CHECKS_PASS` requires all integrity/container/audio/duration-reference/decode checks, all per-item speech-presence window expectations satisfied, and no invalid window. The receipt exposes explicit all-item aggregate evidence for audio-stream presence, complete decode, duration consistency, corruption absence (successful probe plus error-fatal full decode), and window-expectation satisfaction. This status remains a structural engineering result. It is not human validation, does not establish actual language, and cannot pass the frozen lexical or referential gates by itself.

## Fail-closed error behavior

Command-line errors return only the schema version plus a fixed `E_*` code. Exceptions, paths, filenames, identifiers, values, and subprocess text are suppressed. Rejected conditions include:

- quarantine inside the repository or with group/other permissions;
- manifest or media symlinks, traversal, wrong permissions, or out-of-root paths;
- missing, malformed, oversized, or unknown-field manifests;
- bad selection or media digests;
- bad file sizes, durations, windows, item counts, or configuration; and
- missing/unexecutable FFmpeg tools or output exceeding the fixed size bound.

## Controlled production invocation

The coordinator should run a single process after acquisition, substituting local restricted paths without placing them in shell history, repository logs, commentary, or reports:

```text
python3 scripts/audit_childlens_local_media_v1_2.py \
  --quarantine-root <RESTRICTED_ROOT> \
  --manifest <RESTRICTED_MEASUREMENT_MANIFEST> \
  --ffprobe <PINNED_FFPROBE_BINARY> \
  --ffmpeg <PINNED_FFMPEG_BINARY>
```

The raw standard output must first be captured inside quarantine. Only after schema/privacy validation may the aggregate receipt be copied into `output/childlens_feasibility_v1_2/`. Media are processed sequentially. A later separately reviewed orchestration layer may use two CPU workers, but it must preserve aggregate-only output and cannot duplicate full-resolution media.

## Validation

Nineteen focused tests passed on 2026-07-21. They cover a synthetic one-second audio/video container, both duration-reference bases, explicit `PRESENT`/`ABSENT` window expectations and mismatch rejection (including a present-but-invalid window), full decode, aggregate-only output, small-cell suppression, exact-byte/hash checks, unknown-field rejection, traversal, symlinks, owner-only permissions, repository-boundary rejection, fixed errors, installed-package routing, and the instrument-to-learner firewall.

The synthetic sentinel contained no ChildLens data. No model, weight, dataset, license acceptance, authenticated browser, email, external API, learner, tokenizer, or causal arm was used.

## Limitations and next use

- No restricted media audit has yet been run by this workstream; the production aggregate receipt must be generated after acquisition.
- Some containers may omit per-stream duration even when full decode succeeds. The receipt reports that conservatively as missing audio-duration consistency evidence; it does not reinterpret metadata absence as corruption.
- Speech-window validation checks structure only. It cannot establish whether a window contains intelligible speech or whether annotation timing is accurate.
- Language, transcript correctness, speaker role, and visible referential status require genuine qualified humans under the frozen blinded workflow.
- No ASR/alignment runtime or weights are currently local. The human-only route remains valid and controlling.
