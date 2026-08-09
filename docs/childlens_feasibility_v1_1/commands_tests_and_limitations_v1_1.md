# ChildLens v1.1 commands, tests, and unresolved limitations

Version: `childlens-commands-tests-limitations-v1.1.0`  
Date: 2026-07-21  
Scientific outcome executed: no

## Commands rerun for final handoff

All commands below are local and operate only on repository code, reportable aggregate receipts, or synthetic fixtures. None accesses Keeper, email, quarantine, ChildLens payload, AEA, or BabyView.

| Command | Result |
| --- | --- |
| `node --test tests/test_childlens_keeper_redacted_extractor_v1_1.mjs` | PASS: 9 tests, 0 failures. |
| `python3 -m pytest -q tests/test_childlens_feasibility_v1.py tests/test_childlens_restricted_manifest_v1_1.py tests/test_audit_childlens_participant_table_v1_1.py tests/test_audit_childlens_annotations_v1_1.py tests/test_prepare_childlens_restricted_manifest_input_v1_1.py tests/test_validate_childlens_feasibility_v1_1.py` | PASS: 48 tests. |
| `python3 scripts/validate_childlens_feasibility_v1.py` | PASS: v1 structural, privacy-pattern, ancestry, and decision checks. |
| `python3 scripts/validate_childlens_feasibility_v1_1.py` | PASS: v1.1 required-artifact, cross-receipt, historical-v1 immutability, release-binding, frozen-preselection, quarantine-scope, and privacy checks. |
| `git diff --check` | PASS. |

The final handoff therefore includes 57 passing tests: 9 Node extractor sentinels and 48 Python tests. The extractor verifies version `childlens-keeper-redacted-extractor-v1.1.7` fail-closed behavior, including exact key allowlists, constant failures, serialized-function isolation, synthetic-sentinel non-disclosure, and the output-size guard. The Python suites verify deterministic canonicalization and selection, resource ceilings, linkage/grouping rejection paths, cell suppression, live/snapshot binding states, restricted-output path refusal, atomic mode-`0600` output, v1 immutability, and v1.1 cross-receipt consistency.

Other preflight activity is documented in its source receipts rather than reconstructed here: synthetic FFmpeg repeatability, local runtime import, host-capacity probes, permission correspondence review, and redacted release observation. This record does not invent shell commands for actions whose literal invocation was not preserved.

## Execution disposition

- G1 was resolved from an actual signed agreement plus approval and access correspondence, with controlling noncommercial, institutional, secure-storage, no-sharing, aggregate-only, citation, and July-2027 review/deletion constraints.
- The public DOI snapshot identity and a live redacted observation were receipted. Live-to-snapshot exact-byte equivalence remains `NOT_PROVEN`.
- The quarantine was created outside the repository with owner-only directory/file modes and exact-root content-indexing and Time Machine exclusions.
- Only the participant table and final annotations were acquired into quarantine.
- HTML returned by invalid staging download routes was detected, rejected, and unlinked; it was never admitted as dataset content.
- No video, audio, decoded waveform, frames, transcripts, speaker labels, referential labels, ASR output, measurement features, learner data, checkpoint, training, evaluation, or causal result exists from v1.1.
- No commit, push, publication, email send, Keeper share change, or license acceptance occurred.

## Unresolved limitations and their consequences

| Limitation | Evidence status | Consequence |
| --- | --- | --- |
| Exact live-to-DOI byte equivalence | Not proven; checksum-complete multiset comparison unavailable | Do not claim the accessible live bytes equal the DOI snapshot. Bind any future pilot to an immutable source or a deterministic restricted manifest plus observation receipt, and verify each selected object locally. |
| Canonical media manifest | Metadata classes are available, but no selected media bytes were acquired/hash verified | Frozen media selection and the 20-GiB admission gate cannot be completed from this run alone. |
| Actual language(s) | Unobserved because no audio/human review occurred | Do not choose or validate ASR, aligner, tokenizer, or language-specific annotation staffing yet. |
| Audio and utterance timing | No media decode | G3 remains unresolved. |
| Child versus other-person speech | No human auditory labels; automated diarization not run | G4 remains unresolved; role accuracy cannot be claimed. |
| Non-child input and recurring lexical material | No validated transcript or human lexical inventory | G5 remains unresolved. |
| Visible object/action, null, ambiguity, and lag annotation | No media/human referential judgments | G6 remains unresolved. |
| Held-out participant/session noun/action exposure | Participant/date grouping is structurally available, but lexical exposure is not measured | G7 cannot pass. |
| Strong ceiling and weak baseline | Protocol mechanics are predeclared; no empirical pilot validates their measurement inputs | G8 remains design-ready only. |
| Simulator calibration | Only permitted nonidentifying aggregates may be used; physical side streams remain absent | G9 must retain simulator-defined IMU/contact/proprioception/motor modalities. |
| Resource envelope | Low case passes, base is narrow/conditional, high case fails at the recorded storage snapshot | Fresh exact-byte and expansion checks are required before every future shard. |
| Human reliability | No genuine language-matched human validation | Codex cannot close this gate; a blinded human packet must be completed and scored. |
| AEA no-read compliance | One overbroad search matched two nonempirical environment fields | Record as a strict-scope procedural exception; matched values were unused and create no empirical ancestry. |

## Fail-closed handoff

The next execution may proceed only from the frozen metadata/hash selection and only after the selected media source has an admissible immutable or deterministic restricted-manifest binding. Acquire one selective shard at a time, verify size and SHA-256 locally, establish language through qualified human review, then populate the blinded auditory/referential validation packet. If exact media acquisition or qualified human validation cannot be completed, the feasibility decision must remain bounded rather than treating automated or structural evidence as a pass.
