# ChildLens v1.3.1 validation and limitations

## Validation completed

- The loopback author app was stopped; no listener remained on its local port.
- The private workflow disposition was verified aggregate-only: invalid status sealed, gate eligibility false, predictions unrevealed, author lock false, comparison false, and ten blocking triggers installed.
- The aggregate language diagnostic validated all 15 frozen selected items and 912 official candidate windows, applied the predeclared 80% item/window/duration dominance rule, and exported no item language or lexical content.
- A generated eight-second audiovisual source was physically clipped to a two-second audit object. Duration verification and the corrected player API passed; the player received only the bounded object and no start/end seek hints.
- Qwen3-ASR/aligner and Gemma 4 E4B/12B public synthetic smokes ran with macOS network denial and telemetry/offline environment flags. No restricted path was discoverable from either runner.
- The v1.3.1 fail-closed validator reported zero issues across required artifacts, privacy patterns, scientific boundaries, JSON validity, and the v1/v1.1/v1.2/v1.3 historical digests.
- Regression suite: 150 tests passed and one existing opt-in test was skipped. Python compilation and `git diff --check` passed.

## Commands run

```text
python3 scripts/disposition_childlens_author_blocker_v1_3_1.py
python3 scripts/summarize_childlens_language_diagnostic_v1_3_1.py
python3 scripts/validate_childlens_language_blocker_v1_3_1.py
pytest -q tests/*v1_3.py tests/test_validate_childlens_language_blocker_v1_3_1.py tests/test_childlens_language_blocker_v1_3_1.py tests/test_smoke_gemma4_mlx_vlm_v1_3_1.py
python3 -m py_compile scripts/*v1_3_1.py
git diff --check
```

The public model runtime commands used pinned local snapshots under `/usr/bin/sandbox-exec` with `(deny network*)`. Only self-generated German TTS, generated frames, and public audio were processed.

## Unresolved limitations

- German predominance is an offline model diagnostic, not a human census of the release languages.
- No qualified German human transcript, role, or referential label exists. Inter-human reliability and model–human agreement are unavailable.
- The original v1.3 player defect invalidates that attempt even independently of language competence.
- Qwen3-ASR and both Gemma candidates were evaluated on one easy synthetic German sentence; the measurements establish local plumbing, not ChildLens-domain accuracy.
- Gemma E4B emitted numerical warnings in the audio feature extractor. The 12B conversion produced a poor transcript on the same fixture. Conversion-license metadata still needs reconciliation before any restricted-use proposal.
- Qwen3-VL-8B, Qwen3.6, and full Whisper large-v3 were evaluated from official evidence but not executed. Qwen3.6 was ruled out as a default by role and host fit; no 27B/35B weight was downloaded.
- Public model caches leave approximately 76 GiB free, still above the 50-GiB floor. No additional large candidate should be retained without first removing rejected public bakeoff caches.
- No learner, corpus tokenizer, causal arm, or scientific acquisition outcome was run.
