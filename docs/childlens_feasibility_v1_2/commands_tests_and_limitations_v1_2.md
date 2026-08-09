# ChildLens v1.2 commands, tests, and unresolved limitations

## Deterministic commands

```bash
python3 scripts/launch_childlens_native_transfer_v1_2.py
python3 scripts/run_childlens_post_acquisition_v1_2.py
python3 scripts/synthesize_childlens_terminal_v1_2.py
python3 scripts/validate_childlens_feasibility_v1_2.py
python3 scripts/validate_childlens_immutability_v1_2.py
python3 -m pytest -q tests/test_*childlens*v1_2*.py
node --test tests/test_childlens_keeper_redacted_extractor_v1_1.mjs
node --test tests/test_childlens_keeper_restricted_bounds_builder_v1_2.mjs
node --test tests/test_childlens_keeper_selected_admission_extractor_v1_2.mjs
```

The synthesizer validated fixed repository-safe receipt schemas, immutable v1/v1.1 baselines, scientific boundary flags, the acquisition cap, 15-item cross-checks, the instrument firewall, honest candidate-unit semantics, and terminal prerequisites before writing. Its tests are synthetic and never read restricted data. Output replacement is staged and all-or-rollback, with the decision record installed last.

Final results were 158 passing Python tests and 21 passing JavaScript extractor tests (9 redaction, 4 conservative-bound, and 8 selected-admission tests). The production feasibility validator passed, and the historical immutability validator reproduced the frozen v1 and v1.1 artifact counts and digests exactly. Native transfer completed for all 15 frozen objects; post-acquisition preparation completed; and terminal synthesis completed without changing selection, permissions, or scientific thresholds.

## Provenance and privacy boundary

The evidence ancestry is the accessible ChildLens release plus simulator-defined counterfactual modalities only. No AEA or BabyView empirical data, aggregates, priors, vocabulary, tokenizer, checkpoint, weights, or results entered the artifacts. Public methodological references are not empirical ancestry. Raw ChildLens payloads and restricted derivatives remain in the owner-only local quarantine.

The category-only incident receipt is `RESTRICTED_QUARANTINE_PATH_APPEARED_IN_LOCAL_TOOL_LOG_REDACTED_FROM_REPOSITORY_AND_USER_FACING_ARTIFACTS_NO_CORPUS_PAYLOAD`. The literal quarantine path is intentionally absent. No corpus payload was exposed in tool output, and no restricted payload is present in the repository.

## Limitations and exact next task

Automated structural success cannot establish language, transcription accuracy, speaker role, lexical recurrence, visible reference, ambiguity, lag, held-out exposure, or reliability. Official speech-presence windows are not utterances. Genuine independent human work is never inferred from model or Codex agreement.

Exact next task: `AUTHORIZED_LANGUAGE_MATCHED_HUMANS_COMPLETE_TWO_INDEPENDENT_PASSES_AND_ADJUDICATION_THEN_RERUN_SYNTHESIS`.
