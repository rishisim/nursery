# Gemma 4 restricted calibration success-path compatibility erratum v1.6.1

Status: `FROZEN_PRE_RESTRICTED_RERUN_SUCCESS_PATH_ONLY`

The frozen causal execution contract v2 names `output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json` as the aggregate calibration receipt consumed by the future one-shot causal workflow. That planned file has never been created. The v1.6 environment amendment inadvertently named a new success receipt inside its diagnostic namespace.

This additive erratum preserves the v1.6 amendment byte-for-byte and supersedes only its future success-path field. A validated successful calibration must exclusive-create the original planned path with owner-only permissions; it must fail closed if that path already exists and must never overwrite, truncate, replace, or reuse a historical file. The namespaced v1.6 success filename is not created.

Failures and diagnostics remain under `output/nursery_program_convergence_v1/gemma4_restricted_worker_environment_v1_6/`, including `childlens_pseudo_calibration_gemma_substitution_failure_v1_6.json`. No failure artifact may occupy the planned success path.

Receipt schema and content, worker environment, quarantine handling, model, sample, prompt, parser, gates, thresholds, no-fallback rule, scientific boundary, and execution contract are unchanged. This erratum executes nothing and authorizes neither restricted inference nor a scientific endpoint.
