# Commands, validation, and unresolved limitations

## Executed

- Ran the public/synthetic Qwen3-ASR plus forced-aligner canary under OS network denial: PASS.
- Ran the public/synthetic five-frame Qwen3-VL canary under OS network denial: PASS.
- Ran `python scripts/nursery_pseudo_calibration.py` using the frozen restricted sample: terminal calibration status `CALIBRATION_REVISE`.
- Ran the single frozen Qwen2 schema-only engineering retry: completed, schema gate still failed.
- Ran the convergence validator with the aggregate receipt: PASS, zero issues.
- Ran the new synthetic/public test suites: 47 passed.
- Compiled the three new coordinator/worker scripts successfully.
- Confirmed no restricted media-type file or absolute workspace path exists in the new convergence docs/output namespace.

## Limitations

1. No qualified German human evidence exists. All ChildLens lexical and referential values are model hypotheses, never truth.
2. The two ASR paths disagree substantially on boundaries and text. No path is named as reference.
3. Qwen2 failed exact-schema output before and after its only permitted correction, so a two-model visual calibration envelope cannot be formed.
4. Qwen2 and Qwen3-VL share vendor lineage even if the failed path were usable; their dependence would still weaken triangulation.
5. The later confidence-sharpening synthetic lineage remains terminally stopped and contributes no acquisition evidence.
6. The reused positive v1 result is development-only, action-word only, canonical-symbolic, and has no noun endpoint or confirmation run.
7. The legacy v1.3.1 validator still flags one pre-existing absolute workspace path in an execution-design memo. Historical v1-v1.3 digest sets remain exact; the file was not edited because this task requires immutable preservation. The new namespace passes its own privacy validator.
8. BabyView access/governance remains uninitialized and does not block public/synthetic fixture work; it does block any future BabyView-only empirical study.

No learner, tokenizer, scientific checkpoint, causal arm, or acquisition-effect outcome was run.
