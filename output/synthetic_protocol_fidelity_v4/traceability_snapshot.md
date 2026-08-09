# V4 protocol-to-implementation traceability matrix

| Protocol item | Code | Tests | Qualification artifact |
|---|---|---|---|
| Package-only claim and terminal rule | `study.py:terminal_decision` | terminal decision tests | `terminal_decision.json`, `independent_validation_report.md` |
| Four disjoint registries and prior-seed firewall | `protocol.py:load_config`, `SeedFirewall` | registry overlap, bypass, all-operation adversarial tests | `frozen_seed_registries.json`, each run's `seed_operation_log.json` |
| Actual frozen v2 detector runtime | `detector_runtime.py:DetectorRuntime` | wrong hash, hash-only, monkeypatched-call, call-count tests | `detector_runtime_audit.json`, derived evidence ledgers |
| Action-selective sensor/noun independence | `corpus.py:generate_corpus`, `audit_corpus`; `learner.py` noun sensor weight contract | v3-like noun coupling adversarial test, exact factorial audit | `corpus_audit.json`, `noun_selectivity_audit.json` |
| Exact action relation-cell balance | `corpus.py:_relation_candidates`, `audit_corpus` | exact per-concept count test, imbalance perturbation | `corpus_audit.json` |
| Joint competition, iterations, exclusivity objective | `learner.py:fit_joint_competitive` | weight/iteration participation and co-word perturbation tests | `model_results.json`, `competition_audit.json` |
| No primary oracle leakage | `protocol.py:reject_oracle_fields`; `learner.py` fit/predict signatures | injected pointer rejection; key permutation leaves predictions unchanged | `leakage_audit.json` |
| Visibility factor changes observations | `corpus.py:_observation` | factor perturbation/inert-factor adversarial tests | `factor_results.json`, `factor_recomputation.json` |
| Lag/speech time changes updates | `learner.py:_lag_weight` | factor perturbation/inert-factor adversarial tests | `factor_results.json`, `factor_recomputation.json` |
| Level-stratified factor analyses | `study.py:factor_analysis` | pooled-copy adversarial test and exact recomputation | `factor_results.json`, `factor_recomputation.json` |
| Sensor-free non-degenerate gate | `study.py:qualification_gates` | fixture baseline bounds test | `qualification_summary.json` |
| Real oracle control | `controls.py:fit_oracle_alignment` | corrupt oracle target positions lowers score | `control_results.json` |
| Real direct-capacity control | `controls.py:fit_direct_capacity` | corrupt observations/labels lowers score | `control_results.json` |
| Degenerate/non-degenerate inference | `analysis.py:infer_mean` | point-mass and studentized branches | `inference_audit.json` |
| True isolated reproduction | `study.py:run_isolated_reproduction`, `compare_core_artifacts` | missing execution/file mismatch adversarial tests | `reproduction_comparison.json`, two per-run core manifests |
| Pre-freeze receipt and snapshots | `study.py:freeze_package` | receipt/hash verification | `freeze_receipt.json`, `frozen_source_snapshot/` |
| V1-v3 preservation | `study.py:verify_preservation` | modified/missing/added baseline path test | before/after manifests and `preservation_proof.json` |
| Complete artifact manifest | `study.py:write_complete_manifest` | per-file shape/path/hash test | `complete_file_manifest.json` |

