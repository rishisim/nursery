# AEA visible-action v2 implementation clarifications

**Recorded:** 2026-07-14T23:58:00Z  
**Parent protocol:** `aea-visible-action-v2`  
**Ordering:** before new v2 RGB evidence, annotations, split results, or model results were viewed.

These details resolve implementation choices without changing the frozen sample, labels, thresholds, budgets, gates, or decisions.

1. The 48-row carry-forward source is the v1 **prelabel** manifest, whose `audit_label` and `rationale` fields are null. V1 completed labels are never read by a packet builder or reviewer.
2. `judgeable` means `observable_action != uncertain`. `none_visible` is judgeable. All rates use the full 72-row denominator unless explicitly named per-pass.
3. Confidence-weighted exact agreement uses row weight `min(weight(pass_a), weight(pass_b))`, with high = 1.0, medium = 0.75, and low = 0.5. It is descriptive; frozen gates use unweighted exact agreement and unweighted Cohen's kappa. A constant-label kappa is null and fails its gate.
4. A modeled consensus row requires exact action agreement on a frozen modeled label and medium-or-high confidence in both passes. Agency, temporal, and ASR-referent disagreements do not alter its action label but remain reported.
5. The constrained split is solved with SciPy `milp`, integrality on every group-to-fold assignment, exact six-group folds, and deterministic input order. Solver status `infeasible` is the certificate; no alternate seed, solver, class subset, fold count, or relaxed constraint is tried. If feasible, fixed objective scaling makes the three declared objectives lexicographic, and assignments are independently revalidated.
6. Donor feasibility is checked both by the complete-multipartite half-size theorem and by deterministic augmenting-path bipartite matching. A disagreement between the two is an integrity failure.
7. The action-head control is `VideoEncoder(hidden_dim=64, embedding_dim=64)` followed by one linear classifier. Cross-entropy uses inverse class-frequency weights normalized to mean one. Each epoch visits every retained row once in a seed-fixed shuffled order; there is no augmentation, early stopping, tuning, or rerun.
8. Fixed transcript-control prompts are: `locomotion_posture` → “The visible wearer is walking or changing posture.”; `reach_grasp` → “The visible wearer is reaching for or grasping an object.”; `transport_place` → “The visible wearer is carrying or placing an object.”; `state_change_operate` → “The visible wearer is changing or operating an object or device.”; `food_material_handling` → “The visible wearer is handling food or other material.”; `clean_groom` → “The visible wearer is cleaning or grooming.” The tokenizer is fit once to training transcripts plus these prompts.
9. The transcript control uses the same 12 dense-frame indices, image size, encoder dimensions, batch size, learning rate, and 120 epochs as the action head. Its contrastive target is each row's natural transcript; evaluation scores the fixed prompts. It remains descriptive and non-gating.
10. For the conditional IMU diagnostic, macro chance is `1 / number_of_retained_labels`. The paired donor condition replaces each test row's entire IMU trajectory with its frozen test-side donor while preserving the target label and trained classifier. This test-side shuffle is relevant only to the IMU-only diagnostic; it is absent from motor-withheld grounding evaluation.
11. Train-minus-held-out gap is mean fold training balanced accuracy minus pooled group-held-out balanced accuracy. Cluster bootstrap resamples event groups and retains all rows from sampled groups. Plus-one corrected p-values are used throughout.
12. The safe release manifest is parsed only by `load_safe_manifest`, which discards signed URLs at the boundary. V2 artifacts contain aggregate declared sizes and sequence IDs only, never URLs.
