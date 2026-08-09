# V4 pre-freeze fixture repair log

All attempts used only fixture corpus seed `140001`, model seed `140101`, and inference seed `140151`. They are excluded from every outcome registry. No v4 excluded-smoke, future-development, or confirmation identifier was touched while making these repairs. Every attempt is retained under `output/synthetic_protocol_fidelity_v4/`.

1. `fixture_validation/` — FAIL. All gates except `sensor_free_non_degenerate` passed. The absent action score was 0.0 because the first joint objective incorrectly made the co-referring primitive and manner words compete for different events.
2. `fixture_validation_repair1/` — FAIL. Co-referring component words used a shared candidate-normalized responsibility, but the exactly balanced design left two globally equivalent manner mappings and random initialization selected the inverted mapping.
3. `fixture_validation_repair2/` — FAIL. Symmetric initialization alone did not prevent later iterative numerical asymmetry from selecting the inverted global manner mapping.
4. `fixture_validation_repair3/` — PASS. The no-sensor condition explicitly projects the unidentifiable manner equivalence class to its symmetric distribution. This consumes no target or semantic label, preserves learned primitive/noun mappings, produces the predeclared non-floor/non-ceiling 0.5 action baseline, and prevents an arbitrary seed-selected inversion from masquerading as learning.
5. `fixture_validation_final/` — PASS. Exact rerun after adding strict policy validation and snapshot-discoverability hardening; deterministic core results match repair 3.

No change was made after an excluded-smoke result. The final frozen code/config/protocol correspond to attempt 5.
