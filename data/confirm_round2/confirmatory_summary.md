# Confirmatory Round 2 Summary

Date: 2026-07-04

## Run Setup

- Fresh dataset: `data/confirm_round2/episodes.jsonl`
- Episodes: 6000
- Episode IDs: 2000000-2005999
- Seeds: 900000123-906053114, stride 1009
- Existing-set disjointness: pass; 0 seed overlaps and 0 episode-ID overlaps against `babyworld-lite/data/honest/episodes.jsonl` and `data/live-demo/episodes.jsonl`
- Eval command: `run_honest_eval.py --episodes data/confirm_round2/episodes.jsonl --out data/confirm_round2/eval --splits held_out_impulse_mass held_out_composition held_out_material --rules decision_rules.yaml --skip-k-sweep`
- Metric: paired MAE improvement on `target_displacement`, computed as no-touch MAE minus touch MAE. Positive means touch lowers forward-prediction error.
- CI procedure: paired bootstrap 95% CI, same implementation as the honest eval runner.

## Leakage Gate

Status: pass.

The eval input was the JSON frame trace dataset, not rendered pixels. The pre-eval audit found 0 `gif_path` records, 0 caption/overlay-like keys, 0 pixel/image-like keys, and no rendered image files in `data/confirm_round2/` before eval. `build_eval_frame` also completed with `render_frame` and `render_gif` patched to raise, confirming the eval frame path did not use the caption-rendering code.

## Results

| Split | Role | n train | n test | Held-out support | No-touch MAE | Touch MAE | Effect: no-touch minus touch MAE | 95% CI | Decision |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| held_out_impulse_mass | Confirmatory | 5230 | 770 | impulse >= q0.75 and mass >= q0.50 | 12.578 | 10.081 | 2.497 | [1.862, 3.175] | PASS |
| held_out_composition | Confirmatory | 5607 | 393 | shape=cup, action=push | 25.184 | 24.573 | 0.611 | [-0.780, 2.093] | FAIL |
| held_out_material | Consistency check only | 4476 | 1524 | material=metal | 18.657 | 18.257 | 0.400 | [-0.060, 0.857] | Null stable; not a decision criterion |

## Pre-registered Decisions

- `confirm_touch_helps_held_out_impulse_mass`: pass. The improvement CI excludes zero in the preregistered positive direction.
- `confirm_touch_helps_held_out_composition`: fail. The point estimate is positive, but the CI crosses zero.

The held-out material consistency check remains consistent with the earlier null: the fresh CI still crosses zero.

## Side Audit: Material and Touch Dynamics

This was read-only; no code was changed for the audit.

Static trace from `babyworld_lite/sim.py`:

- Material definitions provide `hardness`, `mass`, `friction`, `bounciness`, and `tactile_noise` per material (`sim.py:25-30`).
- `sample_object` copies material-derived `mass`, `friction`, `bounciness`, and `hardness` into each object with jitter (`sim.py:89-112`).
- Event logic uses `mass`, `friction`, and `bounciness`, but not the material string and not hardness directly (`sim.py:115-133`).
- Initial contact occurrence is mostly material-invariant: `contact_happens` depends on action miss chance and RNG, and `is_contact` depends on geometry/time/radius (`sim.py:157-158`, `sim.py:203-204`).
- Contact force is directly material-dependent through hardness: `contact_force = intended_impulse * obj.hardness * force_noise` (`sim.py:207-220`).
- Object motion after contact uses material-derived `mass`, `friction`, and `bounciness`, so later contact counts and normals can be indirectly material-dependent (`sim.py:222-249`).

Empirical check on the fresh manifest:

| Material | n | Contact rate | Mean contact count | Mean first contact force, contacts only | Mean hardness |
| --- | ---: | ---: | ---: | ---: | ---: |
| foam | 1443 | 0.909 | 10.805 | 0.240 | 0.250 |
| metal | 1524 | 0.912 | 14.195 | 0.882 | 0.920 |
| plastic | 1477 | 0.913 | 12.519 | 0.524 | 0.550 |
| wood | 1556 | 0.916 | 13.638 | 0.678 | 0.700 |

Conclusion: touch is not fully material-invariant. The binary contact gate is largely material-invariant at initial contact, but the contact-force channel directly carries material through hardness. The event logic itself uses material-derived physical properties rather than the material label.

## Artifacts

- `generation_summary.json`
- `freshness_disjointness_audit.json`
- `leakage_audit.json`
- `material_touch_channel_audit.json`
- `eval/results.json`
- `eval/forward_prediction_report.txt`
- `eval/coverage_report.json`
- `eval/coverage_report.txt`
