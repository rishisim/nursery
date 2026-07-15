# AEA visible-action rescue feasibility v2

## Decision: REVISE

non-severe frozen annotation stability threshold miss.

This authorizes: one prospectively frozen ontology/interface repair only. It does not authorize reserve access, the locked experiment, downloads, licenses, or outreach.

AEA is an adult, partly scripted sensor-format analogue. These are not developmental findings and are not BabyView-like. Model-assisted labels are development diagnostics, not human annotations or human reliability.

## Dense blinded annotation

The frozen sample contained 72 windows from all 18 development event groups. Each used 31 ordered RGB frames (2,232 queries total). Reserve RGB and IMU accesses were both zero.

Observable-action exact agreement was 81.9% (kappa 0.7770897832817337). Modeled consensus was 38/72 (52.8%). The annotation gate failed; severe failure was False.

Retained modeled support: food_material_handling=9 windows/4 groups, locomotion_posture=10 windows/7 groups, state_change_operate=11 windows/6 groups.

## Split, capacity, transcript, and IMU

The constrained split/donor status was `not_run_annotation_gate_failed` (gate False); no relaxation or second split was attempted.

The supervised action-head capacity status was `not_run_stage_gate_failed` (gate False; mean training balanced accuracy not run).

The separate natural-transcript control status was `not_run_action_head_capacity_not_passed` and was non-gating. It was not run after the frozen stage gate.

The conditional IMU status was `not_run_stage_gate_failed` (gate False). No IMU array was opened after the stage gate failed.

## Additional recordings and storage

Recommendation: **do not use extra storage**. The preregistered ontology/capacity/support-or-power gate did not pass; no expansion is justified.

The safe release has 143 recordings. The remaining 103 annotations+main-VRS components total 181.388 GiB; available space at decision time was 193.832 GiB. Free space is a ceiling, not a reason to acquire data.

## Limitations

- Two context-isolated model passes do not estimate human inter-rater reliability.
- Thirty-one frames are dense temporal evidence, but not continuous video or audio.
- V2 is development/protocol feasibility; no confirmatory effect was tested.
- The v1 reserve is prospective from v1 onward, not pristine relative to the earlier smoke run.
- Threshold proximity cannot override the frozen mechanical decision.
