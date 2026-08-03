# Unity-Native Procedural Prompt-to-Embodied-Episode Gate

Decision: **NO-GO for this bounded implementation (2026-08-03).** This is not a claim that Unity ArticulationBody is universally incapable. The bounded blocker is the current imported MPFB anatomical articulation/drive-frame mapping: neither the Unity dense-Jacobian controller nor the permitted deterministic coordinate-descent fallback reached a collision-free wrist waypoint closely enough to begin tactile grasping.

## Preserved evidence

- Origin: `9524287`, preserving the Unity visual audition PASS at `537d30c` and Unity–MuJoCo Stage-C NO-GO at `9524287`.
- Stage A: API/manual-step/contact-data PASS only. A free supported target stabilized without initial overlap; five callback/geometric digit contacts were observed later; maximum penetration was 2.284 mm; repeated traces were identical. It is not lift evidence.
- Stage B: weighted CC0 MPFB skin registration and body-derived POV PASS. Collider-to-skin median/p95/max were 0.903/2.862/6.364 mm; 68° was frozen after the same-trace 60/68/75 comparison; camera clearance was 44.756 mm and roll -1.318°.
- Stage C compiler/reachability contract: PASS. Three prompts deterministically emit explicit transforms, yaw-aware world-AABB receipts, computed support gaps, per-obstacle sightline intersections/clearances, approach/avatar scopes, and a measured right-hand reach profile. Negative tests reject blocked sightlines, tabletop overlap, and out-of-envelope targets. This is a contract/compiler pass, not manipulation evidence.

| Stage | Result | Scope |
|---|---|---|
| A | PASS | ArticulationBody API, 240 Hz manual stepping, callbacks/geometric contact, repeat trace |
| B | PASS | One weighted skin, collider registration, body-derived POV, 60/68/75 evidence |
| C | PASS | Prompt compiler and concrete scene/reachability contracts |
| D | NO-GO | Fresh-process controller calibration and clean free-object replay do not reach/contact |
| E / full system | NOT RUN | No hero/depth/ID/robustness suite after the Stage-D stop |

## Bounded controller result

The clean replay used Unity 6000.0.80f1 ARM64, manual PhysX stepping at 240 Hz, a free target, one weighted skin, no target pose writes, no attachments, and an empty assistance ledger.

- Initialization: 0 finger penetration, 0 stabilization callbacks, 0 target drift, 0 support penetration.
- Dense Jacobian: 186 rows × 26 columns; selected arm/wrist columns `[1,2,3,4,8,9,10]`; damping 0.08. The direct DLS mapping was finite but diverged for the imported drive frames.
- Lifecycle repair: callbacks are cleared immediately before `Physics.Simulate`; prior-step tactile state commands the next step; current callbacks update dwell/stop state only after simulation.
- Fresh-process coordinate-descent fallback: 52 separate Unity invocations, fixed 480-step horizons, one ±20° pass plus one ±5° pass, and identical initial state hash `e0bd443813c1aba1795b2e0ceae4a33b1f2ed6ea03341255915f36efd1b57d3a`. Every accepted improvement was reproduced once from another fresh start. Error improved from 214.419 mm to 118.400 mm, still outside the frozen 30 mm pre-contact gate.
- Clean replay outcome with those frozen targets: minimum fingertip surface distance 198.624 mm, 0 digit contacts after stabilization, no qualified grasp, 0 lift. The larger replay miss reflects dynamic ramp/settling and does not weaken the already-failed fresh calibration result.
- Callback lifecycle is correct in the retained replay: reset immediately before `Physics.Simulate`, commands use persisted prior-step tactile state, and callbacks/impulse/dwell are consumed after simulation.
- Quarantined: `playroom_trial4` (initial three-digit overlap, 0.1535 N·s step-0 impulse, 0.306 m drift/fall); `arm_coordinate_descent3` and its replay (continuing-state/path-dependent optimizer); all pre-lifecycle-fix callback qualifications.

## Immutable ignored-run receipts

- `stage_d/jacobian_trial1/jacobian_controller_trace.json` — SHA-256 `104ce6fee349da4963a64670a62fb9ca9038065982e24df0c42b1c163e1ae555`
- `stage_d/jacobian_trial3_lifecycle_fixed/jacobian_controller_trace.json` — SHA-256 `c11e36c3f5cfc89e3a9af88020a0c50e98e9ec20d3cd65a5dd4f77324a8c614b`
- `stage_d/fresh_process_calibration/fresh_arm_calibration.json` — SHA-256 `0e051bc28dc50dca5e4ee7cb9b3817bcbb4f368fc770746ee9818c16323c078a`
- `stage_d/fresh_calibration_clean_replay_capture/episode_trial_report.json` — SHA-256 `9257c53a7184329b94e73df02a8ae2219540a30c56af1d62bc13dbb221f9a88f`
- `stage_d/fresh_calibration_clean_replay_capture/episode_trial_trace.json` — SHA-256 `70c414c7fbc52e95989881422d6766bc0b6a8cd607531dae7de38ba17f766921`
- Head failure video — SHA-256 `833d0caded03978299ac59c7ad0a888e5f66a7af3f97002f174b3873987833fc`
- External failure video — SHA-256 `ca5a9fe664a15bc5221210e0b18da08923079445048103fba17d0241485b2437`
- Head/external 0.5 s-cell montages — SHA-256 `c7806a0e16cfc21f326016940ba4d801efb894ae3012884182b424b14b43a869` / `94c56c82e06dea1bd94627d46e6a3915254e2cb3bba369f0b62196816ea3eaa0`

## Consequences

Stages D and E do not pass. The two 15 s, 1920×1080, 30 fps videos are explicitly failure diagnostics captured from the authoritative clean replay; they are not furnished-room hero footage. Their 30-cell montages are row-major at 0.5 s per cell (0.0–14.5 s). A replay collider-overlay video was not generated; the accepted Stage-B separately labeled overlay remains the collider visualization. Three furnished hero videos, synchronized RGB/depth/semantic/instance products, robustness reruns, release diagnostics, and same-trace rerender proof do not exist because no primary free-object manipulation passed. The compact object trace also does not satisfy the requested full joint-state truth schema, another reason Gate E is NOT RUN rather than passed.
