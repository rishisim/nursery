# Bimanual Child-Head Episode and Appearance Gate

Decision: **NO-GO at Stage C and at the integrated truth/visual/appearance gate (2026-08-03).**

Stage B provides bounded free-object physical-kernel evidence. It does not
establish an anatomical or visibly registered bimanual episode. Stage C is a
NO-GO/incomplete because the camera contract and visible-to-physical
registration gates were not met. The 56 s output is therefore only a partial
physical/render demonstration, not a Stage-D or engine/render-kernel PASS.
No implementation rescue or media rerun was performed for this correction.

## Preserved evidence and lineage

- Started clean and detached at verified `origin/embodied-simulation`
  commit `7fc335fcf936e3b61b2c20b5cefd1b1eca2c703c`.
- The Unity visual-audition PASS, Unity–MuJoCo NO-GO, imported-rig NO-GO,
  anatomical Stage-A PASS, and corrected anatomical Stage-B NO-GO remain
  distinct Git evidence. No earlier failure was reinterpreted as naturalistic
  output.
- Only the permitted CC0 avatar
  `b766981d9d3504cea220c0d72ad8aa56cbd80453e910fc76dc8c8814fbd980de`
  and preserved public furniture were copied into the ignored project.
- No restricted ChildLens media or restricted drive was accessed.

## Ordered A→F result

| Stage | Result | Key evidence |
|---|---|---|
| A — seven-DOF coordinate truth | NO-GO, automatic fallback | Set/Get round-trip 0; body/index audit PASS; angular max 0.158°/0.618%; translational max 27.469°/28.486% after the one row-index convention repair |
| B — hybrid hard cell | BOUNDED PHYSICAL-KERNEL PASS | 10.033 s; lift 0.1013 m; turn 25.60°; penetration 1.118 mm; speed 0.1796 m/s; free release; no assistance/object writes/joints/forces. Right contact includes thumb and multiple non-thumb digits. Left contact is only the little digit for 125 physics steps (249 geometric rows, about 0.52 s), so “bimanual support” is qualified and narrow. |
| C — weighted bimanual baseline | NO-GO / INCOMPLETE REGISTRATION | Fixed approximately 50° downward optical pitch violates the near-neutral body-derived mount contract. Head pose is synthesized from time during render replay and camera extrinsics are absent from the physical trace. Palm rotations and per-digit pose/velocity are absent. Contact visibility is only an object-center viewport test; arm registration is only bone-name presence. |
| D — continuous curated baseline | PARTIAL PHYSICAL/RENDER DEMONSTRATION; NOT PASS | 56.033 s and 1,681 frames exist at 240/30 Hz, but the videos are 960×540 rather than 1920×1080. Review shows a sparse primitive tabletop, unclothed skin, a mostly static target/table-facing view, furniture fragments at the top edge, and no meaningful final look toward the window. It is not a furnished naturalistic playroom and is visually worse than the preserved Unity visual audition. |
| E — robustness/truth QA | 2/3 PHYSICAL-CELL ROBUSTNESS; TRUTH INCOMPLETE | nominal PASS; +5 mm lateral FAIL; mass 0.045 kg/friction 0.75 PASS. This does not promote C or D. Required camera/head, per-digit, palm-rotation, object-angular-velocity, contact-registration, depth, semantic, instance, and IMU evidence is absent. |
| F — audio/appearance | AUDIO ARTIFACT PRESENT / APPEARANCE NO-GO | Six synthetic utterances, 48 kHz, zero clipping, and an exact 1,681-frame mux exist. No enhanced appearance video exists. |

## Authority, semantics, and registration limits

The physical target and blue cup are free, non-kinematic PhysX rigidbodies.
Kinematic palms and contact-aware digit colliders are driven at 240 Hz. The
assistance ledger is empty: there are no attachments, joints, object pose
writes after initialization, external forces, hidden supports, or post-step
repairs. Kinematic authority is not biomechanics. Contact impulse is
unavailable and omitted; geometric contact is measured with Unity collision
geometry.

The physical trace records palm positions and aggregate closures, but not palm
rotations or per-digit pose/velocity. It also omits object angular velocity and
authoritative per-frame root/torso/neck/head/camera extrinsics. The rendering
replay synthesizes head motion from `row.time_s` and replays palm position plus
aggregate closure. Consequently it cannot register the physical palm turn or
digit motion to the weighted skin.

`contact_event_frames_visible` is not a contact-registration measurement: it
projects the object center while a contact row exists. It does not project the
measured contact point onto the correct visible digit/object surfaces or test
visible-versus-physical touch timing. Likewise, `both_arms_registered` checks
bone-name presence only. The implementation's `report.passed` predicates are
existence/count receipts, not the frozen visual/truth gates, and cannot confer
a scientific Stage-C or Stage-D PASS.

## Robustness

| Frozen cell | Result | Lift | Turn | Max penetration | Notes |
|---|---:|---:|---:|---:|---|
| nominal | PASS | 0.1013 m | 25.60° | 1.118 mm | Physical-cell result; left contact only the little digit for 125 steps (~0.52 s) |
| target x +0.005 m | FAIL | 0.0183 m | 0° | capture and left assistance fail |
| mass 0.045 kg, friction 0.75 | PASS | 0.1013 m | 25.55° | 1.073 mm | no retuning |

## Immutable ignored-run receipts

Root: `runs/embodied_simulation/bimanual_childlens_gate/` (ignored).

- Full authority trace SHA-256 `7fe3777fe03806984378e9695371e885db31672d88d3df01e6fe4f99b0bc30e7`.
- Full contact trace SHA-256 `5820f755ce45e25a4e191e46729495ee563a5d44363ee8de1a9983de09e4986d`.
- Baseline head video SHA-256 `c133699038a10c12e0ca1ed07b4ef41e1765f8cf731302fdb320370226a21ae6`.
- Baseline external video SHA-256 `8bf8fb480e6d7a2044717cc7d3c26e95360a212e5172e39dd4d8e6d1eea23859`.
- Dense timeline SHA-256 `c584e009b29262ed6648004f115835d7cc45379776ca210476b56047a9a781a2`.
- Audio mux SHA-256 `7e3932fa6e0d280ff09b40b5075fc7697ea57febdaaf20812e818c622c20d0fe`.

Host: Apple M5, 32 GB unified memory, 10-core GPU, Metal 4; Unity
6000.0.80f1 ARM64. Same-machine numerical tolerance only; no universal
bitwise-determinism claim.

## Absent products

There is no accepted enhanced video; baseline/enhanced side-by-side; clean
external plus separately labeled collider/contact overlay; metric depth;
semantic or persistent-instance video; authoritative per-frame camera/head
provenance; palm rotations; per-digit kinematic pose/velocity; object angular
velocity; measured-contact-to-visible-skin projection/timing validation;
derived head IMU; or complete cross-modal drift receipt. These absences, the
Stage-C NO-GO, and the visual/camera failures preclude both
`BASELINE-KERNEL PASS / APPEARANCE NO-GO` and `INTEGRATED PASS`.

This is public/synthetic engineering evidence only. It is not infant-trained,
age-matched, ChildLens-calibrated, human-validated, or biologically
torque-valid.
