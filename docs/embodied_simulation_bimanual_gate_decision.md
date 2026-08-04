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

## Bounded Stage-C reconstruction follow-up (2026-08-03)

Decision: **PASS for the separate eight-second registration question.** This
does not revise the 56-second gate's NO-GO above. It repairs and tests only the
specific claim that the retained free-object kernel can be replayed through the
weighted skin, a correct child-facing workspace, a neutral body-derived camera,
and an inspectable physics overlay.

The reconstruction records palm rotations, independent digit closures, every
physical digit-segment pose and velocity, object angular velocity, measured
contact point/normal/separation, and root/torso/neck/head state at the shared
240 Hz to 30 Hz clock. The approximately 50-degree target-facing camera and the
backwards `-Z` workspace were removed. The replacement optical origin is 45.32
mm outside the head-weighted skin, has zero-degree neutral mount error, uses a
68-degree vertical FOV, and inherits at most 1.61 degrees of roll. The avatar
faces `+Z`; the target-front dot product is 1.0.

The manipulation controller was also corrected after dense video/contact
review. Rotating two wrists in place made the cube look unsupported and dropped
thumb opposition. The accepted primitive instead orbits both hand frames about
the lifted interaction center while the object remains a free dynamic PhysX
body. Across the accepted run:

- lift is 89.65 mm and rotation is 24.99 degrees;
- thumb-plus-nonthumb opposition is qualified in all 91 required render
  intervals, with 812 right-thumb, 1,101 right-nonthumb, and 275 left-support
  contact physics steps;
- maximum finger penetration is 2.697 mm, maximum palm speed is 0.174 m/s,
  maximum commanded palm angular speed is 37.50 degrees/s, and no hand contact
  remains in the final quarter-second;
- the assistance ledger is empty: zero object pose writes after initialization,
  external forces, attachments, or joints;
- a fresh-process same-machine rerun is byte-identical for the authority trace,
  contact trace, and report.

The 241-frame 1920x1080 replay passes its frozen registration checks: palm
maximum error 9.92 mm, contact-to-correct-digit p95/max 7.97/8.02 mm, and first
visible/physical touch difference zero frames. Clean head and external renders
contain one weighted child and zero collider pixels. A third separately
labeled QA video identifies green right-hand colliders, cyan left-hand
colliders, and magenta physical contact points so that these overlays cannot be
misread as a second body.

Full-video review used 4 Hz dense timelines (96 inspected samples across the
three videos), in addition to event frames and `ffprobe`. The child is no
longer backwards and the target is visible throughout touch, lift, turn,
release, and withdrawal. The result is nevertheless a deliberately simple
simulator diagnostic. The unclothed generic skin, rigid cube, sparse stylized
materials, limited camera motion, and only partly readable clean-view finger
envelopment are not ChildLens-like appearance evidence.

Ignored root: `runs/embodied_simulation/stage_c_reconstruction/`.

- Authority trace SHA-256 `73c55f713d7ed022cff36f88c7da36e37db0af5a7826f1ca948dccf76ba04232`.
- Contact trace SHA-256 `e856e892ad7bdebf2432315e4283153794c3a3d30d897e1a343993cb6a07fd83`.
- Registered trace SHA-256 `c634f01edafb8564776adec6b6652988aa49f2f54577f0cb8e0f8be4566d160d`.
- Head video SHA-256 `616b1fd810daeec0ed5dbfb7ace18feb20ad677e838f522293485295ca936c89`.
- Clean external video SHA-256 `89f5555f129030d2ed11217a4352440d97148a2ff1015b19dbaf566f14b0a3ab`.
- Labeled physics-overlay video SHA-256 `452098e146bb77c61cad07f41b163eada35b36a3e2252823040c5759356c587b`.

The smallest next scientific gate is not another longer baseline episode. It
is to add same-state metric depth and persistent identity/contact masks to this
exact accepted trace, then audition one tightly conditioned appearance pass
that must preserve hand/object silhouettes and registered contact timing. No
appearance run was performed here, and no restricted child media was accessed.
