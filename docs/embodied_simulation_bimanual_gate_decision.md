# Bimanual Child-Head Episode and Appearance Gate

Decision: **NO-GO at the integrated truth/visual/appearance gate (2026-08-03).**

The hybrid Unity/PhysX manipulation kernel passes, a 56 s weighted-child
head-view baseline and synchronized event audio were produced, and frozen
robustness is 2/3. The integrated result is nevertheless a NO-GO: the visual
baseline remains tabletop-dominant and visibly synthetic, the final run does
not contain metric depth, semantic/instance renders, or derived head IMU, and
no tightly conditioned temporal appearance method available on this Apple
Silicon host could be run without weakening the identity/contact/temporal
gates. It is not an integrated ChildLens-shaped artifact.

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
| B — hybrid hard cell | PASS | 10.033 s; right/left measured contact; lift 0.1013 m; turn 25.60°; penetration 1.118 mm; speed 0.1796 m/s; free release; no assistance/object writes/joints/forces |
| C — weighted bimanual baseline | PASS | 10.033 s; one weighted child; both `.R`/`.L` arms; fixed head-derived 68° camera; zero proxy renderers; 181 visible contact-event frames |
| D — continuous curated baseline | ENGINE/RENDER KERNEL PASS, visual weakness | 56.033 s, 1,681 frames, 240/30 Hz exact mapping, free red and blue objects, one trace, 11 context objects; full dense review remains tabletop-dominant |
| E — robustness/truth QA | 2/3 manipulation robustness; TRUTH INCOMPLETE | nominal PASS; +5 mm lateral FAIL; mass 0.045 kg/friction 0.75 PASS; depth/semantic/instance/head-IMU unavailable |
| F — audio/appearance | AUDIO PASS / APPEARANCE NO-GO | six synthetic utterances, 48 kHz, zero clipping, exact 1,681-frame mux; no acceptable temporal multi-control appearance route installed |

## Authority and semantics

The physical target and blue cup are free, non-kinematic PhysX rigidbodies.
Kinematic palms and contact-aware digit colliders are driven at 240 Hz. The
assistance ledger is empty: there are no attachments, joints, object pose
writes after initialization, external forces, hidden supports, or post-step
repairs. Kinematic authority is not biomechanics. Contact impulse is
unavailable and omitted; geometric contact is measured with Unity collision
geometry. The weighted-skin videos are explicit same-trace rendering replays,
not a second physical authority.

## Robustness

| Frozen cell | Result | Lift | Turn | Max penetration | Notes |
|---|---:|---:|---:|---:|---|
| nominal | PASS | 0.1013 m | 25.60° | 1.118 mm | right and left contact |
| target x +0.005 m | FAIL | 0.0183 m | 0° | 0.346 mm | capture and left assistance fail |
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

There is no accepted enhanced video, baseline/enhanced comparison, metric
depth video, semantic or persistent-instance video, derived head IMU plot, or
complete cross-modal drift receipt. Their absence is why this cannot receive
`BASELINE-KERNEL PASS / APPEARANCE NO-GO`, which requires full baseline truth,
and why the integrated decision is NO-GO.

This is public/synthetic engineering evidence only. It is not infant-trained,
age-matched, ChildLens-calibrated, human-validated, or biologically
torque-valid.
