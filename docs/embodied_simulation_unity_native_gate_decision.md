# Anatomical Physics Rig and Dexterous Grasp Qualification Gate

Decision: **NO-GO at ordered Stage B (2026-08-03).** The deliberately authored
reduced Unity articulation passes static registration and individual-DOF sweep,
but its engine Jacobian does not agree with fresh-state central differences and
it cannot satisfy the five-waypoint palm gate. Per the preregistered hard stop,
no target-contact, grasp, lift, release, robustness, or hero capture was run.
This is a bounded rig/controller result, not a general claim about Unity
ArticulationBody, the prompt system, or the research program.

## Lineage and preserved evidence

- Started at `04ca8087e9abb6772029e1d08876904afbed29b3`, the then-current
  `origin/embodied-simulation` commit.
- The Unity visual audition PASS, Unity–MuJoCo NO-GO, and prior imported-MPFB
  Unity-native NO-GO remain distinct historical evidence in Git history.
- The prior Unity-native implementation is retained only as historical
  reproduction code. The active rig is
  `AnatomicalPhysicsRigBuilder.cs`, driven by the single canonical
  `embodied_simulation_anatomical_rig.json` manifest.
- The exact prior CC0 weighted avatar was reused from the preserved ignored run
  root. Its SHA-256 is
  `b766981d9d3504cea220c0d72ad8aa56cbd80453e910fc76dc8c8814fbd980de`.
- Unity was the pinned public local editor at `6000.0.80f1` ARM64. No restricted
  ChildLens media or external drive was accessed.

## Ordered gate result

| Gate | Result | Evidence |
|---|---|---|
| A — anatomical rig/static registration | PASS | 22 controlled DOFs; coordinate round-trip 0.000123 mm; collider-to-skin median/p95/max 0.646/2.559/6.694 mm; one weighted skin; no deformation-bone `ArticulationBody`; no independent animation |
| B — FK/Jacobian/five waypoints | NO-GO | settled zero-offset error 0.0016 mm / 0.247 degrees, but four nontrivial waypoint errors 28.5–122.4 mm and 13.7–75.7 degrees; engine-vs-central-difference maximum direction/magnitude errors 167.45 degrees / 83.06% |
| C — contact-free reach/preshape | NOT RUN | prohibited after Stage-B failure |
| D — grasp/lift/rotate/place/release | NOT RUN | prohibited after Stage-B failure |
| E — robustness/replay/synchronized capture | NOT RUN | prohibited after Stage-B failure |

The second Stage-B run repaired two invalid qualification mechanics from the
first attempt: every finite-difference cell used a fresh build with a 1-degree
central perturbation and 1,200 manual 240 Hz convergence steps, and every
waypoint was defined from a separately settled 720-step fresh state. The
zero-offset cell then passed tightly, but the nontrivial cells and Jacobian
columns still failed by large margins. This repeated result is the hard-stop
condition, not a scene-layout reinterpretation.

## Immutable ignored-run receipts

Root: `runs/embodied_simulation/anatomical_physics_gate/`

- `stage_a/stage_a_report.json` — SHA-256
  `5dfcda2489ab3a5396c4913f9b860c0a2568d87ccc5ba3946f452f87ea86730d`
- `stage_a/stage_a_sweep.json` — SHA-256
  `43ef607432b3e577249263187951b80bae370949b36484268bc23bfeb632aa12`
- `stage_b/stage_b_report.json` — SHA-256
  `c05ca65fba37eddde41ca83c631492c7fdb1039e0beafdc568bf13fd9f585b93`
- `stage_b/stage_b_jacobian.json` — SHA-256
  `de4ccad0002bf67c9a51b57f25ef40262b27970e3e651d3db4d1fb1f4e97a005`
- `stage_b/stage_b_waypoints.json` — SHA-256
  `756d984ea56fb5630519036ef6ebbea4e146bbc4749847270a357000312be1d4`
- `stage_b/evidence/capture_report.json` — SHA-256
  `7c4daabf493f2ea405b2b070944cedb6eb30513318507069d98c2b5bc342cda4`
- `stage_b/evidence/frame_ledger.json` — SHA-256
  `4816e9fdafc659d07bf1e5c82658d5366b58781f520319440db61a27895a775c`
- `failure_media/stage_a_dof_sweeps.mp4` — 8.8 s, 1920x1080,
  30 fps, 264 frames; SHA-256
  `5115d0d044669ff0a2ed8c463aaecb1c1f01e22d9bd7311ad9d67d76118a415a`.
- `failure_media/stage_b_waypoint_failures.mp4` — 20.0 s, 1920x1080,
  30 fps, 600 frames; SHA-256
  `b6d2e464ca0823bdb2c4d222cfcee3d98e060a44a2cf2230936a8aefcb6ee5aa`.
- `failure_media/anatomical_rig_stage_b_no_go_diagnostic.mp4` — 28.8 s,
  1920x1080, 30 fps, 864 frames; SHA-256
  `ab7dabac71e5f63ab850c1cf8912e552b684fbaefd571e5ba2212f34b343d43a`.
- `failure_media/combined_dense_timeline.png` — one-second dense timeline;
  SHA-256
  `20471d86f020b0ae1fe22eb5f143d7e9aba0e524f3b1e950aaf8be0cd65334c4`.

These are explicitly labeled failure diagnostics, not manipulation footage.
Stage A shows all 22 controlled DOFs with joint/axis/limit/target labels; its
clean and collider-overlay halves are rendered from each identical frozen
engine state without stepping between passes. Stage B shows five separately
rebuilt and 720-step-settled waypoint trials, each for 120 frames with the red
target, blue observed palm/trail, and live position/orientation error. The
capture replay's final trial errors were 0.043, 31.817, 31.159, 54.267, and
26.085 mm, with 0.575, 25.642, 35.063, 78.807, and 14.398 degrees orientation
error. This independently visible replay also fails; the ordered gate table
continues to report the authoritative qualification receipt above.

The ignored run root is consolidated to `stage_a/`, `stage_b/`,
`failure_media/`, and the runnable `project/`. Task-created superseded attempt
directories and the earlier static-state MP4 were removed from the run root.

## Absent downstream deliverables

There is no first-touch/contact-aligned RGB/depth/identity capture, qualified
opposition graph, grasp/lift/rotation/place/release trace, head-view hero,
clean manipulation external view, release plot, assistance ledger for an
episode, robustness table, replay receipt, or same-trace rerender receipt.
Creating any of them after this Stage-B failure would violate the ordered gate.

## Smallest scientifically credible next step

Do not tune a grasp controller. First isolate the Unity dense-Jacobian row and
anchor-basis convention on the seven-DOF arm alone (no skin, fingers, target, or
gravity), set reduced coordinates directly for symmetric perturbations, and
require every palm translational/angular column to match central differences.
Only after that validator passes should the same verified bases be restored to
the registered visual-follower rig and the five fresh-state waypoint gate rerun.
