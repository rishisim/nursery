# Reference-hand microcell decision

Decision: **NO-GO for the adopted Ultraleap Hard Contact source architecture**.

This is a bounded result for the exact `com.ultraleap.tracking` 7.3.0 source
and the two permitted physical executions. It does not reopen or weaken the
preserved old compliant-finger/controller NO-GO at commit `3e9d2d9`, and it is
not a claim against Unity PhysX or the wider Embodied Simulation program.

## Question and contract

The test asked whether a maintained anatomical Unity hand could perform one
unassisted tabletop grasp with: contact-free approach; visible thumb/index/
middle contact; simultaneous opposing nonzero `Physics.ContactEvent`
point-impulse support for at least 0.30 s; unsupported lift of at least 0.10 m;
commanded opening; and free release/settling.

## Foundation and preflight

- Ultraleap UnityPlugin `com.ultraleap.tracking` 7.3.0 was resolved from the
  exact public commit `833d82e7333a5f37ebc0844d02431acf74f35d24`, with the
  pinned source, package, license, archive, native library, and GenericHand
  hashes recorded in
  `configs/embodied_simulation_reference_hand_microcell.json`.
- Unity `6000.0.80f1` ARM64/Metal compiled the package and built a standalone
  macOS player. The runtime used an explicitly assigned deterministic
  `SyntheticLeapProvider`; no Leap service, hardware, or service clock was
  used. This is a source-backed engineering inference because the manager
  accepts a `LeapProvider`, not an official device-free mode.
- The manager was authored with `HardContactParent`. The stock prefab was not
  instantiated. Runtime receipts report zero `GrabHelper` components and zero
  `GrabHelperObject` instances. Ultraleap grab flags were recorded only as
  diagnostics, never as qualification evidence.
- The physics clock was one explicit `FixedUpdate` command/provider/
  `Physics.Simulate`/record loop at 240 Hz, with exactly 8 steps per 30 Hz
  render marker. The target was a free, non-kinematic, gravity-enabled
  `Rigidbody`; the authority receipt records zero post-initialization target
  pose, velocity, force, torque, kinematic, parenting, or joint writes.

## Physical results

| Execution | Change | Point-contact rows | Thumb/index/middle support | Lift | Result |
|---|---|---:|---:|---:|---|
| Attempt 1 | frozen initial goal | 0 | 0.00 s | 0.000 m | NO-GO |
| Attempt 2 | one permitted wrist/hand vertical alignment repair | 0 | 0.00 s | 0.000 m | NO-GO |

Both traces observed commanded opening and free settling, but neither
observed any measured thumb, index, or middle point impulse. The target never
left its support. The complete raw receipts remain in the ignored run root
`runs/embodied_simulation/reference_hand_microcell/`; their hashes and paths
are frozen in the canonical config.

The failure is therefore earlier than the grasp qualification contract. No third physical attempt was run, and no object assistance, reset, teleport,
ghost recovery, force injection, attachment, or hidden support was added.

## Recorder truth repair

The canonical `PhysicsTruthRecorder` was repaired so a `ConfigurableJoint` now
reports the engine-observed `rotationDriveMode` and reads the active `slerpDrive`
when Slerp is configured, rather than sampling inactive `angularXDrive` fields.
Legacy field names remain for schema compatibility and are explicitly sourced
from the active drive. A focused regression test covers this mapping, and the
sealed source hash in the procedural gate config was updated; the historical
NO-GO receipts remain immutable evidence of their original run.

## Ordered stop and absent artifacts

Because the adopted hand did not produce measured contact in either permitted
execution, the cell is not eligible for PASS or
PROMISING-BUT-ONE-BOUNDED-REPAIR. The following were not generated after the
hard stop: object-free per-DOF qualification/active-drive plots, a frozen
ContactPose-derived grasp goal and optimizer receipt, nominal 1080p Unity
video, collider/contact overlay and dense ffmpeg frames, grasp/lift/release
plots, and the two additional frozen-shape runs. No video is presented as
success evidence; the available receipts are failure traces and diagnostics.

No restricted ChildLens media, external drive, child-trained prior, age-matched
asset, or licensed MANO bundle was accessed.
