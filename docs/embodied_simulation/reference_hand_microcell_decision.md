# Reference-hand microcell continuation decision

Decision: **HAND-QUALIFICATION NO-GO; eligible physical attempts consumed: 0.**

This corrects the earlier commit `02af1d9`. The two old `player_run` receipts
are preserved, but are reclassified as non-eligible prequalification harness /
alignment diagnostics. They used a hand-authored `MakeFinger`/`PoseCommand`
pose before object-free hand qualification and therefore cannot falsify the
Ultraleap donor architecture or consume either physical execution attempt.

## Canonical source and reproducibility

The canonical source and runner are tracked under
`babyworld_lite/childlens_engine_bakeoff/reference_hand_microcell/`. The runner
stages those files into the ignored Unity project and uses the exact pinned
Ultraleap artifact recorded in
`configs/embodied_simulation_reference_hand_microcell.json`. A fresh checkout
can run `stage`, `build`, and `run` without relying on a pre-existing ignored
project source file.

The package was compiled with Unity `6000.0.80f1` on Apple M5 ARM64/Metal.
The manager received an explicitly assigned deterministic
`SyntheticLeapProvider`; no service, hardware, or service clock was used. The
stock manager prefab was not instantiated, and runtime preflight found zero
`GrabHelper` or `GrabHelperObject` instances. Ultraleap’s documented Physical
Hands foundation is [here](https://docs.ultraleap.com/xr-and-tabletop/xr/unity/plugin/features/physical-hands.html).

## Corrected interpretation of earlier receipts

The old receipts at `receipts/player_run` and `receipts/player_run3` answer only
that the bespoke synthetic pose was not a plausible grasp goal. Their target
support predicate was also wrong: the support top was `y=0.74 m`, the target
half-height was `0.05 m`, and its settled center was approximately `y=0.79 m`.
The old `center_y < 0.78 m` test therefore mislabeled support as absent. Phase
presence and a package grab flag likewise cannot prove free release. The old
JSON files and hashes remain unchanged historical diagnostics; their metrics
are not used for this decision.

The corrected definitions are frozen in the canonical config: support requires
measured support contact plus center height at or below the support-top-plus-
half-height tolerance; unsupported lift requires support-contact loss and the
height threshold; free release requires a previously qualified support window,
opening after that window, measured contact loss, and subsequent gravity
settling.

## Object-free qualification result

The tracked harness uses the package’s neutral `TestHandFactory` topology only
as an object-free input fixture. It contains no object, grasp geometry,
ContactPose pose, or target-specific closure. The package-created physical hand
was audited at runtime:

- 16 physical ArticulationBody links and 31 total articulation reduced
  coordinates were recorded, including 25 finger coordinates.
- 20 unlocked finger DOFs were individually swept; all 20 swept ranges passed.
- Every telemetry row records the body, `dofStartIndex`, joint position,
  velocity, force, link transform, and x/y/z `ArticulationDrive` target,
  stiffness, damping, limits, and force limit.
- Physics-derived thumb/index/middle output and GenericHand/contact-site
  landmarks were recorded at 240 Hz with 30 Hz diagnostic frames.
- The visible landmark registration error was zero for the explicit
  physics-derived adapter; this is not an independent animation pass and is
  not used to override the failed fingertip tracking result.
- Active-phase reset/ghost ledger count was zero; palm command speed was zero
  and bounded.
- The frozen fingertip tracking tolerance was `0.008 m`. The maximum
  physics-derived fingertip tracking error was `0.05772646 m`, dominated by
  the thumb, so stable thumb/index/middle tracking did not qualify.

This is a bounded **hand-qualification** failure after ordinary source,
package, axis, clock, and telemetry integration repair. No target was present
and no grasp execution was started.

## Visual evidence

The qualification produced an actual Unity diagnostic video and dense PNG
frames, all ignored under the run root:

- `qualification/dof_sweep_diagnostic.mp4`: 1920x1080, 30 fps, 336 frames,
  11.2 s; verified with `ffprobe` and complete frame decoding.
- `qualification/frames/`: 336 dense Unity frames.
- `qualification/dof_manifest.json`, `trace.json`, `qualification_metrics.json`,
  and `reset_ledger.json` contain the corresponding runtime truth.

The video is failure evidence: the visible package hand/contact-site overlay
does not establish the frozen fingertip tracking tolerance, and no grasp or
target is shown.

## Ordered stop

Because object-free hand qualification failed, the ordered gate stops here:

- no ContactPose seed was consumed;
- no DexGraspNet-style analytic SDF objective or candidate ranking was run;
- no frozen static grasp goal was rendered or executed;
- no eligible physical attempt was consumed;
- no target shapes, lift, release, or downstream child/POV integration were run.

The next authorized repair must address the verified thumb/index/middle
physics-output registration and coordinate convention, then rerun this same
object-free qualification harness. It must not proceed to a grasp until the
frozen tolerances and visual/contact-site registration pass.

No restricted ChildLens media, external drive, child-trained prior, age-matched
asset, human validation, or licensed MANO bundle was accessed.
