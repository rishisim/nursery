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

The separate historical ConfigurableJoint recorder correction reports the
active `slerpDrive` when that discarded controller uses Slerp mode. It is not
used as evidence here: this donor is an ArticulationBody hand, whose runtime
evidence is the recorded x/y/z `ArticulationDrive` state above. ArticulationBody has no `slerpDrive` field in this harness.

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
- 25 finger coordinates were inventoried; 20 unlocked coordinates were
  individually swept and all 20 passed. The five spherical-joint Z coordinates
  (indices 2, 7, 12, 17, and 22) are explicitly recorded as
  `locked_ineligible`; they are not included in the controllable pass count.
- Each sweep uses the declared Articulation axis: x/flexion uses the local
  physical x axis, y/abduction uses local y, and locked z sends no perturbation.
  Active drive target ranges were approximately 22.918 degrees with off-axis
  target deltas below 0.001 degrees. The profile is a frozen ramp/hold/cross
  sequence whose peak command speed is below the package's 1.75 rad/s bound.
- Every telemetry row records the body, `dofStartIndex` using the invariant
  `dof_start_indices[ArticulationBody.index]`, joint position,
  velocity, force, link transform, and x/y/z `ArticulationDrive` target,
  stiffness, damping, limits, and force limit.
- The post-step output is independently reconstructed from the physical
  capsule endpoints using the package's `ToWorldSpaceCapsule` convention; the
  pre-sim `CurrentFixedFrame` is not treated as post-step truth. Capsule/FK
  agreement was 0.0 m. In steady-state rows, thumb/index/middle errors reached
  47.567/15.172/14.671 mm respectively against the frozen 8 mm tolerance.
- The complete package visual route was exercised through the right
  `HandBinder`, with the left root disabled. Preflight found one right binder
but zero active right-hand renderers. Visual registration was therefore not visually demonstrated. The inactive named transforms were numerically
  compared only, with a maximum mismatch of 46.220 mm; this is failure
  evidence, not a visual pass or a copied-transform zero.
- Active-phase reset/ghost ledger count was zero; palm command speed was zero
  and bounded.
- The frozen fingertip tracking tolerance was `0.008 m`. The repaired run's
  maximum steady-state error was `0.04756707 m`, and the all-row maximum was
  `0.04873600 m`, so stable thumb/index/middle tracking did not qualify.

This is a bounded **hand-qualification** failure after ordinary source,
package, axis, clock, and telemetry integration repair. No target was present
and no grasp execution was started.

## Visual evidence

The qualification produced an actual Unity diagnostic video and dense PNG
frames, all ignored under the run root:

- `qualification/dof_sweep_diagnostic.mp4`: 1920x1080, 30 fps, 1536 frames,
  51.2 s; verified with `ffprobe` and complete frame decoding.
- `qualification/frames/`: 1536 dense Unity frames.
- `qualification/dof_manifest.json`, `trace.json`, `qualification_metrics.json`,
  and `reset_ledger.json` contain the corresponding runtime truth.

The video is failure evidence: it shows only the commanded/physical markers
and telemetry overlay on black, with no active right-hand renderers. The full
HandBinder route therefore did not produce a readable anatomical hand, and no
grasp or target is shown. The inactive-transform numeric comparison must not
be described as visual registration evidence.

## Ordered stop

Because object-free hand qualification failed, the ordered gate stops here:

- no ContactPose seed was consumed;
- no DexGraspNet-style analytic SDF objective or candidate ranking was run;
- no frozen static grasp goal was rendered or executed;
- no eligible physical attempt was consumed;
- no target shapes, lift, release, or downstream child/POV integration were run.

The bounded repair cycle ends here. A future authorized task would need to
repair the package visual activation/registration and the thumb/index/middle
post-step tracking convention, then rerun this same object-free qualification
harness. It must not proceed to a grasp until the frozen tolerances and
visual/contact-site registration pass.

No restricted ChildLens media, external drive, child-trained prior, age-matched
asset, human validation, or licensed MANO bundle was accessed.
