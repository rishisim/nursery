# Spec-driven MIMo–MolmoSpaces prototype-kernel qualification

**Terminal decision: `GO_SPEC_DRIVEN_PROTOTYPE_KERNEL_READY`.**

This is a narrow engineering qualification of the bounded
`object_centered_reach_manipulate` prototype. It is not a ChildLens
distribution-generator result, a learner result, a multimodal cue-lift result,
an infant age-match claim, or validation of German language quality.

The manually authored calibrated `EpisodeIntent` compiles to a deterministic,
provenance-rich `ResolvedEpisodeSpec`. The shared-clock physics trace drives
appearance rendering, local German waveform timing, side streams, QA, and the
bundle manifest. Camera and target staging values are resolved fields rather
than clip-name branches. Unsupported activity families fail explicitly.

## What passed

- Actual pinned furnished FloorPlan1 and FloorPlan201 contexts are visible and
  deterministically staged outside occupied scene cells.
- An official MPFB-generated continuous weighted hand/forearm has all five
  fingers. Across the full 901-frame replay, maximum landmark error was
  `2.96e-8 m`, below the frozen `0.012 m` maximum tolerance.
- The canonical FloorPlan201 near miss remained `0.02255 m` from every MIMo
  hand collision geom with zero contact substeps. Its inspected rendered-skin
  gap was `0.02385 m`, above the frozen `0.003 m` visual margin.
- Touch/push is contact driven. Grasp activation follows measured contact,
  causes no pose jump, lifts `0.10194 m`, transports `0.11924 m`, releases, and
  settles with maximum speed `0.000199 m/s`.
- Target area stays above `1.5%`: the lower of the two nominal scene minima is
  `1.5957%`; the lower frozen camera-sensitivity result is `1.5960%`.
- RGB/depth/segmentation, action, contact/touch, qpos/qvel/proprioception,
  vestibular/IMU, object/hand/camera transforms, phase, and speech use one
  validated 901-sample episode clock.
- A second run reproduced every trace stream exactly: maximum numeric error
  `0`, with identical trace SHA-256.
- Six focused cells were produced. Five qualify. FloorPlan1's grasp cell is
  deliberately retained as a diagnostic failure because transport was
  `0.0731 m`, below the frozen `0.10 m` criterion; the gate was not weakened.
- No raw or restricted ChildLens content is in the evidence. ChildLens remains
  the sole empirical ancestry, through already-frozen privacy-safe calibration
  records only.

## Evidence locations

Generated media and traces remain in the ignored external evidence root:
`/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/media/spec_kernel`.
The compact qualification manifest records the representative relative paths
and hashes.

## Next scientific step

Freeze this qualified kernel as the simulator interface, then build a small
ChildLens-aggregate-conditioned sampling layer over resolved specs for this
single activity family. Only after distributional QA should the same
naturalistic vision-language learner be run across preregistered training arms.
