# ChildLens room–hand–camera causal bake-off

**Decision: `NO_GO_TARGET_PROMINENCE_AND_EFFECTOR_FIDELITY`**

Six deterministic 8-second, 960×540, 30-fps Eevee episodes were rendered
across a playroom, kitchen, and living room. The matrix contains three ball
instances, three cup instances, push, roll, touch, lift-and-place, and a
near-miss. The target has no location keyframes in scored code. Push and roll
use rigid-body collision; lift-and-place uses a logged child-of constraint
engaged only after contact. The four positive manipulations displace targets
0.54–0.72 m. The near-miss has zero contact and zero target displacement.
Every episode has 240 shared-clock telemetry rows with hand and target pose,
velocity, action, contact, grasp state, joint, proprioceptive, IMU-like, and
camera signals.

The result nevertheless fails the frozen appearance gate. Conservative
geometry-based projected target area is 0.19–0.34% of the image against the
predeclared 1.5% minimum; 0/6 episodes pass. Direct sheet inspection agrees:
targets remain recognizable but too small for a credible lexical grounding
preflight. The self-authored segmented hand is visible and is materially better
than the prior capsule, but remains stylized rather than a production-quality
anatomical rig. The procedural rooms are furnished and room-distinct, but do
not reach ReplicaCAD/HSSD/3D-FRONT asset fidelity. Inflating objects until they
pass would break plausible scale, so that repair was rejected.

## Camera disposition

The source metadata establishes 1920×1080 at 30 fps, chest/vest mounting, and
an advertised 140-degree wide angle, but does not establish the FOV axis,
projection, intrinsics, or manufacturer coefficients. Exact physical
calibration is therefore not claimed. Before rendering, the protocol froze an
uncertainty set: mild Brown–Conrady rectilinear, 140-degree diagonal equisolid,
and mild polynomial fisheye. The canonical Eevee audition uses a fixed
13.5-mm perspective render plus a fixed -0.18 radial compositor warp; the
other two models remain preregistered sensitivity conditions. This set must
not be tuned against learner results.

## Comparison with the prior pilots

The new audition improves the prior sparse box stages with three identifiable
furnished layouts, more distractors, three instances per category, a visible
segmented effector, actual causal manipulation, and a valid near-miss. Image
summaries and the side-by-side sheets are measurement aids, not ground truth.
The same inspection still rejects target prominence and production effector
quality. Because the upstream appearance gate failed, depth/instance-mask
keyframes and Cycles reference stills were not generated, and no distribution
generator, acquisition learner, cue-lift arm, or final evaluation was run.

## Assets, privacy, throughput, and next step

All canonical geometry is self-authored in the renderer, contains no logo or
readable product text, and is listed in `asset_license_ledger.json`. ReplicaCAD
was not locally cached and no direct official ungated scene package was
identified in the bounded audit. The official CC0 Blender Human Base Mesh
bundle is a full-body unrigged base rather than a ready IK hand. Poly Haven's
CC0/API policy was verified, but its bounded object subset did not solve the
room or hand gate. Median parallel-run wall time recorded per Blender process
was 309.7 seconds (parallel resource contention included); the single-process
reference was about 120 seconds for 240 frames.

No ChildLens frame, audio, transcript, identifying path, or raw measurement is
in these records. No AEA or BabyView material was accessed.

The exact next step is a new bounded asset integration—not learner work:
obtain a license-audited furnished room package and a rigged anatomical
hand/forearm, then stage physically plausible target distances that meet the
already frozen low/central/high prominence regimes under every camera model in
the uncertainty set. Rerun only this six-cell appearance/causal audition.

This follows Michael Frank's advice by refusing to generate a large simulated
distribution until the simulator can reproduce a measured, developmentally
relevant visual property while retaining trustworthy embodied side streams.
The mechanics are now promising; the visual bridge is not yet qualified.
