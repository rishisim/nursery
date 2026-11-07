# Step 2: preserve richer bodily information

Design reference in [PLAN.md](../PLAN.md). This step defines the sensory change
before the [controlled learning comparison](body_representation_comparison.md).
No richer representation or new sensor has been implemented by this document.

## The idea

Give the learner a more faithful description of bodily experience using the
information we already have. The current representation discards spatial and
temporal detail. That justifies improving it; the resulting language benefit
remains a question for the next steps.

Two different arrangements of touching vertices can both become "50% contact"
after averaging a finger patch. Preserve that distinction without supplying
object identity as an input.

## What the learner currently receives

The [body feature code](../nursery/learner/body.py) converts 1,556 binary hand
contact indicators into 22 anatomical averages, keeps 90 finger rotations,
and supplies 27 arm-position/movement/gravity values. Optional physical groups
add ten load values and four skin-temperature/contact-area values. The body
window is clip-bounded, approximately five seconds before and after the
utterance, with one exported sample per second.

The [physical model](../nursery/simulator/physical.py) estimates load from mass
and acceleration; temperature and contact area are per-hand proxies. They are
not pressure maps, friction measurements or deforming contact simulation.
Recorded motion does not react to assigned mass or material.

## What can be recovered, and what remains missing

| Missing distinction | Existing information we can use | Information requiring additional work |
|---|---|---|
| Different contact arrangements inside one patch | Saved 1,556 contact bits and their hand-vertex mapping can preserve finer contact locations. | Pressure at those locations is not encoded in binary contact. |
| Hand shape versus pressure or surface properties | Recorded finger pose, joints and hand geometry can describe configuration and geometry around contact. | Pose alone does not supply pressure, friction, stiffness or material identity. |
| Brief contact changes and rapid adjustments | Original higher-rate GRAB motion/contact can provide denser histories where source frames and assets remain available. | Interpolation of one-second exports cannot recover unrecorded events. |
| Spatial pressure distribution and detailed mechanics | Preserve contact locations and motion-derived load without excessive averaging. | A contact-force model or measurements are needed for distributed pressure and detailed mechanics. |
| Texture, friction and deformation | Available surface geometry can describe coarse local shape where touched. | These sensations require suitable properties, models or measurements. |
| Spatial thermal detail | Retain existing contact geometry and per-hand temperature dynamics. | A spatial thermal field requires a new model/representation; it is not in the current scalar proxy. |

The first candidate is finer contact structure, recovered temporal detail and
a useful description of recorded hand geometry. Do not expand every channel
merely to increase dimensionality. Finger pose is already comparatively rich;
make its available geometry useful without claiming new sensations.

## How we define the candidate

1. Trace each channel from recordings through export, normalization and encoder
   input. Record what exists, what is discarded and what is modeled.
2. Choose a contact encoding that retains finer spatial arrangements. Specify
   the anatomical/vertex mapping, dimensions and missing-data treatment.
3. Verify higher-rate sources and source-frame correspondence before choosing
   a sample rate. For substituted objects, recompute contact against the
   substituted mesh; original-object contact labels do not automatically apply.
4. Keep coordinates and units explicit. Geometry features must describe bodily
   configuration and accessible contact; hidden labels, whole-object identity
   and untouched surfaces are not ordinary sensory inputs.
5. Preserve temporal span, clip boundaries and source partitions. Fit
   normalization on selected training clips only. Record required changes in
   encoder capacity, sensory exposure and compute.

Freeze one candidate before language comparison. Testing the full representation
change measures that package; it does not identify which channel caused a gain.
A channel ablation is a later targeted question if needed, not an automatic sweep.

## The output we want

A compact channel specification: source, mapping, units, sample rate, window,
dimensions, normalization and missing-data rules. Include examples of contact
or motion patterns that were collapsed before and are now distinguishable.
List unavailable sources and sensations honestly.

Maintain a separate deferred-information list for pressure, friction, texture,
deformation and spatial heat transfer. Active exploration and substantial new
physical simulation remain outside this first change. Preserve recordings,
licensed assets, caches and old runs.

## Before the comparison

Check alignment, clip isolation, finite values, source coverage and whether
information survives the encoding. Labels can score diagnostics but cannot
be ordinary learner inputs or manufacture distinctive sensory signatures.

Exact contact granularity, recovered sample rate, pose encoding and necessary
encoder-size changes remain open. Resolve them from available data and
training/development evidence before Step 3, without selecting on test scores.
