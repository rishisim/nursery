# Step 3: preserve richer bodily information

Design reference in [PLAN.md](../PLAN.md). This step defines the sensory change
before the [controlled learning comparison](step_4_body_representation_comparison.md).
The richer candidate is now implemented, exported and frozen; see
[Frozen candidate](#frozen-candidate--10-november-2026). No new sensor was added.

Gaze counts as embodied experience. In Steps 3–6, B denotes the remaining
contact, pose, motion, load and skin channels; G denotes steering gaze and its
attended close-up. This step changes B while keeping G fixed in the subsequent
comparison.

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
encoder-size changes were open. They are resolved below from available data and
training/development evidence, without test scores.

## Frozen candidate — 10 November 2026

Rich B is frozen for Step 4. It changes how existing recordings are summarized;
it adds no new sensation. Choices were made from data audits and one
pre-declared train→dev rule. No test or s8 data were used.

### What the audit found

- The retargeted GRAB source is 120 Hz for all 1,063 export clips (100%
  coverage). All 102,720 exported seconds map back to an exact source frame.
- Timing loss is large. In 28 sampled training clips, 99% of contact onsets
  and offsets fall between the 1 Hz samples. Brief contacts last a median 92 ms.
- Spatial loss is small by exact count. Only 6.8% of contacting training
  samples share a 22-patch vector with a different vertex pattern.
- 632 clips use substituted objects. Their contact is recomputed against the
  substituted mesh (5 mm rule). All 105 substituted meshes are on disk.

### Channel specification

Window: the same 11 steps over [t − 5 s, t + 5 s], bounded by the clip. Each
step summarizes the 30 Hz source frames (every 4th 120 Hz frame) in the 1 s
bin centred on its existing 1 Hz frame, within its own clip. The two outer steps
use only the half-bin inside the span, so the body window does not grow.

| Group | Dims | Content | Units / normalization |
|---|---:|---|---|
| Contact | 48 | 22 patch fractions, time-averaged over the bin (same map as current B) | raw [0, 1] |
| | | Per hand: share of bin frames with any contact; contact onsets in the bin | standardized |
| | | Per hand, wrist frame, over contacting frames: contact centroid (3), principal extents (3), mean surface normal (3), opposition = 1 − \|mean normal\| (1), span = largest distance between touching points (1) | metres; standardized on contact bins; 0 when no contact |
| Fingers | 100 | Bin-mean 90 axis-angle values; mean joint angular speed per finger (10) | rad, rad/s; standardized |
| Arm | 31 | Bin-mean elbow/wrist positions (12) and velocity from 30 Hz central differences (12), head-camera frame; speed std per joint (4); gravity in camera frame (3) | m, m/s; standardized |
| Load | 10 | Unchanged from current B (already from 120 Hz, ±0.5 s) | standardized |
| Skin | 4 | Unchanged from current B | standardized |
| **Total** | **193** | Current B: 153 | |

- Sources: native clips use GRAB contact labels; swapped clips use contact
  recomputed at every 30 Hz frame. Geometry uses only the hand's own touching
  vertices; no object geometry, pose or identity enters any feature.
- Normalization statistics come from the selected training clips only (630
  clips, 63,153 rows), through the same fit mask as current B.
- Missing seconds are handled as in current B.
- Encoder: unchanged family and width (per-group MLP 64, bidirectional GRU 64).
  Only first-layer input sizes change: encoder parameters 179,520 → 182,080;
  full additive learner 1,817,387 → 1,825,107; encoder compute 1.96M → 1.99M
  multiply-adds per example.
- Export: `nursery/simulator/body_rich.py`, 83 min on 8 CPU workers, 595 MB in
  `outputs/renders/clean_856/body_rich/` (ignored). Learner option:
  `representation="rich"` in `load_store`, `--body-representation rich` in the
  learner CLI. The current representation stays the default. Gaze-world wiring is
  Step 4's work.

### Open choices, resolved

- **Sample rate:** 30 Hz summaries per 1 s step, not a longer sequence. This
  catches brief contacts while keeping sequence length and encoder size fixed.
- **Contact granularity:** 22 patches. Rule fixed in advance: contact-only
  logistic probe for held category, train → dev; choose the smallest
  granularity within 1 point of the best. Dev accuracy: 22 patches 24.2%,
  32 phalanx regions 24.8%, 64 phalanx × palm/back regions 23.9%. A secondary
  leave-one-subject-out check within training favoured 64 (26.6% vs 24.3%);
  it did not select.
- **Pose encoding:** axis-angle kept, averaged over the bin, plus finger speed.
  Contact geometry carries the hand-shape information around contact.

### What became distinguishable

Five pairs of training seconds have identical current 22-patch vectors but
different contact (`outputs/analysis/step3_body/build/contact_examples.csv`).
For example, a water-bottle lift and a duck lift both read as "right thumb tip
22%" in current B. Rich B separates them: the bottle is a whole-hand grasp in
contact 70% of the second with an 8.2 cm span; the duck is a light thumb–index
touch with three onsets and a 4.9 cm span.

### Checks before Step 4 (all passed)

- Alignment: 0 of 102,720 centre frames differ from the existing contact export
  (native and swapped); hand pose and joints match exactly.
- Isolation: no bin uses frames outside its clip; outer half-bins and clip
  resets are unit-tested.
- All values finite. Coverage: 1,063 clips, 856 sessions, 102,720 rows.
- Original body and physical exports unchanged (1,712 files hash-verified).
- Information survives encoding (train → dev, diagnostic labels only):
  - All-body held-category probe: current B 23.7%, rich B 31.7% (9,517 dev
    contact seconds).
  - Brief contacts (under 1 s) visible in the step input: current B 32%,
    rich B 100% (1,005 per-hand episodes).
- Three CPU training steps through the additive learner ran with finite,
  nonzero encoder gradients. Full test suite: 258 passed.

These are input diagnostics, not language results. Whether rich B helps word
learning is Step 4's question. Evidence: `outputs/analysis/step3_body/`.

### Still unavailable (deferred)

Pressure and its spatial distribution, friction, texture, stiffness and
deformation, and spatial heat transfer. All need a contact-force or material
model, or measurements. Load and skin remain the existing proxies. Active
exploration remains deferred.
