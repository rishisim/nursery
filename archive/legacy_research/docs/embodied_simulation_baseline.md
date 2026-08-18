# Embodied Simulation scientific baseline

Status: architecture reset

Baseline date: 2026-07-27
Complete pre-reset tree: Git commit `7571c0e`

## Scientific purpose

Build a small but credible prompt-to-episode system whose eventual output is an
approximately 60-second child-view episode plus synchronized simulator-truth
streams. The causal question is whether training-only embodied side
modalities—action state, proprioception, contact/touch, IMU-like signals, and
object state—improve transferable lexical and language grounding when
naturalistic vision and speech are weakly aligned. A negative or inconclusive
causal result is acceptable if the design is sound.

Michael C. Frank's July 6, 2026 bridge governs the program: measure selected
aspects of the natural child-data distribution, reproduce those aspects in
simulated data while generating otherwise unavailable physical side streams,
and evaluate the same learner apples-to-apples. A free-standing synthetic
oracle without a naturalistic calibration anchor is not the target.

## Intended causal architecture

The preferred causal spine is:

`prompt → structured activity/language plan → developmentally profiled,
scene-conditioned action policy → embodied body/head/hand controller →
deterministic physics in a richly furnished interactive scene → synchronized
truth streams and rendering controls → baseline render → optional tightly
conditioned appearance render → timed speech/audio → cross-modal QA`

The child's actions and embodiment must produce the camera trajectory. Camera
pose should arise from locomotion or root motion, torso and neck/head
orientation, attention targets, a camera mount transform, and bounded
event-linked residual motion. An independently generated camera path that a body
is forced to follow is out of scope.

The first prototype should use a bounded action vocabulary: look, reorient,
approach, reach, touch, grasp, inspect, rotate, shake, bang, transfer, drop, and
retrieve. An LLM may propose a structured activity plan but may not directly
invent unconstrained joint trajectories. Physics, collision constraints,
contact, and object state are authoritative.

Diffusion or world-model video may later serve as a conditioned appearance layer
or action proposal. It is never the source of truth for contact, touch,
proprioception, IMU, or object state. Any video-first inverse-simulation route
must physically execute and rerender retargeted actions.

## Empirical and governance boundaries

- ChildLens is the sole empirical child-data source for the current prototype.
  AEA and BabyView measurements, examples, vocabulary, checkpoints, and
  empirical ancestry must not enter ChildLens calibration.
- ChildLens contains naturalistic child-centered data from ages 3–5. It permits
  provisional young-child developmental calibration, never final infant
  calibration and never an "infant-trained" claim.
- BabyView is a future, separately governed infant/toddler confirmation study.
- Predominantly German ChildLens recordings have no German-speaking human
  annotator available. Model–model sensitivity analyses must not be described
  as human validation or ground truth.
- Raw or restricted ChildLens media stays local. Only permitted aggregates may
  be exported.
- The external drive is outside routine repository operations. Do not copy,
  decode, move, rename, delete, or expose its restricted media without explicit
  task authorization.
- The frozen Qwen3-VL/Gemma 4 categorical pseudo-calibration failed its
  pre-outcome abstention/null-envelope gate. The gate must not be relaxed or
  reinterpreted, and the outcome was not a causal null.

## Established engineering evidence

The pre-reset repository established useful engineering facts without selecting
the next platform:

- A deterministic spec-driven MIMo–MolmoSpaces kernel produced synchronized
  physics, contact, proprioception, vestibular/IMU-like, object, camera, depth,
  segmentation, speech-timing, and rendering streams.
- Contact-before-grasp, a zero-contact near miss, lift, transport, release,
  settling, and exact trace reproducibility were demonstrated in bounded cells.
- An MPFB continuous hand/forearm replay aligned closely to the exported
  skeleton and removed an earlier disconnected collision-geometry proxy.
- Furnished FloorPlan1 and FloorPlan201 contexts were exercised.
- A 24-episode conditioned batch passed its timing and physics gates.

These facts demonstrate component feasibility, not naturalistic visual
qualification. The same batch failed all three frozen visual-distribution
intervals:

| Feature | ChildLens 90% interval | Generated mean |
| --- | ---: | ---: |
| Grayscale motion | 0.1095–0.1273 | 0.01635 |
| Adjacent DINO persistence | 0.6912–0.7330 | 0.96843 |
| Scene-change rate | 0.1069–0.1558 | 0.00000 |

A bounded camera-nuisance repair overshot motion and scene change while
persistence remained too high. Arbitrary jitter is therefore not an authorized
repair. Some historical renders also exposed visible collision-body or
pink-limb artifacts.

Exact implementations, tests, configs, receipts, and decision reports for these
historical results remain available at commit `7571c0e`. They are evidence and
candidate precedents, not active architecture.

## Frozen viewer–view hardware/privacy gate

The official ReViV plus ViPE calibration route stopped at
`STOP_HARDWARE_PRIVACY_NO_LOCAL_CUDA`. The available Apple Silicon/Metal host
cannot run the official CUDA 12.4 paths, and restricted ChildLens video cannot
be sent to remote compute. This was a hardware/privacy decision, not a
measurement-validity result, ChildLens result, controller result, appearance
result, or causal null.

The frozen configuration remains at
`configs/childlens_viewer_view_preflight_v1.json`, and its decision record
remains at
`docs/childlens_viewer_view_preflight_v1/decision_report.md`. Their historical
runner, tests, and referenced receipts are recoverable from commit `7571c0e`;
they are intentionally absent from the reset branch until a governance-approved
local CUDA route exists.

## Next scientific gate

Before rebuilding the larger system, run a bounded architecture-and-simulator
feasibility study using only public or synthetic inputs. Compare viable
rich-scene substrates rather than assuming one. Candidate references include
the historical engine, ProcTHOR/AI2-THOR, Habitat, EgoGen/EgoInteract when
practically available, and OmniGibson only if its NVIDIA requirements are
realistically satisfiable.

The gate should attempt one deterministic 15–20 second episode:

`look → reach → grasp → inspect → manipulate → release`

Minimum evidence:

- a body-derived head-mounted camera;
- a richly furnished interactive scene;
- no visible or physical penetrations;
- persistent object identity;
- contact-authoritative grasp and manipulation;
- synchronized RGB, depth, segmentation, action, proprioception, contact/touch,
  IMU-like, object-state, and camera streams;
- deterministic replay and compact provenance;
- acceptable asset and code licensing;
- realistic execution on available hardware.

The gate is an engineering and simulator-feasibility study. It must not inspect
restricted ChildLens media, claim distributional calibration, launch a learner,
or run a causal cue-lift experiment.

## Repository policy after reset

No implementation is canonical until a simulator gate is explicitly authorized
and scoped. New work should update one canonical implementation and protocol in
place. Generated media, datasets, complete runs, caches, downloads, logs,
checkpoints, and rendered reports must remain in ignored or OS-temporary roots.
Git should retain only compact frozen configs, aggregate result tables,
provenance manifests, and concise decision reports.
