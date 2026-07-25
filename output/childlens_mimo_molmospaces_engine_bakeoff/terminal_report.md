# MIMo–MolmoSpaces bounded engine bake-off

## Terminal decision

`NO_GO_ENGINE_LADDER_EXHAUSTED`

This is not the prior six-clip NO_GO repeated. The approved ladder was actually
run: pinned stock MIMo, pinned stock MolmoSpaces with classic and attempted
Filament renderers, MolmoSpaces custom-robot API attachment, direct MIMo+iTHOR
MJCF composition, causal push and post-contact grasp, a zero-reported-contact
control, a second furnished scene, synchronized German speech, and exact
trajectory replay in Blender.

The combination is technically viable as a sensor-rich prototype, but it is not
qualified for the distribution bake-off. The decisive failures are visible:

1. The near-miss telemetry reports zero hand–target contact, yet inspected frames
   show the hand geometry visibly intersecting/overlapping the ball. It is
   therefore not a true zero-contact visual control.
2. The canonical native close vest view is dominated by self-occluding MIMo
   collision geometry. The full five-finger hand exists and is causally active,
   but the camera evidence is not naturalistic.
3. Blender exactly replays the causal transforms and clears target prominence,
   but its articulated hand remains a disconnected joint-blob visual proxy, not
   a validated high-quality skin. It cannot be used to conceal the native
   geometry/contact mismatch.
4. The second integrated furnished context (iTHOR FloorPlan201) rendered, but
   produced neither a visible target nor causal contact. Two qualified scene
   contexts were therefore not demonstrated.
5. The post-contact spring grasp is causal and moves the object 14.1 cm in 3-D,
   but only 1.22 cm vertically. It verifies the constraint interface; it does not
   establish a convincing lift/place.

## What ran and what passed

- MIMo v2.0.0 at commit `040b0ae4914cbfb26afdf830aa81775b90922f3f`
  ran natively on arm64 with the 24-month full five-finger model. The stock
  episode retained RGB, 198-position state, 183-velocity state, 621-value
  proprioception, 14,055-value touch, and six-value vestibular streams on one
  clock. This passes the embodiment/sensor preflight; its stock room and view
  fail appearance.
- MolmoSpaces 0.2.0 at commit
  `c2f1b583f087e1d3994e1377574843b759d9d0f8` loaded iTHOR FloorPlan1 with
  242 bodies, 2,116 geoms, 1,493 meshes, 44 textures, and 67 joints. The classic
  renderer produced a genuinely furnished kitchen plus depth, segmentation, and
  state metadata at about 93 rendered frames/s.
- Filament was attempted through the documented extra and a direct package
  route. The available wheel is Linux x86_64; no native arm64 macOS wheel was
  available. Classic MuJoCo was the documented fallback, not an inferred
  ecosystem failure.
- MolmoSpaces route A (`Robot.add_robot_to_scene`) compiled MIMo with 90
  actuators and 56 sensors. A complete high-level runtime adapter still requires
  bespoke `RobotView`, humanoid kinematics, and controller contracts. Route B
  directly composed and rendered the full model, so the conditional mesh-import
  route C was not required.
- The repaired push moved a real-scale 10 cm ball 9.66 cm through sampled
  hand contact. The grasp constraint is engaged only after physical contact.
  Speech is local macOS Anna German synthesis synchronized to episode time; it
  verifies timing/interface only and is not human validation.
- Blender replay used the exported 240-frame body, target, and camera transforms
  without rerunning physics. F-curve validation found a maximum body-position
  discrepancy of `2.97e-8 m`. Target area across the frozen mild Brown–Conrady,
  diagonal 140-degree equisolid, and mild polynomial models had a 2.365% minimum,
  above the frozen 1.5% threshold.

## Visual inspection

Inspection covered beginning/pre-action, contact/closest approach, and
post-action frames rather than exit codes alone. Stock MolmoSpaces is furnished
and substantially richer than the prior procedural rooms. Native integrated
frames show a real five-finger hand, but also severe arm/palm occlusion. The
near-miss sheet visibly contradicts the nominal zero-contact stream. Blender
improves target/room legibility and retains exact transforms, but its finger
segments remain visibly disconnected.

No generative VLM was used as referential truth or as the acquisition learner.
No ChildLens raw frame, audio, transcript, identifying path, AEA/BabyView
measurement, or private material was inspected or included.

## Developmental and claim boundary

ChildLens remains the sole empirical source and covers ages 3–5. The simulated
embodiment is MIMo at 24 months, within MIMo's validated range but not age-matched
to ChildLens. The narrow permitted description is: **ChildLens-calibrated
environmental/timing statistics with an infant-like simulated embodiment**.
This is provisional developmental calibration, not final infant calibration.
The predominantly German speech route is model-generated timing evidence, not
human validation or ground truth.

## Throughput and storage

The exercised hybrid ran at roughly 24–26 rendered frames/s on the M5. Stock
MIMo ran at about 38 simulation steps/s and stock MolmoSpaces at about 93
rendered frames/s. Ignored local environments/cache use 7.1 GiB and the external
bake-off tree uses 16 GiB, below the 50 GiB bound. The external volume retained
about 160 GiB free.

## Exact next step

Do not start the distribution generator or learner. The smallest next
qualification task is a geometry-only repair: bind a license-clean continuous
five-finger skin to the exported MIMo hand skeleton, validate skin-to-collision
surface distance and contact alignment numerically, construct a collision-aware
near-miss controller with a positive minimum-separation certificate, and rerun
the same frozen FloorPlan1/FloorPlan201 cells. The current causal telemetry,
camera set, reach envelope, and gates should remain unchanged. Only after that
repair passes both scene contexts should the project proceed toward Michael
Frank's distribution-matched simulator bridge.

## Evidence map

Generated media remain ignored/external:

- stock MIMo: `/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/media/stock_mimo/`
- stock MolmoSpaces: `/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/media/stock_molmospaces/`
- causal push: `/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/media/hybrid_push/`
- grasp probe: `/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/media/hybrid_lift/`
- near-miss failure: `/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/media/hybrid_near_miss/`
- second-context failure:
  `/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/media/hybrid_push_floorplan201/`
- exact replay and side-by-side:
  `/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/media/blender_replay_push/`

Compact receipts, the repair ledger, provenance ledger, decision matrix, and
aggregate evidence are stored beside this report.
