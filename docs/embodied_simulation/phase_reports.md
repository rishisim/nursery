# Embodied simulation phase reports

This file is the canonical compact checkpoint record for the adaptive Phase 0–6
build. Complete runs, media, assets, caches, and logs remain under ignored roots.

## Phase 0 — contract and environment freeze

Gate decision: **PASS**.

The 19.5-second vertical slice, scene variants, immutable head-camera mount,
rates, seeds, collision policy, physical-first grasp fallback, numerical and
visual gates, exact public component pins, execution environments, and output
layout are frozen in `configs/embodied_simulation_vertical_slice.json` before
Phase 1 outcomes.

Recovered canonical source paths from commit `7571c0e` comprise the existing
`babyworld_lite/childlens_engine_bakeoff` package, its six relevant configs, and
three focused historical test files. No historical output, run, render,
checkpoint, cache, copied asset, or dataset was recovered. The recovered kernel
is explicitly an unqualified repair starting point because it contains the
known chest/world camera, hand mocap weld, object weld, and runtime collision
disabling defects.

Validation and preflight evidence:

- Five new contract tests pass; the entire recovered static test set is checked
  before Phase 1 work begins.
- Public source checkout verified exact commits: MIMo
  `040b0ae4914cbfb26afdf830aa81775b90922f3f`, MolmoSpaces
  `c2f1b583f087e1d3994e1377574843b759d9d0f8`, and MPFB
  `f4f4f1ffa8203585730a7ce433b66738777ba168`. License files confirm MIT,
  Apache-2.0, and GPL-3.0 code/CC0 core assets respectively.
- Local host: Apple M5 arm64, 32 GiB, macOS 26.6; Python 3.11 and FFmpeg 8.0.1
  are available, CUDA is not.
- Juno login preflight found Slurm `a30`, A30 MIG, `h100`, H100 MIG, and `h200`
  partitions with 6–80 GiB GPU forms, CUDA 12.4, SingularityCE 4.2.2, 46 GiB
  free home storage, and 123 TiB free shared scratch. Hugging Face returned HTTP
  200 using public traffic. A bounded A30 compute-node probe queued without an
  allocation and was cancelled cleanly; compute-node driver version therefore
  remains an explicit pre-run check rather than an inferred value.

Actual artifacts: exact public source checkouts are in ignored `.external/`
directories. No Phase 0 run media was produced. The canonical future run root
is ignored `runs/embodied_simulation/phase_<n>/<run_id>`.

Repository status at checkpoint: recovered source/config/tests and the frozen
contract/report are ready for focused validation, commit, and push. No generated
artifact is staged for commit.

Deviation from the requested preflight: the Juno A30 driver query could not run
because the requested allocation remained pending. The queue request was not
left behind, and the missing driver value is recorded without weakening any
scientific threshold.

Smallest next step: replace the recovered chest/mocap/weld kernel with an
articulated head-camera/body-hand implementation while preserving the frozen
contract and static collision policy.

## Phase 1 — embodied geometry and physical authority

Gate decision: **PASS**.

The 19.5-second vertical slice now executes as one deterministic MIMo/MuJoCo
trace in the furnished FloorPlan201 scene. The head-view camera is derived only
from the articulated root/torso/head chain and the immutable mount. The right
arm, forearm, palm, and five fingers use a static physical collision layer that
remains enabled for the entire episode. RGB exposes a separate co-articulated,
non-colliding native MIMo appearance layer with the clean `skin` material.

Canonical files changed:

- `physics_kernel.py`, `run_kernel_episode.py`, `trace_render.py`, and
  `determinism.py` implement the authoritative trace, gates, replay, and bundle;
- `mpfb_overlay_renderer.py` and `depth_composite.py` retain bounded diagnostic
  appearance tooling without governing the Phase 1 result;
- `configs/embodied_simulation_vertical_slice.json` records exact runtime pins,
  the fixed mount/scene placement, and the native-MIMo appearance decision;
- focused coverage was added to `tests/test_childlens_engine_bakeoff.py` and
  `tests/test_embodied_simulation_contract.py`;
- compact metrics are in `docs/embodied_simulation/aggregate_results.json`.

Validation and experiment evidence:

- Full repository validation passes: 13 tests.
- Physics produced 4,680 steps and 1,171 synchronized truth samples. Two
  complete executions produced byte-identical trace SHA-256
  `92fc1f5f55e06698d93497f18f4c7f29015dad1f014e3218a32797c3af724004`;
  maximum numeric replay error is 0.
- Maximum camera-mount errors are `4.44e-16` m and `4.89e-16` rad. Shared-clock
  error is `7.85e-13` s; IMU-like acceleration and gyro comparisons both have
  RMSE 0; persistent object identity changes are 0.
- The static 649-geom relevant collision policy hash remains unchanged. The
  minimum relevant distance is `-0.00160093` m, below the frozen 0.002 m depth
  allowance, with 0 persistent penetration frames. The near miss has 0 contact
  substeps and 0.03857 m clearance.
- First contact occurs at 7.5125 s, reaches five distinct finger bodies and
  67.50 N. After two physical attempts, the frozen contact-gated soft assist
  engages at 9.15 s following 1.4917 s of multipoint evidence; its engagement
  jump is 0.000167 m / 0.0167 degrees and collisions remain enabled.
- Lift is 0.08550 m, object rotation 80.84 degrees, head turn 22.83 degrees,
  and head-turn contact retention 1.0. Release is physical; the final settled
  window is 2.433 s with 0.01415 m/s maximum speed.
- The authoritative render has 586 frames at 640x480/30 fps. Collision-proxy
  and skin-artifact pixel counts are both 0. Contact/release frame offsets are
  0; maximum visible contact error is 4.444 px and 0.005990 m; the target is
  visible in every rendered frame. Manifest hashes and HDF5 stream shapes were
  independently rechecked, and FFprobe confirms 586 H.264 frames.

Actual artifacts are under ignored
`runs/embodied_simulation/phase_1/qualified`: `baseline_rgb.mp4`,
`external_qa.mp4`, `episode_trace.npz`, `episode_trace_replay.npz`,
`render_streams.h5`, QA JSON records, `inspection_sheet.png`, and
`episode_bundle_manifest.json`. The ignored `mpfb_diagnostic/` comparison is
retained; it aligns landmarks to 0.000271 m maximum but fails its projected
contact appearance check and is not authoritative. Superseded preflight and
all-frame MPFB-composition intermediates were moved to the macOS Trash and are
recoverable; no pre-existing user run was removed.

Repository status: implementation commit `4a5bb6c` (`Qualify embodied MIMo
vertical slice`) is pushed to `origin/embodied-simulation`. Complete runs,
media, dependency checkouts, and logs remain ignored.

Deviation from the preferred architecture: following the explicit design
decision and bounded visual comparison, native MIMo is the authoritative
deterministic visible hand/forearm and MPFB is diagnostic only. This changes no
frozen physics, camera, synchronization, contact, penetration, or determinism
threshold. A one-time fixed camera-mount calibration was made before hard-gate
qualification because the provisional eye-axis transform faced the floor; the
mount is immutable during execution. Root translation is explicitly recorded
as bounded collision-checked locomotion assist, and grasp assist is fully
flagged.

Smallest next step: instantiate the unchanged event semantics in sparse,
household, and messy clutter variants and run the Phase 1 hard gates per scene.

## Phase 2 — scene and controller stress qualification

Gate decision: **PASS**.

The unchanged 19.5-second event sequence passes every Phase 1 physics,
camera, contact, synchronization, identity, determinism, rendering, and native
appearance check in the required sparse (3 distractors) and household (12
distractors) FloorPlan201 variants. The 24-distractor messy variant is retained
as the explicitly permitted bounded diagnostic failure: the released cup meets
authored `clutter_15` at 17.0125 s, reaches -0.005532 m with 8 persistent
penetration frames, and therefore fails the unchanged 0.002 m / 0-frame gate.
Its replay is byte-identical, the failure pair and time are stable, and it does
not affect either required variant.

Canonical files changed:

- `physics_kernel.py` adds co-located physical and visible clutter layers while
  excluding static clutter from MIMo body dynamics;
- `run_kernel_episode.py` resolves the frozen ordered scene-family prefixes and
  records variant provenance without changing action semantics;
- `trace_render.py` keeps group-4 scene/clutter collision proxies hidden from
  authoritative RGB and measures visible authored clutter;
- `run_scene_stress.py` is the canonical three-variant Phase 2 entry point;
- `configs/embodied_simulation_vertical_slice.json` freezes the 23 placements
  before Phase 2 outcomes, and focused tests plus the compact aggregate record
  were updated in place.

Validation and experiment evidence:

- Full repository validation passes: 15 tests. All three episode manifests were
  rehashed successfully; each HDF5 bundle contains 586 depth frames at 480x640,
  586 segmentation frames at 480x640x2, and 586 clutter-area samples. FFprobe
  confirms 586 H.264 frames at 640x480/30 fps for every variant.
- Every scene produces 4,680 physics steps and 1,171 truth samples. Camera-mount
  errors remain `4.44e-16` m / `4.89e-16` rad; shared-clock error remains
  `7.85e-13` s; object identity changes and replay numeric error remain 0.
- Sparse and household both keep the true near miss at 0.03857 m with 0 contact
  substeps, first contact at 7.5125 s with five finger bodies, and the flagged
  assist at 9.15 s with a 0.000167 m engagement jump. Their minimum relevant
  distance is -0.001601 m with 0 persistent penetration frames.
- Sparse uses 651 relevant collision geoms; authored clutter is visible in
  0.2986 of frames and covers at most 0.01006 of the image. Household uses 660
  relevant geoms; clutter is visible in every frame and covers at most 0.04291.
  Both render contact to 4.444 px / 0.005990 m, with 0 collision-proxy pixels
  and 0 skin-artifact pixels.
- Messy uses 672 relevant collision geoms. All non-penetration checks pass,
  including its exact replay SHA-256
  `ee0e42e3e7b88fa710fae56b00acba52a0d2b9a5de0ab1d5ad089a91445d80b5`;
  clutter is visible in every frame and reaches 0.08733 image area. Its isolated
  release/clutter penetration is the sole failed hard check.

Actual artifacts are under ignored
`runs/embodied_simulation/phase_2/stress`, with per-scene baseline/external
videos, synchronized traces and render streams, QA receipts, contact sheets,
and hashed episode manifests. `phase_2_stress_qa.json` is the ignored aggregate
run receipt. Task-created preflight runs, stdout logs, and the MuJoCo root log
were moved to the macOS Trash and remain recoverable; no pre-existing run was
removed.

Repository status: verified implementation and compact aggregate commit
`abb6f7c` (`Qualify embodied scene stress variants`) is pushed to
`origin/embodied-simulation`; this checkpoint record is committed and pushed
separately. No generated run artifact is tracked.

Deviation from the Phase 0 freeze: the exact root-relative clutter placements
were resolved and frozen immediately before the first Phase 2 outcome because
Phase 0 had frozen counts and semantics but not coordinates. No action, physics,
camera, contact, synchronization, appearance, or determinism threshold changed.
The messy failure was not repaired by moving a post-outcome placement or
weakening a threshold; it remains the allowed reproducible diagnostic. The
native-MIMo appearance decision remains in force and MPFB was not rerun.

Smallest next step: extract the four frozen 3–5 second windows from a qualified
required-scene trace, then preflight and run geometry-protected Cosmos 3 Nano
and OSCAR-2B appearance cells on Juno using public/synthetic inputs only.

## Phase 3 — controlled appearance experiments on Juno

Gate decision: **PASS** (deterministic baseline retained; no neural render was
accepted).

All 36 frozen cells completed: four qualified trace windows, three seeds, and
three render conditions. The deterministic baseline is the only authoritative
acceptable render. Cosmos 3 Nano failed the frozen output-resolution invariant
in every cell. OSCAR-2B passed the automated protected-foreground checks, but
manual artifact review rejected every cell for changes outside the protected
core that altered visible hand/object geometry or occlusion. This is the
explicit Phase 3 option-(b) pass and does not change simulator truth.

Canonical files and compact records changed:

- `appearance_experiment.py` prepares frozen cells, validates outputs, performs
  protected-foreground compositing, and writes compact QA receipts;
- `juno_appearance_setup.sh` and `juno_appearance_run.sh` pin and execute the
  public Cosmos 3 Nano and OSCAR-2B environments on Juno;
- `configs/embodied_simulation_appearance.json` freezes windows, seeds,
  conditioning, masks, resolution, timing, acceptance rules, and exact source
  and model revisions;
- `tests/test_embodied_appearance.py` covers cell preparation, mask protection,
  resolution/timing rejection, audio rejection, and acceptance logic;
- `docs/embodied_simulation/aggregate_results.json` records the compact outcome.

Validation and experiment evidence:

- The baseline produced 12/12 valid cells at 640x480 and 15 fps with exact
  frozen frame counts (49, 49, 61, and 53 by window), zero audio streams, and
  byte-identical output across seed labels within each window.
- Cosmos 3 Nano generated 12/12 cells with exact frame counts and no output
  audio, but all were 736x544 rather than the frozen 640x480. All 12 were
  rejected; no post-outcome crop, resize, seed, mask, or threshold change was
  applied.
- OSCAR-2B generated 12/12 cells at the required resolution, rate, and length.
  All passed automated invariants: protected-core decoded and encoded maximum
  pixel error 0, camera translation/rotation error 0, event-frame offset 0,
  authoritative object-identity changes 0, and audio streams 0. The appearance
  proxy improved in 2/12 cells.
- All 12 OSCAR inspection sheets were reviewed. Every cell was rejected for one
  or more hard visual failures outside the protected core: duplicate/oversized
  or malformed hands, changed finger geometry/count, duplicated or changed cup
  pose/identity, and incorrect occlusion/release state. The two proxy-improved
  cells failed the same hand-geometry requirement.
- Juno setup job `311790` and execution job `311795` passed; execution took
  42:29 on an NVIDIA H200 NVL with 143,771 MiB, driver 550.163.01, and reported
  CUDA 12.4. Samples had sound generation disabled and all accepted candidates
  were required to contain no audio.
- Exact public pins include Cosmos source
  `404b9bf2144640834c63ae7d9e7269e0f4ea02cb`, Cosmos3 Nano
  `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`, OSCAR source
  `4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb`, and OSCAR-2B
  `c9781ffa7dd8556d862d7d9f338a2ea008a58ca6`; the full dependency receipt is in
  the ignored run.

Actual artifacts are under ignored
`runs/embodied_simulation/phase_3/appearance`: prepared conditioning inputs,
12 baseline cells, 12 raw Cosmos outputs, 12 raw OSCAR outputs, protected
composites, per-cell QA JSON, inspection sheets, and the manual-review record.
The corresponding Juno run is at `/work/dal503972/embodied_phase3`; only compact
records and manifests are committed. No ChildLens media or other restricted
input was read, copied, decoded, or transferred.

Repository status: the verified Phase 3 implementation and bounded Juno repair
commits through `5fe84d4` are pushed to `origin/embodied-simulation`; this
checkpoint and compact aggregate are committed and pushed separately. Complete
runs, model weights, environments, downloads, media, and logs remain ignored.

Deviations from the frozen plan were bounded to execution compatibility. Juno
required pinned CUDA/NPP and FFmpeg/PyAV library exposure, and Cosmos model
construction required its checkpoint-bundled public sound tokenizer even
though sound generation was disabled. Several failed setup/execution attempts
are preserved in ignored receipts. The frozen cells, seeds, source trace,
masks, event timing, geometry rules, and thresholds did not change. Cosmos's
resolution mismatch was rejected rather than repaired after seeing the result.

Smallest next step: add the canonical bounded prompt/language composer and
compile one uninterrupted approximately 60-second authoritative physics trace
with synchronized speech, then render and bundle it without hidden resets.

## Phase 4 — prompt, language, and continuous 60-second candidate

Gate decision: **PASS**.

One 60.0-second prompt-to-episode candidate now compiles, executes, replays,
renders, receives separately generated speech, and bundles end-to-end. It is a
single uninterrupted MuJoCo trace with no hidden reset. The deterministic
native-MIMo baseline remains authoritative; no Phase 3 neural output was used.

Canonical files and compact records changed:

- `compile_episode.py` resolves the prompt into a bounded action/language plan
  and validates the real `yellow_cup_authored` scene identity before execution;
- `physics_kernel.py` composes look/reorient/approach/reach/touch/grasp,
  inspect/rotate/head-turn, shake/bang/transfer, physical release/settle, and a
  second contact-gated retrieve without directly posing the object or camera;
- `run_continuous_episode.py` is the canonical compile/execute/render/audio/QA
  entry point, and `speech_audio.py` creates authoritative local speech plus
  transcript and deterministic timing records;
- `configs/embodied_simulation_episode.json` freezes the 60-second schedule,
  six utterances, action vocabulary, speech settings, and Phase 4 gates;
- `configs/embodied_simulation_vertical_slice.json` now records the authored
  collision-enabled support catch tray used for physical put-down/retrieval;
- focused coverage is in `tests/test_embodied_continuous_episode.py`, with the
  compact outcome in `docs/embodied_simulation/aggregate_results.json`.

Validation and experiment evidence:

- Full repository validation passes: 24 tests. Both the 30-file continuous
  manifest and nested 17-file simulator manifest rehash without mismatch.
  HDF5 contains 1,801 depth frames at 480x640 and 1,801 segmentation frames at
  480x640x2. FFprobe confirms the accepted mux has 1,801 H.264 frames at
  640x480/30 fps plus one 48 kHz mono AAC stream and lasts 60.033333 seconds.
- Physics produced 14,400 steps and 3,601 synchronized truth samples from 0 to
  60 seconds. Independent execution produced the exact same trace SHA-256
  `c0519057e3a55aa6eeef23cbc85a3f2190d5e4ed9a3211e1655268a4bca8968e`;
  maximum replay error is 0, object identity changes are 0, hidden resets are
  0, and the largest consecutive target-position step is 0.012021 m.
- Camera-mount error is `4.44e-16` m / `5.07e-16` rad. Shared-clock error is
  `2.08e-11` s and both IMU-like RMSE values are 0. All 661 relevant collision
  geoms stayed enabled; minimum relevant distance is 0.0 m with 0 persistent
  penetration frames.
- The near miss remains non-contact at 0.038999 m. First cup contact occurs at
  10.5125 s with five finger bodies. Two physical attempts precede each of two
  flagged assists, engaged at 12.15 and 52.15 s; maximum engagement jump is
  0.000259 m / 0.0225 degrees and collisions remain enabled.
- Maximum lift is 0.11980 m, rotation 91.79 degrees, and head turn 22.85
  degrees with 1.0 contact retention. Shake spans 0.04032 m vertically, bang
  records a 4.37 N support contact, lateral transfer spans 0.21580 m, and the
  retrieve has 221 contact samples and lifts 0.11137 m. Physical release is
  followed by a 6.0-second settled interval with 0.06193 m/s maximum speed.
- Rendering produces 1,801 authoritative frames with 0 collision-proxy pixels,
  0 skin-artifact pixels, 0 camera replay error, and 0 contact/release frame
  offset. Maximum contact alignment is 5.891 px / 0.005997 m, within the
  unchanged 6 px / 0.006 m limits. The target is visible in every frame; the
  full inspection sheet was visually reviewed with no pink limb or exposed
  collision proxy.
- Six local English utterances produce an exact 60.0-second PCM16/48 kHz mono
  waveform with 0 clipped samples. All six starts match their planned behavior
  phase and show the target. Transcript, utterance intervals, deterministic
  word subdivisions, and waveform hashes are recorded; neural-render audio is
  absent. This is synthetic engineering language, not developmental or human
  validation.

Actual artifacts are under ignored
`runs/embodied_simulation/phase_4/candidate` (about 510 MiB):
`accepted_episode.mp4`, authoritative `baseline_rgb.mp4`, `external_qa.mp4`,
two exact traces, depth/segmentation HDF5, activity/language plan, speech WAV,
transcript/alignment, cross-modal QA, contact sheet, manifests, and replay
receipts. Superseded Phase 4 preflights and regression outputs were moved to a
recoverable macOS Trash location after their compact repair evidence was
consolidated. No pre-existing run was removed and no restricted ChildLens media
was accessed, decoded, moved, or transferred.

Repository status: plan commit `6304cd2`, implementation commit `ce6e7b8`, and
bundle-metadata commit `a556260` are pushed to
`origin/embodied-simulation`; this checkpoint and aggregate are committed and
pushed separately. Complete traces, media, dependency material, logs, and run
intermediates remain ignored.

Deviations from the initial Phase 4 freeze were bounded and explicit. Before
the first Phase 4 simulator outcome, utterance `u06` moved from 58.5 to 59.0 s
so it begins in `final_dwell`. Failed preflights then motivated phase-relative
IK tracking, support-aware hand feedback, a collision-enabled 0.18 m authored
catch tray, and a side/above retrieve approach. The tray keeps a physically
released cup reachable without a reset or direct object pose intervention; a
rerun of the 19.5-second Phase 1 physics qualification passed with the added
collider. No frozen camera, collision-enablement, penetration, contact,
synchronization, determinism, plan-action, seed, or acceptance threshold was
weakened or moved.

Smallest next step: run the frozen 12-episode engineering/generalization batch
across scene variants, targets/distractors, clutter, and seeds, then select the
final candidate by compact predeclared criteria rather than visual preference.

## Phase 5 — generalization and bridge-oriented qualification

Gate decision: **REPAIR-AND-CONTINUE → PASS**.

The frozen 12-cell matrix completed across sparse, household, and messy scene
variants; authored yellow-cup and red-ball targets; and two bounded placement
seeds. Nine episodes pass the unchanged continuous physics, synchronization,
determinism, render, speech, and cross-modal contract. Each scene contributes
three passes, both required scenes exceed the frozen batch minimum, and the
three remaining failures define an exact, bounded operating boundary rather
than an architecture gate.

Canonical files and compact records changed:

- `generalization_batch.py` materializes the frozen product matrix, validates
  bound contracts/assets, executes or revalidates each 60-second episode,
  measures the exact pinned DINO/grayscale bridge vector, applies the frozen
  selection order, and retains only compact cell and aggregate receipts;
- `physics_kernel.py` and `run_kernel_episode.py` support recorded bounded
  target/support/distractor offsets and both registry target geometries while
  preserving the Phase 4 base trace byte-for-byte;
- `trace_render.py` now projects the physical MuJoCo contact surface endpoints
  and uses authoritative depth to identify genuinely occluded contact points;
- `configs/embodied_simulation_generalization.json` freezes the 3×2×2 matrix,
  calibration instrument, gates, selection policy, and bounded QA repair;
- focused coverage is in `tests/test_embodied_generalization_batch.py`, and the
  compact outcome is consolidated in `aggregate_results.json`.

Validation and experiment evidence:

- Full repository validation passes: 30 tests. All 12 continuous manifests
  independently rehash, all bundles share one schema, and every HDF5 file has
  depth shape 1,801×480×640 and segmentation shape 1,801×480×640×2. FFprobe
  confirms every accepted mux has 1,801 H.264 frames plus one AAC stream.
- All 12 cells execute 14,400 physics steps and 3,601 truth samples. All replay
  with maximum numeric error 0; object identity changes, persistent penetration
  frames, unflagged assist frames, collision-proxy pixels, and skin-artifact
  pixels are 0 in every cell. Maximum mount error is `4.58e-16` m / `5.18e-16`
  rad and maximum shared-clock error is `2.08e-11` s. Near-miss clearance spans
  0.03125–0.11348 m with zero near-miss contacts.
- The repaired visible-contact check reaches at most 5.8915 px and 0.0059994 m,
  within the unchanged 6 px / 0.006 m limits. The first matrix attempt had only
  two required-scene passes because the QA calculation projected a MuJoCo
  margin midpoint and scored depth-occluded contacts. Projecting the two
  physical surface endpoints and excluding only points hidden by a nearer
  authoritative depth sample repaired the measurement; it did not change a
  trace, rendered geometry, seed, threshold, camera, controller, or collision.
- Nine cells pass end-to-end: cup 6/6 and ball 3/6; sparse, household, and messy
  each pass 3/4. All nine passing cells use two flagged contact-gated assist
  intervals totaling 34.3333 s (2,061 truth frames); none is an unassisted
  physical-grasp-only result. Across all cells, lift spans 0.11259–0.16171 m,
  rotation 91.79–117.69 degrees, head turn 22.85–23.05 degrees with 1.0 contact
  retention, shake 0.03623–0.04773 m, and transfer 0.06346–0.21580 m.
- The three failures are the red ball at seed `20260731` in all three clutter
  variants. The same trace SHA-256
  `0a0ed33bf7d72ee65570229c5ff1c2190130ab3abc9f3f285985dde2686b31c5`
  has 26 retrieve-contact samples but no second assisted lift and only one
  assist interval. Its failure is therefore target/placement controller scope,
  not clutter or nondeterminism; the red ball passes in every scene at the
  second frozen seed.
- Exact 1 Hz bridge measurement with hash-bound `facebook/dinov2-small` gives
  motion mean 0.06762 (range 0.06466–0.07207), adjacent DINO persistence mean
  0.81942 (0.81071–0.83886), and scene-change mean 0.00424 (0–0.01695). All
  12 cells miss all unchanged ChildLens intervals 0.1095–0.1273,
  0.6912–0.7330, and 0.1069–0.1558 respectively. This remains a transparent
  model–model/provisional young-child distribution diagnostic; it is not human
  validation, infant calibration, a simulator-truth failure, or a causal null.
- Inspection sheets for the selected cup, the passing ball, and the bounded
  failing ball were reviewed. Object identity and finger count remain stable;
  no pink limb, collision proxy, duplicate hand, or hidden reset is visible.
  The deterministic native-MIMo appearance remains simplified but acceptable.

Actual artifacts are under ignored
`runs/embodied_simulation/phase_5/batch` (about 5.6 GiB): 11 newly executed
episodes plus the revalidated Phase 4 reference cell, per-cell authoritative
videos/audio, traces, HDF5 truth renders, QA sheets, manifests, the first-attempt
compact receipts, final `batch_aggregate.json`, and
`visual_distribution_qa.json`. The exact frozen selection chooses
`sparse__yellow_cup_authored__20260731`, with trace SHA-256
`c0519057e3a55aa6eeef23cbc85a3f2190d5e4ed9a3211e1655268a4bca8968e`
and accepted mux SHA-256
`93885e1f6cdf9eaed34ab32d8795f489e67ee6541cc7795765a0330aa0f378c3`.
Task-created preflights and root MuJoCo logs were moved to the recoverable
macOS Trash after compact evidence was retained; no pre-existing run was
deleted.

Repository status: implementation commit `3fc6b1d` and bounded contact-QA
repair commit `d92c0d7` are pushed to `origin/embodied-simulation`; this
checkpoint and compact aggregate are committed and pushed separately. Full
runs, model weights, environments, downloads, media, and logs remain ignored.

Deviations from the frozen plan are limited to the recorded contact-projection
QA repair. The matrix, target definitions, placements, seeds, action semantics,
physics, render, speech, selection order, and all acceptance thresholds stayed
fixed. ReViV/ViPE was not run because its synthetic-only instrument check was
optional and could not substitute for the already frozen ChildLens hardware/
privacy stop or improve this engineering gate. No restricted ChildLens media
was accessed, decoded, moved, or transferred.

Smallest next step: assemble the selected ignored run into the final accepted
bundle, add a compact preview and replay/verification entry point, independently
rehash every component, and write the final cross-domain decision report.

## Phase 6 — accepted child-view episode and simulator-truth bundle

Gate decision: **PASS**.

The frozen Phase 5 selection is now an accepted 60.0-second child-view episode
and self-contained simulator-truth package. The episode is one continuous
MuJoCo trace with an immutable articulated-head camera mount, authoritative
native-MIMo RGB, separately generated speech, depth/segmentation, all physical
and embodied side streams, exact replay evidence, cross-modal QA, provenance,
and a 16-sample human-viewable preview. No neural appearance output is included.

Canonical files and compact records changed:

- `final_bundle.py` verifies the hash-bound Phase 5 source, copies only the
  required deliverables into the ignored final root, describes every NPZ/HDF5
  truth stream, builds the preview, records exact public-source/environment
  provenance and replay instructions, evaluates 19 final acceptance checks,
  and emits a hash manifest;
- `configs/embodied_simulation_final_bundle.json` freezes the selected cell,
  source/config hashes, required files and streams, preview samples,
  authoritative appearance policy, and unchanged acceptance/governance bounds
  before assembly;
- `tests/test_embodied_final_bundle.py` covers the frozen selection and
  deliverables, truth-schema construction, preview composition, strict gate
  behavior, and manifest corruption detection;
- this report and `aggregate_results.json` are the only compact Phase 6 outcome
  records retained in Git.

Validation and experiment evidence:

- Full repository validation passes: 35 tests. The final gate passes 19/19
  checks. An independent rehash finds zero mismatches across all 50 manifest
  entries; 51 files exist including the manifest. The copied trace and replay
  are byte-identical SHA-256
  `c0519057e3a55aa6eeef23cbc85a3f2190d5e4ed9a3211e1655268a4bca8968e`,
  and an independent array-by-array comparison is exact, including NaNs.
- Physics contains 14,400 steps and 3,601 truth samples from 0 to
  60.00000000002078 s. Hidden reset count, object-identity changes, persistent
  penetration frames, and replay numeric error are all 0. Camera-mount error is
  `4.44e-16` m / `5.07e-16` rad; both IMU consistency RMSE values are 0.
- The near miss remains non-contact at 0.038999 m. RGB contact/release offsets
  are both 0 frames. The latest bound surface/depth contact QA evaluates 808
  observations (301 visible, 507 depth-occluded) and reaches 1.3191 px /
  0.0059973 m under the unchanged 6 px / 0.006 m gates. Collision-proxy and
  skin-artifact pixel counts are 0, and the preview was visually reviewed with
  no pink limb, duplicate hand, exposed collision proxy, or identity change.
- HDF5 contains exactly 1,801 synchronized depth frames at 480x640 and
  segmentation frames at 480x640x2; its render times match the indexed truth
  samples exactly. The accepted mux SHA-256 is
  `93885e1f6cdf9eaed34ab32d8795f489e67ee6541cc7795765a0330aa0f378c3`.
  FFprobe confirms 1,801 640x480 H.264 frames at 30 fps plus one 48 kHz mono AAC
  stream over 60.033333 s. The authoritative baseline SHA-256 is
  `3761381ee08be564f8caa5a01deff2bea8577b79ce7109098a2ff04f3bc43446`.
- The two contact-gated grasp-assist intervals remain fully disclosed: 2,061
  truth frames / 34.3333 s total, with 0 unflagged frames. Collision-checked
  locomotion assist is flagged for 181 truth frames. This is therefore an
  assist-heavy qualified engineering episode, not an unassisted dexterity
  result. The 12-cell operating envelope remains 9 passes and three exact
  red-ball/base-placement retrieve failures.

Actual artifacts are under ignored
`runs/embodied_simulation/phase_6/final` (about 516 MiB). The accepted episode
is `accepted_episode.mp4`; `baseline_rgb.mp4` is the authoritative visual
stream; `episode_trace.npz`, `render_streams.h5`, `speech.wav`, transcript and
alignment, QA receipts, truth schema, public-asset/dependency provenance, and
replay instructions form the synchronized package. `qa_preview.png` is the
16-sample contact/behavior sheet. `final_bundle_manifest.json` has SHA-256
`352285da5d1c81229abaa864e604416e8b5e0a631eb36d8eaa56eaefa5044be0`.
All media, traces, XML replay components, and generated records remain ignored.

Repository status: implementation/config/test commit `f47ec5e` (`Assemble
accepted embodied episode bundle`) is pushed to
`origin/embodied-simulation`; this compact checkpoint and aggregate are
committed and pushed separately. No complete run, media, model, dependency
environment, cache, log, or binary artifact is tracked.

Deviations from the frozen Phase 6 contract: none. Before that contract was
frozen, the selected source cell was requalified with the already recorded
Phase 5 contact-surface/depth QA repair so the final bundle did not preserve an
older midpoint-only receipt. Its physics trace, baseline, mux, HDF5, controller,
camera, collisions, seeds, event timing, and all thresholds remained unchanged.
The earlier explicit architecture deviation remains: native MIMo is the
authoritative visible hand/forearm, MPFB is diagnostic only, and no neural
appearance passed Phase 3 invariants.

Qualification boundaries remain explicit. Engineering qualification passes.
The ChildLens viewer-view path remains
`STOP_HARDWARE_PRIVACY_NO_LOCAL_CUDA`; restricted media was not inspected,
decoded, moved, or transferred. Measurement validity is not established and no
ChildLens scoring occurred. All three provisional ChildLens ages-3-to-5 visual
distribution intervals remain unmet, so this is not naturalistic or infant
calibration and not human validation. Appearance validity is deterministic
baseline only. No learner comparison or causal evidence was produced.

Smallest next step: preserve the ignored bundle and compact manifest as the
qualified simulator deliverable; any learner comparison, BabyView confirmation,
or causal study requires a separate governed protocol and authorization.

## Unity–MuJoCo feasibility preflight — 2026-08-02

Gate decision: **PROMISING-BUT-REPAIR**.

This is a new, bounded architectural feasibility check for the requested
Unity-rendered, MuJoCo-authoritative child-view replacement.  It deliberately
does not reuse or qualify the prior MIMo appearance shell, its camera, or its
contact-gated grasp assist.  The retained question is narrower: can this host
bootstrap the required official components and render through the designated
native truth-control fallback without touching restricted media?

Findings:

- Local discovery found no Unity Editor, Unity Hub, Unity activation file,
  Filament installation, or Python MuJoCo package.  These are missing
  dependencies, not hardware failures.  Unity Personal is conditionally
  feasible only if the account holder verifies its current eligibility; an
  agent must not create an account, accept terms, or assert the user's revenue
  or funding status.  Thus Unity activation is a user-action condition, not a
  failed runtime test.
- The official MuJoCo Unity package was checked at tag `3.3.7`, commit
  `f1d45bd5422c74beddfb0d1deb590a02583d21de`, Apache-2.0.  Its `MjScene`
  performs the MuJoCo step from Unity `FixedUpdate`, then synchronizes Unity
  state.  A future Unity run must lock Unity's fixed delta to the MJCF timestep,
  have exactly one `MjScene.StepScene` per tick, disable PhysX authority for
  target objects, and sample every stream from that tick.
- Official source builds succeeded in ignored `.external/native_fallback/`:
  MuJoCo `3.3.7` at the above commit and Filament `v1.71.5` at
  `b0f2090cccf3c11f524ed8053746aba3de6be199`, both Apache-2.0.  Filament's
  off-screen Metal backend selected the Apple M5 and produced a valid 320×240
  RGB image.  The compact source smoke test is
  `native_filament_smoke.cpp`; its generated image and two-second playable
  clip are ignored run evidence, not episode evidence.
- The initial native render exposed two ordinary integration defects: the
  material was compiled for OpenGL rather than Metal, and PPM export retained
  the alpha byte.  Recompiling with `matc -a metal` and writing RGB triples
  repaired both.  No scene, camera, object, or dynamics result is inferred
  from this smoke test.
- Apple Silicon/Metal is therefore supported by the native fallback build;
  neither an unsupported platform nor an actual renderer failure has been
  found.  No restricted ChildLens media was accessed, decoded, moved, uploaded,
  or used for thresholds or tuning.

Required episode evidence is **not yet produced**: no furnished room,
child-proportioned articulated rig, physical unassisted grasp, synchronized
truth bundle, depth/object-ID stream, contact timing, mesh/collision
registration, penetration report, identity report, or replay receipt exists
for this preflight.  The smoke render must not be represented as a child-view
episode or as a physical qualification.

The one warranted repair is to implement the existing 15–20 second contract
once, in the native MuJoCo + Filament truth-control fallback: the new rig must
derive camera pose exclusively from root/torso/neck/head plus a fixed mount,
drive reach and fingers via deterministic scene-conditioned primitives and IK,
and prove a contact/friction-supported grasp without parenting, pose writes,
welds, equality attachments, snapping, teleporting, or assist forces.  Only
after that fallback trace and its synchronized diagnostics pass should a
licensed/activated Unity run be attempted as the appearance comparison.
