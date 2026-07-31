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
