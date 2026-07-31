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
