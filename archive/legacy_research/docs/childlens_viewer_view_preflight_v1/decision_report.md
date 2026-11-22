# ChildLens Viewer–View Calibration Preflight v1

## Decision

**STOP — hardware/privacy gate.** The official ReViV plus ViPE measurement
route cannot be executed on the available local machine. The host is an Apple
M5 MacBook Pro with 32 GB memory and Metal graphics; it has neither an NVIDIA
GPU nor `nvidia-smi`, `nvcc`, or a CUDA runtime. ReViV's released environment
is pinned to PyTorch 2.6.0 CUDA 12.4 and a CUDA `torch-scatter` wheel. ViPE is
GPU-oriented and its current release uses CUDA fused kernels and cuVSLAM.
Porting either instrument to Metal would no longer be validation of the
predeclared official measurement route. Sending restricted ChildLens video to
remote compute is forbidden. Therefore simulator instrument validation,
ChildLens canary scoring, expansion, and controller repair were not authorized.

This is a hardware/privacy terminal decision, not a measurement-validity result,
not evidence about ChildLens, and not a causal null.

## What was established

- The official sources were inspected at ReViV commit
  `de23a67009685e3878e4bad49d33f023d4b7a085` and ViPE commit
  `95a8816947602ddc26fcb7a80bea4f9313059578`. ReViV code is Apache-2.0;
  its released weights are non-commercial research only. ViPE code is
  Apache-2.0 except optional separately licensed components.
- ReViV uses local two-second windows. The metric path samples 32 frames at
  16 fps and 512×512; its outputs remain model-derived estimates.
- The repository's frozen extension receipt identifies the authoritative
  existing sample as 30 participant-distinct ChildLens recordings. The prior
  categorical Qwen3-VL/Gemma 4 route remains stopped at its frozen abstention
  and null-envelope gates and was not reinterpreted.
- Read-only external-volume inventory found the encrypted restricted ChildLens
  sparse bundle plus existing simulator episode assets and known trajectory
  side streams. The sparse bundle was not mounted, and no recording was opened,
  decoded, copied, moved, renamed, or uploaded.
- The prospective measurement quantities, uncertainty procedure, abstention
  rules, simulator recovery checks, overlap/preprocessing checks, ReViV–ViPE
  agreement checks, canary/expansion rule, and bounded controller rule are
  frozen in `configs/childlens_viewer_view_preflight_v1.json`.

## Meaning for the Michael Frank bridge

The bridge remains scientifically well specified but untested at this step.
No new natural viewer/view distribution was measured, so no distribution can
honestly be reproduced in the simulator and no camera/head controller change
is licensed. Existing simulator-known camera, IMU, proprioception, contact,
touch, action, and object-state streams remain simulator truth; ReViV and ViPE
would only be measurement instruments.

## Smallest honest options

1. Attach or provision a governance-approved local Linux workstation with an
   NVIDIA GPU and CUDA 12.4 or a compatible official CUDA path, sufficient
   storage for both released checkpoint sets, and keep the restricted volume
   physically local. Run the frozen five-recording canary only after simulator
   validation passes.
2. Move the encrypted restricted volume to an institution-managed, non-cloud
   on-premises CUDA workstation under the same ChildLens governance, with
   explicit approval that no raw/intermediate artifacts leave that boundary.
3. If neither local CUDA route is available, retain this terminal stop. A future
   protocol may evaluate a different estimator, but it must be prospectively
   initialized as a new instrument study rather than substituted into this one.

## Privacy, provenance, and limitations

ChildLens is the sole empirical child-data source and is described only as
naturalistic child-centered data from ages 3–5: a provisional developmental
calibration, not final infant calibration. BabyView and AEA measurements,
examples, vocabulary, checkpoints, and empirical ancestry were not mixed in.
No raw or restricted ChildLens content was inspected during this task. No
private data was sent to any cloud service or external API. Only aggregate
repository receipts and filesystem metadata were read.

The decision does not establish that ReViV or ViPE is stable or unstable on
ChildLens, does not validate their outputs against humans or ground truth, and
does not assess controller mismatch. It only establishes that the frozen,
privacy-compliant official-instrument route cannot run on the available
hardware.

## Reproduction

Run the media-blind host gate:

```bash
python3 scripts/childlens_viewer_view_preflight.py
```

Exit status 2 is the expected fail-closed result on this host. Focused tests:

```bash
python3 -m pytest -q tests/test_childlens_viewer_view_preflight.py
```
