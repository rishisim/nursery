# Prior work retained by reference

This note records the small subset of prior Nursery findings that may inform
the synthetic-video program. It is not an active protocol, implementation, or
empirical result for this branch.

## EgoBabyVLM and Machine-DevBench

At commit `0c8a700`, the repository contained a macOS/MPS-compatible
integration with the public EgoBabyVLM Machine-DevBench evaluator. The
integration completed the public trials and closely reproduced a published
CLIP reference result.

The provisional Nursery learner was not scientifically interpretable on that
benchmark: its training vocabulary covered only 13 of 1,414 unique benchmark
target words. A future learner must pass a prospectively frozen vocabulary and
task-coverage gate before the benchmark is treated as an outcome.

Synthetic-trained models are not eligible for the official BabyView-only
challenge track. Machine-DevBench may still be useful as an external
evaluation protocol for a scientifically distinct synthetic-data study.

Relevant historical paths include:

- `docs/egobabyvlm_evaluation_feasibility.md`
- `scripts/run_machine_devbench.py`
- `scripts/train_nursery_checkpoint.py`
- `babyworld_lite/grounding/machine_devbench_adapter.py`

## ChildLens viewer-view preflight

At commit `7571c0e`, the frozen ReViV/ViPE preflight stopped at a
hardware/privacy gate. The available Apple Silicon host had no supported local
CUDA route, while restricted ChildLens media could not be sent to remote
compute.

No ChildLens recording was decoded or scored in that preflight. It produced no
ChildLens distributional measurement, generator-control result, or scientific
null. That result should not be reinterpreted.

Relevant historical paths include:

- `docs/childlens_viewer_view_preflight_v1/decision_report.md`
- `configs/childlens_viewer_view_preflight_v1.json`

## Recovery

Historical files can be inspected without restoring them to the active tree:

```bash
git show 7571c0e:path/to/file
```

If a historical component becomes necessary, recover only the smallest
relevant implementation and adapt it to the canonical protocol. Do not restore
the legacy experiment or output trees wholesale.
