# Smallest five-arm paired execution design

Status: `DESIGN_ONLY_NOT_AUTHORIZED`  
Date: 2026-07-22  
Scope: synthetic code, configuration, and aggregate engineering receipts only

## Decision

The smallest design that is both five-arm complete and genuinely parallel is a 2×2 seed grid: two corpus seeds crossed with two model seeds. It creates four atomic paired bundles and twenty arm trainings. Four identical GPUs can execute the four bundles concurrently; two identical GPUs execute two waves. A single Apple-silicon host executes the same four bundles serially with exactly one MPS trainer.

This is the minimum *procedural* experiment that exercises both corpus-seed and model-seed variation. Four bundles are not enough for a stable hierarchical population claim. If the next scientific protocol requires inferential claims across children, corpus draws, or initializations, its larger seed grid must be frozen before any result is inspected; the grid cannot be expanded after seeing an effect.

No outcome was run for this design. No AEA, ChildLens, BabyView, restricted manifest, or quarantine payload was read. The only result evidence used was aggregate synthetic resource telemetry and the non-outcome construction microbenchmark.

## Frozen candidate grid

The proposed minimum seed grid is:

- Corpus seeds: `510101`, `510113`
- Model seeds: `510211`, `510223`
- Corpus instance label: `synthetic-minimum-paired-v1`
- Bundle count: 4
- Arms per bundle: 5
- Total arm trainings: 20

The numbers are unused in the inspected code/config/report namespace. They are candidate seeds, not authorization. The scientific freeze must add them to a purpose-specific registry and fail closed on any collision with prior, smoke, development, or confirmation seeds.

## Atomic paired bundle

The atomic scheduling and retry unit is exactly one `(corpus_instance_id, corpus_seed, model_seed)` bundle. It contains all five conditions in this fixed order:

1. `strong_alignment_ceiling`
2. `weak_vl_baseline`
3. `weak_synchronized_side`
4. `weak_episode_shuffled_side`
5. `weak_time_shifted_side`

Conditions are serial on one assigned device. A scheduler must never put different arms of one bundle on different devices, run them concurrently, or retry a single arm in isolation. If an arm fails, the worker discards its private temporary bundle and reruns the complete bundle from the immutable initialization receipt under the same plan digest.

Each bundle creates one initialization state from its model seed, records its digest, and clones that exact state independently for each arm. An arm is never initialized from a preceding arm's trained state. RNG namespaces for initialization, example order, augmentation, dropout, and evaluation must be deterministically derived and recorded; changes caused by the experimental manipulation may affect inputs only through the frozen condition transform.

The four weak arms must have byte- or digest-identical receipts for RGB/text experience, examples and splits, tokenizer, architecture, initialization, optimizer, example order, update count, batch/padding shapes, stopping rule, and compute budget. The strong-alignment ceiling should also share initialization, architecture, optimizer, update count, batch shapes, stopping, and compute; it differs only in the predeclared strong language-event alignment. The current causal protocol explicitly lists the equality invariants only for the weak arms, so the strong-arm extension is a required pre-launch protocol amendment.

All five primary evaluations accept vision and text only. Side channels, side-tower state, and side-derived features are absent from the exported evaluation model.

## Genuine GPU parallel schedule

Use one complete bundle per GPU and only identical GPU models:

| GPUs | Wave 1 | Wave 2 | Training concurrency |
|---:|---|---|---:|
| 4 identical GPUs | bundles 0–3 | none | 4 |
| 2 identical GPUs | bundles 0–1 | bundles 2–3 | 2 |
| 1 GPU or MPS | bundle 0, then 1, then 2, then 3 | n/a | 1 |

Every job records the physical GPU model, driver/runtime, precision mode, deterministic settings, library lock, and allocated CPU/memory. Mixing GPU types, CUDA versions, precision policies, or backends inside one run is a hard stop. More than four GPUs does not justify splitting arms; extra devices remain unused for training or may perform non-scientific integrity checks after all payloads are sealed.

Workers do not create child pools. The controller alone owns parallelism. Each worker sets:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
TORCH_NUM_THREADS=1
TORCH_NUM_INTEROP_THREADS=1
```

CPU-only deterministic preparation and evaluation may use two workers by default, increasing to four only after the frozen-shape canary passes memory and equivalence checks. CPU tasks may be parallelized across bundles, never across condition-specific code paths that could alter pairing.

## Safe single-Mac fallback

The M5/32-GiB fallback uses the same four-bundle plan with `backend=mps`, `max_trainers=1`, and `cpu_workers=2`. The controller prepares immutable inputs, then executes all five arms of bundle 0 serially before moving to bundle 1. At most one MPS-heavy process exists globally. Evaluation can use two CPU workers only after the trainer releases MPS memory; it must not overlap with another training process.

The single-Mac schedule is scientifically paired but not parallel in training. Its value is identical plan semantics and safe resumability. Thermal throttling and shared unified-memory pressure make its wall time less predictable; bundle-level telemetry must be recorded but excluded from scientific equality digests.

## Existing executable paths and commands

The plan/validator implementation is:

- `babyworld_lite/child_only_v1/parallel.py`
- `scripts/plan_child_only_v1_parallel.py`
- paired-contract validation in `babyworld_lite/child_only_v1/protocol.py`

The current causal contract is `docs/child_only_prototype_v1/causal_protocol_v1.json`, with observed construction digest `52ded736d54af136b5767ede1dc8a8125f71a2bbfdd0184e247248170fc7da13`.

Plan-only construction dry runs:

```bash
cd /Users/rishisim/Documents/research/nursery
PYTHONPATH=. .venv/bin/python scripts/plan_child_only_v1_parallel.py \
  --protocol-digest 52ded736d54af136b5767ede1dc8a8125f71a2bbfdd0184e247248170fc7da13 \
  --corpus-instance-id synthetic-minimum-paired-v1 \
  --corpus-seeds 510101 510113 \
  --model-seeds 510211 510223 \
  --backend mps --max-trainers 1 --cpu-workers 2 \
  --out <new-empty-plan-path>
```

```bash
cd /Users/rishisim/Documents/research/nursery
PYTHONPATH=. .venv/bin/python scripts/plan_child_only_v1_parallel.py \
  --protocol-digest 52ded736d54af136b5767ede1dc8a8125f71a2bbfdd0184e247248170fc7da13 \
  --corpus-instance-id synthetic-minimum-paired-v1 \
  --corpus-seeds 510101 510113 \
  --model-seeds 510211 510223 \
  --backend slurm_cuda --max-trainers 4 --cpu-workers 2 \
  --out <new-empty-plan-path>
```

The in-memory dry run validated four bundles and twenty arms. Its candidate plan digests were:

- MPS: `6e915ea59671b0357fc1d9af572f61ee5491600328d86aa15d78abbaefa19cf0`
- four-worker Slurm CUDA: `1d0b3370b52972f3d6c7b4308496214ae54b99151917e87aedea56def36ff705`

These are construction-plan receipts only. The current CLI hard-codes the non-scientific construction profile and `outcome_task_authorized=false`; it cannot authorize or launch a scientific run. After an outcome protocol is frozen, the plan must be regenerated from its new digest, so neither candidate plan digest is reusable.

Planner/merge tests:

```bash
cd /Users/rishisim/Documents/research/nursery
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_child_only_prototype_v1.py::test_parallel_plan_keeps_conditions_coupled_and_mps_serial \
  tests/test_child_only_prototype_v1.py::test_parallel_merge_is_canonical_and_fails_on_incomplete_inventory
```

There is currently no scientific five-arm training worker or Slurm submission entrypoint. Accordingly, no honest `sbatch` or local outcome-launch command exists yet. Adding such an entrypoint is new infrastructure and was outside this workstream. A future launcher must consume a validated plan and map array index to the existing `slurm_array_index_to_bundle_id`; it must not accept free-form condition, seed, or budget overrides.

## Expected outputs

Before launch:

- immutable outcome protocol and digest;
- canonical plan and plan digest;
- complete expected bundle-ID inventory;
- seed-registry collision receipt;
- data, tokenizer, architecture, initialization, optimizer, order, update, batch-shape, stopping, compute, environment, and device receipts;
- one frozen-shape synthetic canary receipt with forward/backward/optimizer timing, peak device/system memory, and disk per complete bundle.

Each worker writes only a private temporary directory. On success it atomically renames that directory to the bundle ID. The final directory contains exactly:

- `result_manifest.json` with complete status for all five conditions, plan/protocol identity, shared digest receipts, payload path, and payload SHA-256;
- one direct-child sealed payload containing the frozen five-arm outputs and evaluation/manipulation-check records.

Runtime telemetry must live outside the canonical shard root or inside the frozen payload schema; the existing merger rejects extra files. It must not change scientific equivalence hashes.

The canonical merge emits nothing unless all four expected bundle directories are present, complete, nonsymlinked, hash-valid, and bound to the same plan/protocol. It sorts by corpus instance, corpus seed, and model seed. There is no partial aggregate, best-seed report, or condition-by-condition early result.

## Stop conditions

### Before any training

Stop if any of the following holds:

- the protocol remains `CONSTRUCTION_ONLY_NOT_FROZEN_FOR_OUTCOME` or lacks explicit outcome authorization;
- the complete seed registry, strong-arm equality amendment, model/update budget, primary contrasts, uncertainty method, or stopping rule is unfrozen;
- selected inputs, tokenizer, initialization, or simulator calibration lack immutable ancestry receipts;
- a fixed-shape full training canary has not measured backward/optimizer time, peak memory, and disk;
- allocated GPUs differ in model, runtime, precision policy, or deterministic capability;
- projected peak use breaches the frozen memory/disk reserve;
- any worker can accept an arm-specific compute, seed, initialization, or data override.

### During a bundle

Fail the complete bundle, without inspecting or merging its scientific metrics, on OOM, non-finite loss, missing/extra examples, digest mismatch, altered batch/order/update count, condition-incomplete output, evaluation-side leakage, device mismatch, nonzero exit, payload hash failure, or failure of a frozen manipulation check. Remove the private temporary shard and retry only the whole bundle under the identical plan if the frozen retry policy permits it.

### Before aggregation

Stop without an aggregate on missing, duplicate, extra, symlinked, foreign-plan, foreign-protocol, nonfinal, condition-incomplete, or checksum-conflicting shards. Never add seeds, change arms, alter updates, or relax a gate after inspecting outcomes.

## Resource and wall-time evidence

The construction scaffold has 348,933 parameters, approximately 1.33 MiB of FP32 parameters and 5.32 MiB of rough weights/gradients/Adam arithmetic state. On the M5, its fixed two-example, four-frame, 64-pixel *forward-only* path measured 0.00207–0.00237 seconds per batch. It excludes backward, optimizer, data loading, real clip dimensions, final model width, and evaluation, so it is not a scientific runtime estimate.

For four bundles with `U` training steps per arm, that measurement implies only a forward-path lower-bound coefficient:

```text
single MPS: 20 arms × U × (0.00207–0.00237 s) = U × (0.0414–0.0474 s)
four GPUs:  5 arms × U × T_forward_on_frozen_GPU, plus one bundle evaluation
two GPUs:  10 arms × U × T_forward_on_frozen_GPU, plus two bundle-evaluation waves
```

Backward and optimizer work commonly dominate this lower bound; no hour estimate is defensible until the exact batch, clip/window shape, model width, update count, evaluation inventory, and device are frozen and benchmarked end to end.

Aggregate synthetic CPU scheduler receipts provide a limited parallelism prior: four workers completed eight synthetic units in 5.26 seconds versus 15.91 seconds serial (3.03× speedup), with maximum worker peak RSS about 111 MiB and a conservative projected concurrent/parent peak about 11.4 GiB. An earlier receipt observed 2.87× speedup. These support a two-worker default and four-worker ceiling for CPU preprocessing/evaluation, but they do not predict neural training time or memory.

The wall-time equations to fill from the required canary are:

```text
T_bundle = 5 × T_arm_train + T_bundle_eval + T_bundle_seal
T_4GPU   = T_bundle + T_merge
T_2GPU   = 2 × T_bundle + T_merge
T_MPS    = 4 × T_bundle + T_merge
```

Use the slowest measured arm/device time, not the mean, and apply at least a 20% wall contingency after the complete-bundle canary. Memory is the peak of one arm per device, not five arms simultaneously. Disk admission should reserve at least twice the measured sealed-bundle size times four, plus immutable input and merge space. The small scaffold's arithmetic state is not a substitute for this measured admission check.

## Limitations and exact next task

This design validates scheduling semantics, not scientific readiness. The current protocol is construction-only, the strong-ceiling equality contract needs one explicit amendment, the final seed count and inferential claim are unresolved, and no training-worker CLI exists. Most importantly, the preserved microbenchmark is forward-only and fixture-shaped; it cannot justify a firm wall-hour or storage estimate.

The exact next task is a separately authorized **outcome-protocol freeze and full-training synthetic canary**: freeze the model/batch/window/update/evaluation shapes and seed claim, add the strong-arm equality invariant, implement a no-free-form-overrides bundle worker against the existing plan/merge API, and measure one complete five-arm bundle on the target GPU class without using empirical child-corpus payloads or reporting an acquisition effect. Only then can the plan be regenerated and a real launch command issued.
