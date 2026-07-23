# Future parallel execution design v1

This is a launch design for a later separately authorized outcome task. Construction does not launch training.

The atomic shard is one complete `(corpus_instance_id, corpus_seed, model_seed)` paired bundle containing all five conditions in fixed order. Conditions are never distributed across workers or devices. Within a bundle they execute serially on the assigned device from independent clones of one immutable initialization receipt.

The controller canonically freezes all expected bundle IDs from the protocol digest and seed grid before launch. Each worker writes only a private temporary directory, verifies all condition states and content hashes, then atomically renames it to the bundle ID. Workers never create child pools. Resume can execute only missing expected bundle IDs under the identical plan digest; it cannot inspect outcomes to change seeds, arms, or budgets.

## Device policy

- Local Apple silicon/MPS: exactly one training process globally. CPU-only generation and evaluation may use a conservative two to four controller-owned spawn workers, subject to measured memory. MPS conditions stay serial.
- Local CUDA: at most one training process per explicitly assigned GPU; coupled conditions remain serial on that GPU.
- Cluster CUDA: Slurm/job-array indices map deterministically to whole bundle IDs. `CUDA_VISIBLE_DEVICES` and allocated CPU count are explicit; no shard may span an incomplete condition subset.
- Every worker sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, PyTorch intra-op threads, and PyTorch inter-op threads to one. Only the controller owns parallelism.

## Canonical merge

Merge sorts by corpus instance, corpus seed, model seed, then the fixed condition order. It rejects missing, duplicate, extra, symlinked, failed, nonfinal, or condition-incomplete shards; foreign plan/protocol digests; mismatched data/tokenizer/architecture/initialization/optimizer/order/update/compute receipts; and payload checksum conflicts. No partial aggregate is emitted.

The executable planner/validator is [parallel.py](../../babyworld_lite/child_only_v1/parallel.py), with a plan-only CLI at [plan_child_only_v1_parallel.py](../../scripts/plan_child_only_v1_parallel.py). The current CLI cannot authorize or launch a scientific outcome.
