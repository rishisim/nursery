# ChildLens feasibility v1: future parallel execution plan

Date: 2026-07-21  
Status: launch design for a later, separately authorized task  
Scientific outcome executed: no

## Purpose and authorization boundary

This plan reduces wall time without changing the causal unit or exposing an outcome during feasibility work. It is not a launcher and does not authorize acquisition, transcription, learner training, evaluation, or terms acceptance. Processing begins only after the exact ChildLens release and data-use permissions are recorded, the selective pilot is approved, and the relevant protocol phase is frozen.

The plan inherits the five-arm construction design:

1. strong-alignment ceiling;
2. weak vision-language baseline;
3. weak plus synchronized simulator-defined side cues;
4. weak plus matched whole-episode shuffled side cues; and
5. weak plus within-episode time-shifted side cues.

ChildLens supplies only permitted video/audio ecology and a real vision-language baseline. Any IMU, contact, proprioception, or motor streams are simulator-defined counterfactual modalities and must never be labeled or logged as ChildLens measurements.

## Phase graph

Each phase consumes an immutable manifest and emits checksummed receipts. A later phase cannot start from a partial upstream inventory.

```mermaid
flowchart LR
  A["Terms and exact release gate"] --> B["Metadata-only manifest and byte budget"]
  B --> C["Selective sharded acquisition"]
  C --> D["Quarantined measurement instruments"]
  D --> E["Human correction and reliability"]
  E --> F["Feasibility gates and protocol freeze"]
  F --> G["Immutable complete five-arm bundles"]
  G --> H["Fail-closed canonical merge"]
```

The current feasibility task may stop at E/F and must not enter G/H as an outcome run.

## Metadata, preprocessing, and annotation parallelism

The controller freezes the selected manifest, exact byte total, checksum algorithm, permitted derivative roots, tool receipts, annotation protocol version, and expected work-unit IDs before dispatch.

- The authenticated sanitized live UI currently shows 192 MP4s; displayed sizes for 191 sum to approximately 201.33 decimal GB and one MP4 lacks a displayed size. The share is unpinned and its relationship to the public paper's 354-file/108.58-hour full corpus is unresolved. Treat it as a possibly annotated subset, not a versioned release identity.
- Metadata and download are serialized by shard. Each shard is checksum-verified before use; no worker may broaden the selection from content.
- The exact preflight byte sum of selected files controls admission. Do not estimate bytes from the public-paper duration or file-count averages, and do not select the unknown-size MP4 until its size is resolved.
- CPU preprocessing begins with two spawn workers and may increase to four only after a fixed benchmark shows adequate memory and I/O headroom. Ten host cores do not authorize ten workers.
- Each work unit is a complete video or a predetermined, nonoverlapping window set. A worker writes only its private temporary directory and atomically renames after schema and checksum validation.
- Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, PyTorch intra-op threads, and PyTorch inter-op threads to one. The controller alone owns process parallelism.
- ASR/diarization and visual measurement instruments are quarantined from learner artifacts. On the local host, only one MPS process runs at a time; measurement instruments and learner training never overlap.
- Annotation assignments are frozen before annotators see content. Reliability units are selected from metadata strata, not model confidence or apparent success. Double-coded units remain independent until the adjudication stage.
- Transcripts, speaker labels, exact times, frames, tracks, and identifiers remain restricted. Worker logs contain opaque work-unit hashes and aggregate status only, never payload excerpts.

The merge rejects missing, duplicate, extra, overlapping, symlinked, failed, or nonfinal work units; manifest/tool/protocol digest mismatches; and checksum conflicts. A partial annotation aggregate is never promoted as complete.

## Atomic scientific scheduling unit

For a later separately authorized outcome, the indivisible shard is one complete
`(childlens_corpus_instance_id, corpus_seed, model_seed)` paired bundle. The controller derives every expected bundle ID from the frozen protocol digest and seed grid before training. It never creates or retires bundles after inspecting outcomes.

Within a bundle:

- all five conditions run serially, in a frozen order, on one assigned device;
- every arm starts from an independent clone of the same immutable initialization receipt;
- weak arms share the identical ChildLens RGB/text records, split, corpus-local scratch tokenizer, example order, padding/batch shapes, optimizer, update count, stopping rule, and compute budget;
- side modalities are simulator-defined, training-only, and absent from the evaluation interface;
- each arm writes to a private path and the bundle becomes final only when all five arm receipts, checkpoints, metrics, and hashes validate; and
- no arm is dispatched separately, migrated to a different GPU type, or reused to complete another bundle.

This coupling is the causal-control boundary. Parallelism is across complete bundles, not across experimental arms.

## Local Apple silicon plan

The audited host has 32 GiB unified memory, 10 CPUs, MPS available, and no CUDA.

- Exactly one learner training process globally.
- Conditions and bundles execute serially on MPS.
- At most two to four controller-owned CPU workers may prepare already-frozen inputs or evaluate immutable checkpoints, subject to a measured memory gate.
- Do not overlap CPU work that causes swap, I/O saturation, or MPS memory pressure. The controller records peak resident memory, swap, free disk, and elapsed time without inspecting scientific scores.
- Resume may launch only an expected, wholly missing or demonstrably invalid bundle under the same protocol digest. A bundle with any valid final arm plus missing arms is quarantined for forensic validation; do not improvise arm-level continuation unless the frozen resume protocol explicitly proves identical state.

The local host is suitable for preprocessing, feasibility instrumentation, engineering validation, and small frozen runs. It is not a low-latency base/high multi-bundle platform; the base planning envelope is about 10.4 serial days under provisional assumptions.

## CUDA and Slurm plan

Use only a homogeneous GPU pool for a protocol instance: identical GPU model and memory capacity, with recorded driver, CUDA, framework, precision, determinism, and container/environment digests. A GPU receives one complete bundle at a time.

For local multi-GPU CUDA:

- map one controller process to each explicitly assigned GPU;
- set `CUDA_VISIBLE_DEVICES` before framework initialization;
- keep all five arms of a bundle on that GPU; and
- cap host CPU/data-loader concurrency so GPU count does not multiply nested threads.

For Slurm:

- freeze a canonical, sorted bundle manifest before `sbatch`;
- map each array index deterministically to exactly one complete bundle ID;
- request one GPU per task and explicit CPUs/memory/scratch;
- require the cluster scheduler to constrain the job array to the frozen homogeneous GPU type;
- stage only that bundle's immutable inputs into private job-local scratch;
- write outputs to a private temporary destination, validate locally, then atomically publish the complete bundle; and
- retry only the same array index with the same plan, inputs, environment, GPU type, and budgets. Scheduler failure never authorizes a new seed or changed arm.

If `B` is the number of bundles, `G` is the number of identical GPUs, `U` is frozen updates per arm, and `s` is benchmarked seconds per update, estimated training wall time is
`ceil(B/G) * 5 * U * s / 3600`, plus frozen evaluation/checkpoint overhead. This formula is capacity planning only and is never updated from effect direction.

## Canonical merge and failure policy

The sole canonical merger sorts by corpus instance, corpus seed, model seed, and fixed condition order. It requires the exact expected inventory and rejects:

- missing, duplicate, extra, condition-incomplete, nonfinal, or symlinked bundles;
- foreign corpus instance, release, data, split, tokenizer, architecture, initialization, optimizer, order, update-budget, stopping, evaluation, environment, or protocol digests;
- mixed GPU types or unrecorded device/environment receipts;
- payload checksum conflicts or learner artifacts with measurement-instrument ancestry;
- any AEA or BabyView empirical ancestry; and
- any result produced after outcome-driven rescheduling, added seeds, early termination, or protocol mutation.

No partial cross-bundle aggregate is emitted. Failures are reported as engineering failures with opaque IDs; scientific endpoints are not inspected to decide retries. Once the expected inventory validates, evaluation and uncertainty analysis run exactly once under the frozen procedure.

## Launch checklist

All items must be true before future dispatch:

- exact accessible ChildLens release and checksummed manifest are versioned;
- terms explicitly permit the proposed local processing, restricted derivatives, retention, and aggregate exports;
- selected shard exact preflight bytes fit the scenario allocation and the 20 GiB raw cap, 73 GiB namespace cap, and 50 GiB free-space floor; the unknown-size MP4 is absent unless its exact size has been resolved;
- language, ASR/diarization instrument, human correction, and reliability protocols are frozen;
- corpus-local scratch tokenizer and scratch learner have provenance receipts with no external learner ancestor;
- measurement instruments are contractually quarantined and pass no embeddings/features to the learner;
- held-out participant/session split, vocabulary/exposure gates, strong ceiling, weak baseline, five arms, seeds, updates, stopping rule, endpoints, and uncertainty procedure are frozen before outcomes;
- all complete bundle IDs and device assignments are precomputed;
- MPS/CUDA/Slurm throughput, memory, checkpoint size, and storage were measured without computing an arm contrast;
- canonical merge negative tests reject every forbidden inventory/provenance mutation; and
- authorized operators can stop for privacy, terms, capacity, or correctness failures without viewing or optimizing a scientific effect.

No commit, push, upload, publication, license acceptance, Keeper-share change, or external contact is part of this plan.
