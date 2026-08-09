# ChildLens v1.1 future parallel execution plan

Version: `childlens-future-parallel-plan-v1.1.0`  
Date: 2026-07-21  
Status: launch design only; no learner or causal execution authorized

## Governing rule

Parallelism may reduce wall time, but it may not change the empirical corpus, frozen sample, endpoints, thresholds, pairing, compute budget, arm definitions, or decision rule. ChildLens remains the sole empirical child corpus. Any IMU, contact, proprioception, or motor stream remains an explicitly simulator-defined counterfactual modality and is never described as ChildLens-measured.

One coordinator owns the permission receipt, immutable/restricted release binding, frozen manifest digest, selection digest, quarantine index, instrument pins, human-validation receipt, and final merge. Workers receive only the minimum shard or aggregate needed for their bounded task. No worker may independently substitute an episode, expand the sample, change a threshold, or inspect acquisition effects.

## Phase 0: close the current blockers

These tasks may run independently, but the coordinator must merge their receipts before media admission:

1. **Release/manifest closure:** bind the selective pilot either to an accessible immutable revision or to the deterministic restricted manifest plus observation receipt. Record that live-to-DOI byte equivalence is not proven unless a checksum-complete comparison succeeds.
2. **Resource re-admission:** measure fresh free bytes, exact selected raw bytes, namespace allocation, derivative/scratch bounds, owner-only modes, ACL, indexing, Time Machine, and cache disposition. Abort outside the 20-GiB raw, 73-GiB namespace-peak, and 50-GiB post-peak-free limits.
3. **Instrument closure:** after genuine language establishment, install only the pinned locally licensed ASR/aligner route inside quarantine, disable egress, and sentinel-test decode and logs. Measurement outputs remain restricted and are not learner ancestry.
4. **Human-validation preparation:** finalize the blinded auditory/referential packet, scoring implementation, annotator instructions, and reliability calculation before labels are opened.

No content worker starts until the coordinator verifies all four receipts against the frozen pilot digest.

## Phase 1: selective local feasibility pilot

Acquisition is serial by shard even if safe metadata and QA work runs in parallel. Only one raw object may be in flight. Each object is admitted by expected immutable identity/size, downloaded into an owner-only staging location, verified by local SHA-256, and atomically promoted. Invalid HTML or any other unexpected content type fails closed and is unlinked without becoming a manifest entry.

After verified media exists, the coordinator may schedule these bounded tracks:

- **CPU decode/timing:** begin with two workers, one numeric thread per worker. A fixed content-blind resource benchmark may raise this to at most four workers. Reduce to two on memory pressure, page-out growth, I/O saturation, or nondeterminism.
- **Local measurement instrument:** at most one MPS process globally. Do not overlap it with CPU preprocessing when the benchmark shows unified-memory or I/O contention. No second MPS process and no learner training are allowed in the feasibility task.
- **Human auditory review:** qualified language-matched humans establish source language, correct scientific transcript/timing, and label speaker role. ASR is never gold.
- **Human referential review:** blinded raters label visible object/action candidates, null cases, ambiguity, and lag using the frozen ontology. Automated vision outputs may propose queues only if the annotation-instrument amendment and human disposition rules are satisfied; their features never enter learner inputs.
- **Reliability/evaluation QA:** a separate worker computes frozen error/agreement metrics and cell-suppressed aggregates only after the blinded labels are locked.

Every worker writes restricted outputs to quarantine. The coordinator exports only allowlisted, cell-suppressed receipts. Content-level failures may stop the pilot; they may not trigger outcome-driven resampling.

## Phase 2: later scientific protocol freeze

If and only if all essential feasibility gates pass, a separate task may freeze the compact temporal CLIP+ study. The proposed learner remains scratch/random: a scratch vision tower, corpus-local scratch tokenizer/text tower, InfoNCE, matched MLM, and DINO-style self-supervision. Fixed pretrained ASR/VTC/vision systems are measurement instruments only: no instrument embedding, feature, vocabulary, tokenizer, score, checkpoint, or weight may enter learner ancestry. LLaVA remains deferred unless a later audit justifies it.

The frozen learner task must specify immutable data bundles, all arm definitions, exact seed list, common initializations, example order, augmentations, optimizer schedule, step/compute budget, evaluation moments, failure/retry rules, and side-cue removal at evaluation before any outcome is visible.

## Phase 3: GPU/Slurm paired-bundle execution

The scheduling unit is one **complete paired seed bundle**, not one arm. A bundle contains every matched arm required for that seed, shares the frozen ChildLens episode order and compute budget, and completes on the same GPU type and software/container digest. This preserves within-seed causal pairing while allowing different complete seed bundles to run concurrently.

Operational rules:

- Assign each complete seed bundle to an identical GPU model class; record GPU UUID/type, driver, CUDA/cuDNN, container, code, data-manifest, and protocol digests.
- Keep initialization derivation, example order, augmentation schedule, optimizer steps, precision, effective batch size, and evaluation timing matched across arms.
- Do not scatter matched arms of one seed across heterogeneous GPU types or independently tune/retry an arm after outcome inspection.
- A failed bundle is retried as a complete bundle under a frozen technical-failure rule. Partial arm results remain sealed and are not used to select a retry.
- Checkpoints and logs are restricted, noncommercial, nonshared ChildLens derivatives and inherit the approved retention deadline.
- Workers expose only health/capacity signals while runs are active. Outcome tables are unsealed only after all planned bundles and exclusions are fixed.
- Evaluation uses no side cues and preserves held-out participant/session grouping. Strong-alignment ceiling and weak/matched controls use the same immutable evaluation set.

This structure permits GPU-level parallelism without breaking the paired causal comparison. It does not authorize the future outcome run.

## Coordinator merge and stop rules

The coordinator performs a deterministic merge over bundle receipts and checks that all input/protocol/environment digests match. It stops rather than repairs opportunistically if a permission expires, a restricted digest changes, a participant/session grouping violation appears, an instrument leaks into learner ancestry, a worker exceeds resource bounds, a restricted payload enters Git/tool output/external service, or an arm/bundle deviates from its frozen compute contract.

The current exact next task is narrower than GPU execution: obtain hash-verified selected ChildLens media under the existing permission receipt, then complete genuine language-matched human auditory and referential validation. Until those gates close, no learner or causal arm should be launched.
