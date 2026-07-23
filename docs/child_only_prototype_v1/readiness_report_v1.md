# Child-only prototype construction readiness report v1

Date: 2026-07-19  
Terminal state: `CHILD_ONLY_PROTOTYPE_CONSTRUCTION_READY`  
Contract: `child-only-scientific-contract-v1.0.0`

## Outcome

The bounded child-only prototype is construction-ready. The new namespace is `babyworld_lite.child_only_v1`; it does not import historical adult-data, sensor-alignment, weak-alignment outcome, development-launch, grounding-pilot, or corrective-alignment modules. No scientific acquisition outcome was executed. No restricted child record was accessed. No AEA file, empirical value, aggregate, distribution, prior, language, feature, tokenizer, checkpoint, or result entered the new path. Existing evidence and the paused corrective-v4 line remain untouched.

This terminal state means the boundary, schema, learner interface, withholding, causal-arm, evaluation, deterministic construction, resource, parallel, reuse/retirement, and documentation gates pass. It does not mean the learner works scientifically, resembles a child corpus, or produces positive cue lift.

## Current access and scientific assumptions

- BabyView access: unavailable/pending in this task.
- ChildLens access: unavailable/pending in this task.
- Selected corpus: none.
- All numeric data and tensor shapes: `NONSCIENTIFIC_CONSTRUCTION_FIXTURE`, excluded from ecological and acquisition claims.
- Audio/TTS: deferred.
- Scientific learner/tokenizer: must be initialized from scratch after selecting one child corpus instance.
- External pretrained weights: diagnostic-only and prohibited from scientific ancestry.
- Apples-to-apples replication: separate corpus instance, separate tokenizer, separate model construction, identical frozen procedure; no pooling or transfer.

## Gate evidence

| Gate | Evidence | Status |
| --- | --- | --- |
| Versioned boundary and provenance | `scientific_data_boundary_contract_v1.md`, machine policy, positive/negative DAG tests | PASS |
| AEA exclusion and one-corpus isolation | Adult-source marker rejection, mixed BabyView/ChildLens ancestry rejection, fresh artifact-lineage tests | PASS |
| Complete episode schema | Actual RGB fixture frames, timed text, IMU, proprioception, contact/touch, motor, object trajectories | PASS |
| Hidden truth separation | Distinct model-visible, instrumentation, and hidden-oracle roots; exact closed keys and file inventories | PASS |
| Separate calibration definitions | Independent BabyView/ChildLens specs with definitions, units, uncertainty, missingness, and export constraints; no empirical values | PASS |
| Post-access adapter gate | Corpus/calibration adapters reject no-access, no-selection, wrong-corpus, and wrong-instance bindings before iteration | PASS |
| Temporal CLIP+ scaffold | Scratch BERT-like text tower, compact spatial/temporal ViT, raw-side encoder, candidate-event/null aligner | PASS |
| Test-time withholding | Exported module has vision/text/similarity state only; exact batch rejects side, detector, oracle, and corpus metadata | PASS |
| Causal arms and matched controls | Five canonical arms; weak common receipts; no-self same-split shuffle and nonzero time-shift validation | PASS |
| Corpus-grounded evaluation | Noun/object, verb/action, novel-composition, and cross-world/render-seed specs; Machine-DevBench secondary/coverage-gated | PASS |
| Determinism | Repeated fixture manifests have the same canonical digest; mutations and extra files fail | PASS |
| Future parallel execution | Whole paired bundles, one MPS trainer, CUDA/Slurm mapping, thread caps, canonical fail-closed merge | PASS |
| Reuse/retirement | Clean-room reuse map and AST import isolation test | PASS |

## Validation performed

- `.venv/bin/python -m pytest -q tests/test_child_only_prototype_v1.py` → **22 passed**.
- Python compilation passed for every new package module and both new CLIs.
- Both standalone CLIs load successfully from the repository without a `PYTHONPATH` workaround.
- `scripts/validate_child_only_prototype_v1.py --benchmark-device mps` passed all nine construction gates. The final strengthened pass, including explicit machine-policy validation, is the non-overwriting report `output/child_only_prototype_v1/construction_validation_mps_v3.json`; earlier construction passes remain preserved beside it.
- The local official EgoBabyVLM checkout remained clean at commit `224621caf0628270b6115845ac75a65b984234a3`.
- Historical AEA and corrective-v4 tests were deliberately not run; doing so was outside the child-only construction boundary.

The construction fixture has two arbitrary episodes solely to exercise serialization and negative cases. No test asserts or computes positive acquisition lift.

## Host and engineering estimate

The local host reports 32 GiB unified memory, 10 CPUs, MPS available, and CUDA unavailable. The default scratch scaffold has 348,933 parameters: about 1.33 MiB of FP32 parameter bytes and a rough 5.32 MiB weights/gradients/Adam-state arithmetic estimate, excluding activations, input tensors, allocator reserve, and framework overhead.

Across the preserved fixed random two-example MPS construction passes, forward time ranged from approximately 0.00207 to 0.00237 seconds, or approximately 844 to 966 fixture examples/second. This is an engineering interface measurement only: it excludes backward/optimizer/data loading, uses arbitrary 64-pixel frames and four-frame clips, and cannot estimate final scientific runtime or model quality. Exact batch size, clip duration, updates, bundle count, and wall-clock estimate remain unresolved until selected-corpus measurements and a frozen post-access model shape exist.

The future local policy caps concurrent MPS training at one. CPU-only generation/evaluation may use two to four controller-owned workers after memory measurement. CUDA/cluster execution uses deterministic whole-bundle job arrays. All nested numeric-library and PyTorch worker threads are capped at one.

## Decisions that genuinely require selected child data

1. Which single corpus and restricted release instance is authorized for the first study: BabyView or ChildLens.
2. Which specified measurements are actually available and permitted under that corpus’s terms, including usable timing, participant/episode clustering, visual measurement reliability, and any head-motion stream.
3. Corpus-specific language, alignment, visibility, ambiguity, duration, motion, and missingness estimates with uncertainty. Missing measures must be marked unavailable, never borrowed.
4. The fresh corpus-local tokenizer vocabulary/support and which noun, verb, and composition endpoints have adequate exposure. The Machine-DevBench coverage threshold must be frozen before interpreting its score.
5. Corpus-grounded world assets, concepts, utterance templates/constraints, and trial inventories that can be constructed without exporting identifying examples.
6. Exact clip/window shapes, model width within the compact family, batch size, update budget, corpus/model seeds, stopping rule, manipulation checks, inference thresholds, uncertainty procedure, and resource projection. These must freeze before any outcome is seen.
7. Whether the combined generated side bundle is sufficiently informative under its predeclared manipulation check. Cue-specific ablations remain unavailable unless that check triggers the separately frozen branch.

## Exact next-task brief — do not launch now

**Task title:** Child-only selected-corpus calibration freeze and acquisition outcome v1

**Preconditions:** The user has obtained restricted access, explicitly selected exactly one corpus/release instance, confirmed permitted local processing/export constraints, and authorized a separate scientific outcome task. No action is taken merely because access is pending.

**Scope:**

1. Bind exactly one corpus adapter and create a new corpus-instance provenance DAG. Reject all adult-data ancestry, the other child corpus, parent checkpoints, shared tokenizers/weights, and construction-fixture promotion.
2. Execute only the selected corpus’s frozen calibration measurement specification; export only permitted nonidentifying aggregate payloads with units, uncertainty, coverage, missingness, and limitations. Mark unavailable measurements without substitution.
3. Build a fresh corpus-local tokenizer and scratch temporal CLIP+ initialization. Create corpus-grounded simulator, split, and evaluation artifacts under new scientific IDs; validate hidden-oracle separation and vocabulary support.
4. Before any outcome-producing run, freeze the scientific protocol digest: selected calibration payload, all five arms, combined side construct, trial endpoints, primary contrast, manipulation/coverage gates, seeds, initialization receipts, batch/update/compute budgets, stopping rule, inference/uncertainty procedure, and canonical parallel plan. No later adaptation to effect direction.
5. Run complete corpus-seed × model-seed paired bundles with conditions coupled serially per assigned device. On this host use one MPS trainer; on CUDA/Slurm use whole-bundle job arrays. Merge only a complete, checksum-valid expected inventory.
6. Evaluate with the exported vision/text-only model on corpus-grounded noun/object, verb/action, novel-composition, and cross-world/render-seed tasks. Treat Machine-DevBench as secondary only if its predeclared corpus-specific coverage gate passes.
7. Report synchronized-minus-shuffled as the primary contrast with all controls, hierarchical uncertainty, manipulation checks, failures, resource use, and the fixed claim boundary. Null or contrary results are valid. Do not claim infant learning, corpus equivalence, or efficacy.
8. Preserve all artifacts under a new version; do not commit, push, contact anyone, accept licenses, pool corpora, or modify historical evidence unless separately authorized.

That task is intentionally not launched by this construction-ready decision.
