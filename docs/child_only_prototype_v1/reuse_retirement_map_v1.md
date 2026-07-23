# Reuse and retirement map v1

## Reuse as clean-room concepts

| Concept | Source precedent | Child-only disposition |
| --- | --- | --- |
| RGB scene pixels without captions/contact markers | `babyworld_lite/grounding/pipeline.py` raw renderer | Reimplemented in the construction fixture; legacy numeric settings are not copied. |
| Physically separate model-visible and oracle records | Grounding pipeline split files | Strengthened to three roots: model-visible, object instrumentation, and hidden oracle; closed keys and exact file inventory. |
| Whole-episode, same-split derangement | Grounding control generator | Reimplemented as a manifest gate requiring no self-donor, exact permutation, matched shape/missingness, and whole-bundle covariance preservation. |
| Time-shift and null controls | Grounding/weak-alignment control ideas | Reimplemented as nonzero within-episode whole-bundle shifts and shape-matched null streams under one shared architecture. |
| Test-time side withholding | Motor-free evaluation tests | Strengthened to a separately exported vision/text-only module plus exact typed input rejection. |
| Canonical hashes and deterministic manifests | Later synthetic integrity utilities | Reimplemented narrowly with canonical JSON, content SHA-256, root confinement, no symlinks, and missing/extra-file rejection. |
| EgoBabyVLM multimodal extractor surface | Official local checkout at commit `224621c…` | Reuse only the vision/text embedding and similarity interface concept. Primary scaffold is temporal and scratch-initialized. |
| Whole corpus-seed × model-seed bundles | Prior parallel work | Reimplemented as a small device-aware planner/merge contract; all five conditions stay on one worker/device. |

## Retire from the main line

- Feature-level corrective learners and detector/corrector representations are not imported, resumed, rehearsed, or treated as the primary learner.
- The paused `corrective_alignment_v4` line remains untouched.
- Adult-data, sensor-alignment, development-launch, weak-alignment outcome, and grounding-pilot loaders are not imported into `babyworld_lite.child_only_v1`.
- The legacy `babyworld_lite.sim.Episode` object is not reused because it co-locates visible streams with hidden goals, event labels, causal graphs, and counterfactuals; its renderer also overlays label-bearing information.
- The legacy CNN/GRU grounding model is retired as the primary scientific learner. Only the two-tower contrastive idea remains.
- Existing permissive Machine-DevBench adapters remain secondary integration precedents; the new primary evaluator rejects unknown mapping keys and has no side state.
- Operational one-shot authorization, byte-level checkout preservation, replay prevention, package-security hardening, and elaborate subprocess machinery are out of scope for this bounded construction stage.

No retired module is imported by the new namespace; an AST test enforces this boundary.
