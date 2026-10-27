# Current Nursery plan

## Question

Does bodily experience improve language acquisition over the same video and speech alone? The long-term hypothesis is a higher learning plateau. A gain at one data amount or a fixed-data training plateau does not establish that claim. Keep body-present understanding separate from picture-only transfer after learning with body inputs.

## Current checkpoint

- Data and matched speech/features: `outputs/renders/clean_856` and `outputs/runs/clean_body`; preserve these inputs and prior runs.
- Baseline: additive linear learner with the working 512-dimensional visual/word representation. The joint Transformer is historical, not the default next architecture.
- In matched corrected-exposure rerun 4.14 (seed 0), body-present clips-test was 52.28% video-only versus 47.54% full body plus load/skin (Δ −4.74 pp, paired-trial 95% CI [−8.60, −0.88]); s8 was 48.03% versus 46.00%. Picture-only transfer was 67.12% versus 66.22% rendered and 64.60% versus 64.02% Konkle. This fixed-data comparison found no body benefit and does not test a higher ceiling. Independent review passed. See the [4.14 record](archive/language_learning/experiments.md#414-matched-corrected-exposure-rerun).
- Speech remains weak, delayed and nonheld by design. Arms must receive the same speech; ordinary learning examples must not be selected using hidden referent identity. See the [design history](archive/language_learning/design_intent.md).

## Current step and next direction

1. **Trustworthy experiences and tests — closed.** Speech-alignment and body-present scoring repairs and the matched comparison are complete and independently verified. This closes that repair-and-comparison task, not the broader scientific question or a claim of benefit. See the [implementation details](docs/learner.md#speech-alignment-and-body-present-scoring) and [4.14 record](archive/language_learning/experiments.md#414-matched-corrected-exposure-rerun).
2. **Small diagnostic capability — diagnostic complete; frozen gates and independent review passed.** With scaffolded heavy/light words, video plus 10-d load scored 92.42% on two held-out categories versus 37.88% for video (Δ +54.55 pp); deranged load reduced accuracy 46.21 pp. Video was below chance, so the gap does not show video cannot infer weight. Picture-only scores were 40.91% after load training and 37.88% for independent video, both below chance. This does not test ordinary weak noun learning or a higher language ceiling. See the [4.15 record](archive/language_learning/experiments.md#step-2-heavy-light-capability-test).
3. **One transfer-focused sensory-pretraining experiment — ACTIVE, planned; not implemented or run.** Test whether aligned visual-sensory prediction before weak naming helps a visual representation that remains useful after the body branch is removed. This uses sensor streams already present in the frozen corpus; it is not a new interaction, mesh, word, or heavy/light test.

**Question and arms.** Does aligned sensory prediction improve later picture-only language learning beyond matched visual temporal pretraining and a deranged-sensor control? Run exactly three arms:

- **V:** visual temporal pretraining only.
- **A:** the same visual pretraining plus aligned sensor prediction.
- **S:** the same visual pretraining plus sensor prediction on deranged targets.

All arms train on all five frames of each selected five-frame window. The shared visual objective has weight 1 and is the mean existing `consistency_loss` over all four adjacent frame pairs. In A and S, add 0.5 × existing `predict_loss`: 22 contact, 22 contact-change, 2 aperture and 2 view targets. Use only the four within-window change adjacencies; do not add a sixth-frame target. S swaps complete five-frame sensor-target windows across different training-source clips within the same subject/mask stratum, using a fixed seed-0 derangement with full coverage of all 631 source clips.

**Frozen sampling and leakage controls.** Use the current clip seed-0 split (631 training clips). In sorted clip order, take the first eligible contiguous occurrence per clip and choose one valid five-frame window uniformly with seed 0; freeze the same windows across arms and seeds. Fit normalization only for the existing aperture target (a finger-flexion proxy), using selected training-window seconds. Keep contact/view BCE targets binary [0,1] and contact-change difference-MSE targets unstandardized, as defined by the existing loss. Clip metadata may define boundaries and splits only. Do not use mass, object ID, referent identity, words or speech labels to select pretraining examples or as inputs.

**Phase 1: pretraining.** Use the existing learner/model and losses, with no added architecture, raw-finger input, skin or extra body load. Use existing zero `center_mu` and identical initial visual-projection weights across arms within each seed; match minibatch order and permutations too. Train exactly 50 passes (500 updates at batch 64) per arm and seed 0/1/2, with AdamW lr 3e-4, weight decay 0.1 and soft tau 0.1. Save and use each fixed final checkpoint; do not select on dev. Preserve a common visual objective and training opportunities across V/A/S.

**Phase 2: language transfer.** Copy only the pretrained 512-d visual projection into identical VIDEO-ONLY language learners; discard the body branch and predictors. Keep the projection trainable. Initialize fresh word vectors and logit scale from the common initialization for each seed, use a fresh optimizer, and reset a shared per-seed minibatch-order stream across arms. Use corrected weak-noun speech outputs and cached features from `outputs/runs/step1_matched` (18,613 training pairs), without label-based filtering. Train 30 epochs, batch 256, AdamW lr 3e-4, weight decay 0.1, soft tau 0.1, language auxiliary weight zero; seeds 0/1/2 yield nine runs total.

**Evaluation and frozen decision gate.** Primary outcome is trial-weighted accuracy on corrected `clips_test` (570 queries, 115 target-clip clusters), averaged equally across the three seeds. For each arm, select the first maximum `picture_soft` accuracy on corrected `clips_dev` (486 queries, 119 clips). Pass only if A−V and A−S are each at least +5 percentage points, both contrasts are positive in all three seeds, and each 95% target-clip bootstrap lower bound is above zero. Use 10,000 draws, seed 0: resample the 115 source clips as whole clusters, retain all their trials, compute trial-weighted accuracy per seed and average the three seed accuracies equally. This gate is exploratory because the test sets have been previously consulted; reused candidate foils also induce dependence that the target-clip bootstrap does not account for. The +5 pp threshold is a pilot threshold, not a power claim.

Report s8 (787 queries, 158 clusters) secondarily. Report rendered held-out asset and Konkle picture transfer separately, each at the first maximum rendered-dev epoch. All language readouts are body-absent: this experiment does not test body-present understanding. Compare held-out prediction loss with mean/persistence descriptively only; it does not select checkpoints or add a gate. A clip-test gain alone does not prove object representation; reused picture probes are corroboration only.

**Execution boundary.** Before any run, implement only the minimum changes in the existing train/model/evaluate/CLI/tests path needed to realize this protocol. Run locally on MPS with a total budget of two hours for all nine planned runs; report timeout/incomplete runs as incomplete, and do not compare arms trained for unequal budgets. No sweeps, extensions, additional arms beyond the nine planned runs, new simulation, or architecture redesign. Proposed run root: `outputs/runs/step3_pretrain/` (already ignored); do not create it during planning. Step 3 is planned, not run, and later work is not automatically queued.

4. **Broader generalization (conditional).** Vary object instances, interactions and people, including similar grasps for different objects and different grasps for the same category. Keep video-only exposure matched; replayed frames do not substitute for independent experience.
5. **Learning curves (conditional).** Compare learners across independent amounts of training experience to distinguish faster learning from a higher eventual ceiling. Continue reporting body-present understanding and picture-only transfer separately.

This is the single direction document; completed experiments stay in the [archive](archive/README.md). Later stages remain conditional hypotheses, not an automatic experiment queue.

## Where to look

- [Archived language protocols and results through 4.15](archive/language_learning/experiments.md)
- [Earlier cue grid and confirmation](archive/language_learning/cue_experiments.md)
- [Research synthesis and alignment audit](docs/literature/body_input_review.md)
- [Learner implementation and diagnostic flags](docs/learner.md)
- [Speech-coupling evidence and parameter provenance](docs/speech_coupling.md)
- [Older simulator and motion work](archive/README.md)
