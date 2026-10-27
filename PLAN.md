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
3. **Transfer-focused sensory pretraining — complete; frozen transfer gate failed.** The three arms were visual-only pretraining (V), visual pretraining with aligned sensory prediction (A), and visual pretraining with shuffled sensory prediction (S). Across three matched seeds, body-absent weak-noun test accuracy was V 52.63%, A 51.87%, S 52.87%. A−V was −0.76 pp (95% target-clip CI [−1.94, +0.42]); A−S was −0.99 pp (CI [−2.29, +0.36]); neither contrast passed any of its three criteria. Aligned sensory prediction was learned descriptively (A loss 1.4768 versus S 2.1508 and the training-mean reference 2.1026), but did not demonstrate weak-noun transfer. This does not prove harm or refute all pretraining approaches. All language evaluations were body-absent; the broader representation hypothesis remains unresolved. Step 3 is complete; no rerun, sweep or step 4 is queued. See the [4.16 protocol and result](archive/language_learning/experiments.md#step-3-transfer-focused-sensory-pretraining).
4. **Broader generalization (conditional).** Vary object instances, interactions and people, including similar grasps for different objects and different grasps for the same category. Keep video-only exposure matched; replayed frames do not substitute for independent experience.
5. **Learning curves (conditional).** Compare learners across independent amounts of training experience to distinguish faster learning from a higher eventual ceiling. Continue reporting body-present understanding and picture-only transfer separately.

This is the single direction document; completed experiments stay in the [archive](archive/README.md). Later stages remain conditional hypotheses, not an automatic experiment queue.

## Where to look

- [Archived language protocols and results through 4.16](archive/language_learning/experiments.md)
- [Earlier cue grid and confirmation](archive/language_learning/cue_experiments.md)
- [Research synthesis and alignment audit](docs/literature/body_input_review.md)
- [Learner implementation and diagnostic flags](docs/learner.md)
- [Speech-coupling evidence and parameter provenance](docs/speech_coupling.md)
- [Older simulator and motion work](archive/README.md)
