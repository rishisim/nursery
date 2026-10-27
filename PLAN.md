# Current Nursery plan

## Question

Does bodily experience improve language acquisition over the same video and
speech alone? The long-term hypothesis is a higher learning plateau. A gain at
one data amount or a fixed-data training plateau does not establish that claim.

Keep two outcomes separate: understanding with body inputs present, and
picture-only transfer after learning with body inputs. A reliable gain in either
advances the build; neither is established by a supervised sensor diagnostic.

## Current checkpoint

- Data: `outputs/renders/clean_856`; speech, embeddings and matched learner runs:
  `outputs/runs/clean_body`. Existing data and runs are preserved in place.
- Baseline: the additive linear learner with the working 512-dimensional
  visual/word representation. The joint Transformer remains a historical
  comparison, not the default next architecture.
- Latest matched corrected-exposure comparison (4.14, seed 0): body-present
  clips-test was 52.28% video-only versus 47.54% full body plus load/skin
  (difference −4.74 pp, paired-trial 95% CI [−8.60, −0.88]); s8 was 48.03%
  versus 46.00% (−2.03 pp [−5.46, 1.53]). Picture-only transfer was 67.12%
  versus 66.22% rendered and 64.60% versus 64.02% Konkle. This fixed-data,
  single-seed result shows no body benefit; it does not test a higher ceiling.
  Independent review passed. Full protocol and limits are in the
  [experiment record](archive/language_learning/experiments.md#414-matched-corrected-exposure-rerun).
- Weak, delayed and nonheld naming are intentional. Both arms must receive the
  same speech; do not select ordinary learning examples using hidden referent
  identity. See the [design history](archive/language_learning/design_intent.md).

## Current step and next direction

1. **Trustworthy experiences and tests — closed.** Speech-alignment and
   body-present scoring repairs and the matched comparison are complete and
   independently verified. The comparison found no body benefit in its primary
   or picture-only outcomes; its single-seed intervals are conditional on the
   fixed models and trials, and the old 4.13 scores are not directly comparable.
   This closes the repair-and-comparison task, not the broader scientific
   question or a claim of benefit. See the [implementation details](docs/learner.md#speech-alignment-and-body-present-scoring) and [matched rerun record](archive/language_learning/experiments.md#414-matched-corrected-exposure-rerun).
2. **Small diagnostic capability — complete; frozen gates and independent review passed.**
   With scaffolded heavy/light words, video plus 10-d load scored 92.42% on the
   two held-out categories versus 37.88% for video (Δ +54.55 pp, paired source-
   clip 95% CI [+36.36, +71.21]); load derangements reduced accuracy 46.21 pp.
   The video result is below 50% chance, so the gap does not show that video
   cannot infer weight. Picture-only scores were 40.91% after load training and
   37.88% for independent video, both below chance; no useful transfer was
   established. This narrow synthetic diagnostic does not test ordinary weak
   noun learning or a higher language ceiling. See the [step 2 record](archive/language_learning/experiments.md#step-2-heavy-light-capability-test).
   Step 3 and later work remain conditional; this result does not auto-queue it.
3. **One embodied learning mechanism (conditional).** Test whether coherent
   sensory experience helps form reusable object representations before sparse
   speech names them.
   Preserve the working visual baseline and give video-only controls the same
   temporal history and comparable training opportunities. Test one mechanism
   at a time; sensory pretraining remains a hypothesis, not a demonstrated fix.
4. **Broader generalization (conditional).** Vary object instances, interactions
   and people, including similar grasps for different objects and different
   grasps for the same category. Keep video-only exposure matched. More replayed
   frames do not substitute for independent experience.
5. **Learning curves (conditional).** Compare learners across independent
   amounts of training experience to distinguish faster learning from a higher
   eventual ceiling.
   Continue reporting body-present understanding and picture-only transfer
   separately.

This is the single direction document; completed experiments stay in the
[archive](archive/README.md). Later stages are conditional hypotheses requiring
bounded, recorded protocols and decisions from earlier results, not an automatic
experiment queue.

## Where to look

- [Archived language protocols and results through 4.15, including the step 2 diagnostic](archive/language_learning/experiments.md)
- [Earlier 90-run cue grid and 100-run confirmation](archive/language_learning/cue_experiments.md)
- [Current research synthesis and alignment audit](docs/literature/body_input_review.md)
- [Learner implementation and diagnostic flags](docs/learner.md)
- [Speech-coupling evidence and parameter provenance](docs/speech_coupling.md)
- [Older simulator and motion work](archive/README.md)
