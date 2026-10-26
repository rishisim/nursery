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
- Last completed language comparison (4.13, seed 0): video-only 48.3% versus
  full body plus load/skin 42.4% on new clips from familiar people; s8 43.7%
  versus 40.9%. No confirmed language gain. Load/skin-only remains untested.
- Weak, delayed and nonheld naming are intentional. Both arms must receive the
  same speech; do not select ordinary learning examples using hidden referent
  identity. See the [design history](archive/language_learning/design_intent.md).

## Current step and next direction

1. **Trustworthy experiences and tests — implementation repairs verified.**
   Speech jitter stays within source clips while preserving weak, delayed and
   nonheld naming. Shared body-present scoring uses strict matched foils and
   aligned-frame renderer truth. This is not evidence of a language gain or
   proof that step 1 is scientifically solved. See the [implementation details](docs/learner.md#speech-alignment-and-body-present-scoring).
   The next comparison uses existing learners on matched corrected exposure;
   this step adds no experiment or training run.
2. **Small diagnostic capability (conditional).** If the verified comparison
   leaves a useful question, test whether relevant bodily information can
   support language on unfamiliar objects, for example heavy/light during
   lifting. Check that the sensory evidence is physically consistent and whether
   video already reveals it. Success would establish a useful capability, not
   broad noun learning or a higher language ceiling.
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
experiment queue. This implementation and documentation update included no new
experiment, training run, or persisted dataset regeneration.

## Where to look

- [Completed language protocols and results, steps 1–4.13](archive/language_learning/experiments.md)
- [Earlier 90-run cue grid and 100-run confirmation](archive/language_learning/cue_experiments.md)
- [Current research synthesis and alignment audit](docs/literature/body_input_review.md)
- [Learner implementation and diagnostic flags](docs/learner.md)
- [Speech-coupling evidence and parameter provenance](docs/speech_coupling.md)
- [Older simulator and motion work](archive/README.md)
