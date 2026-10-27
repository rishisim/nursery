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
2. **Small diagnostic capability — active; protocol frozen and implementation validated.**
   Test whether load supports binary heavy/light word learning on held-out
   object categories excluded from diagnostic training images. Use existing
   data split by independent clips (88 train, 18 dev,
   22 test); categories are train: doll/rock/tube/bottle/trumpet/bowl, dev:
   clock/train, test: pyramid/stapler. Heavy is ≥300 g, light ≤150 g; exclude
   the middle, balance both labels per category, and use one eligible held/lifted
   center per clip with five same-clip RGB frames at ±2 s. Exclude dev/test
   categories from train windows and test categories from dev windows. Use
   scaffolded heavy/light utterances paired with lifting, not ordinary weak
   naming. Mass and metadata generate labels and audits only, never model input;
   normalize from train-accessible frames only. Use the existing additive
   512-d word/visual learner, 3 fixed seeds (0/1/2), 100 epochs, batch 16,
   lr 3e-4, weight decay 0.1, soft tau 0.1,
   body dropout 0.5 and auxiliary losses zero. Compare video plus 10-d load
   against the same video without load. Evaluate a dev-selected binary word
   test (chance 50%). Run six models total, capped at 30 minutes, with no
   autotuning; compare body-present results with the independent video models
   and report body-trained models' picture-only ablation separately. Pass
   requires mean body accuracy ≥75%, mean paired gain ≥15 pp, positive gain in
   all seeds, and a 10,000-replicate paired source-clip bootstrap of mean seed
   gain (seed 0) with lower bound above zero; 10 fixed within-test-category
   load derangements must reduce mean accuracy by ≥10 pp. Independent review
   found no issues; all 72 tests passed (19 property, 53 existing learner).
   Preflight confirmed the 88/18/22 clip split and 665 normalization seconds.
   The protocol is ready for the bounded six-model run; no models have trained.
   This is a narrow capability diagnostic: load is analytic F=m(g−a) on fixed GRAB
   motions and does not change motion or RGB; pretrained DINO novelty and two
   held-out categories limit interpretation. It cannot establish infant-like
   weak noun acquisition or a higher language ceiling.
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

- [Completed language protocols and results, steps 1–4.14](archive/language_learning/experiments.md)
- [Earlier 90-run cue grid and 100-run confirmation](archive/language_learning/cue_experiments.md)
- [Current research synthesis and alignment audit](docs/literature/body_input_review.md)
- [Learner implementation and diagnostic flags](docs/learner.md)
- [Speech-coupling evidence and parameter provenance](docs/speech_coupling.md)
- [Older simulator and motion work](archive/README.md)
