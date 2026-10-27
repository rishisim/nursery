# Small experiment: learning words across ambiguous encounters

Design reference for a small fixed-data comparison alongside the
[main study](acquisition_curves.md). No run has started.

## The idea

Heavy/light showed that the learner can connect informative body signals to
words. Here we introduce a different challenge: each encounter contains several
possible referents, so the learner has to accumulate evidence across encounters.

For example, suppose “dax” names a category of object:

| Encounter | Heard | Seen |
|---|---|---|
| 1 | “Look, dax.” | An object from that category and a spoon |
| 2 | “Look, dax.” | Another view or example from that category and a ball |
| 3 | “Look, dax.” | Another view or example from that category and a book |

The learner receives observations, not the category labels in this table.
Changing distractors make the repeated relationship informative. Body signals
may supply additional evidence about the objects and interactions.

## Keep it small and controlled

Start with four object categories, each assigned a fixed unfamiliar word form.
Choose them for independent clip coverage, visible ambiguity and available
sensors—not learner performance. Reuse existing renders and features after
confirming they supply suitable distractor variation.

Each training utterance names a visible category, but multiple objects remain
plausible within that encounter. This is intentionally more scaffolded than the
main weak-speech task. Include both held and nonheld named objects at a fixed,
recorded balance: “held” must not be a perfect answer rule. Keep distractor,
subject and background associations from deterministically revealing the word.

Use all five body streams, the same time windows and the same language loss as
the main study. Labels may construct and score this controlled dataset, but
category IDs, referent labels and hidden physical parameters are never learner
inputs. The vocabulary and language are deliberately simplified.

## How we run it

Use two conditions: video only and additive video + body. There is no joint
learner in this small study. Their architecture is defined in the
[main design](acquisition_curves.md#the-three-learners), not implemented separately here.

Use one fixed small training set, with three paired seeds. Balance word exposure
and report independent clips and utterances per word. Keep the word/category
mapping and test panel fixed. There is no training-data sweep.

Split by source clip so replays cannot cross partitions. Train fresh models for
each seed, select checkpoints on development data, and test on unfamiliar clips
and varied views. New-instance testing requires genuinely held-out instances.

Testing uses four candidate situations and one learned word, with the same
candidate choices for all learners. Exactly one candidate contains the target
category within the permitted context. Body-enabled learners receive each
candidate's own sensory history. Chance is 25%.

As a supporting check, evaluate the same trained additive models with complete
body histories shuffled across situations. This is evaluation only, not another
training condition. Keep channels together within each history and freeze the
donor and missing-data rules before evaluation.

## The output we want

**A compact results table**, like the heavy/light presentation, reporting mean
held-out accuracy and uncertainty across seeds. The desired result is stronger
word understanding with body input after the same ambiguous encounters. This
is the hypothesis, not a promised result. No acquisition curve is produced here.

| Condition | Held-out accuracy |
|---|---|
| Video only | — |
| Additive video + body | — |

Report shuffled-body accuracy in one supporting sentence below the table.

The blanks are future measurements. For reference, the earlier heavy/light
table reported video 37.88%, video + load 92.42%, and shuffled load 46.21%.
Those were a different, two-choice task with 50% chance; they are a presentation
example, not numerical targets for this experiment. See the
[saved heavy/light record](../archive/language_learning/experiments.md#step-2-heavy-light-capability-test).

Include a few example encounters beside the table. Acquisition curves and the
joint-architecture comparison remain in the main study.

## Running in parallel

After shared implementation checks, train separate models with separate
manifests and ignored run directories. Do not use this study's weights or
measurements to change an already-running main comparison.

Before launch, confirm category/clip feasibility, freeze sampling and training
settings, and set a short runtime cap from a timing check. Runtime is not yet
measured. New rendering and open-ended sweeps are outside this small study.
