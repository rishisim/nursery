# Current Nursery plan

## Active objective

Measure direct language-acquisition curves for video-only (V) versus the same
video plus bodily inputs (V+B), across amounts of independent training
experience. The long-term hypothesis is a higher language-learning ceiling;
the existing experiments neither establish that benefit nor rule it out.

## Current direction and five steps

Improve the bodily information available to the learner before choosing how to
pretrain it or introducing another fusion design. Preserve useful detail already
present in the recordings and explicitly record sensations that require new
physical modeling or measurements. Active exploration remains deferred.

The documents below follow the earlier study format: the idea, the experience
or comparison, how it would work, and the output we want. They are design
references, not frozen execution protocols or authorization to launch runs.
Concrete features, data amounts, seeds, budgets and runtime limits remain to
be settled before their respective experiments.

| Step | Document | Question and current status |
|---|---|---|
| 1 | [Visual baseline](docs/visual_baseline.md) | Use 16×16 vision and retain recorded 4×4 results. Code support is preserved on the acquisition branch; new main-task measurements are pending. |
| 2 | [Body representation](docs/body_representation.md) | What contact, pose and temporal detail can existing information supply, and which sensations remain unavailable? Design only. |
| 3 | [Representation comparison](docs/body_representation_comparison.md) | Does richer body help under the same additive setup, without wordless pretraining? Design only. |
| 4 | [Body training recipe](docs/body_training_recipe.md) | After the representation is understood, does wordless pretraining help? Additive remains the default; joint interaction is deferred. |
| 5 | [Language acquisition curves](docs/language_acquisition_curves.md) | Does the selected video-plus-body setup help across amounts of experience? This is the main outcome. |

Steps 1 and 2 meet in Step 3. Step 4 settles the learning recipe before Step 5.
A weak intermediate result is a design signal for the build, not a standalone
paper framing or an automatic instruction to launch more conditions.

## Decisions to keep consistent

- Compare 16×16 video, 16×16 video plus current body, and 16×16 video plus richer
  body at one fixed data amount. Keep the recorded 4×4 video result as a labeled
  historical reference. Quantify a resolution effect only when task, data,
  trials and training settings are comparable.
- Use additive fusion and no wordless pretraining for the initial representation
  comparison. Apply the same development-only body-weight selection rule to
  both body conditions; test outcomes do not tune the design.
- Preserve natural weak speech, source-clip partitions and accessible time
  spans. Finer temporal sampling does not grant a longer body window. Record
  necessary encoder-capacity and compute changes.
- If wordless pretraining is selected for the curves, use only each point's own
  selected training clips, including normalization. Extra optimizer passes are
  compute, not new independent experience. Both arms share DINO's pretrained
  visual prior; this is not perception learned from scratch.
- Count independent source clips on the experience axis. Separate uncertainty
  across training seeds from uncertainty across test clips. Reused panels
  remain exploratory; repeated seeds are not new test situations.
- Keep new physical-property tasks, substantial new sensory simulation, active
  exploration and joint-architecture searches deferred.

## Preserved starting points

The completed [cross-situational study](archive/language_learning/cross_situational_learning.md)
and [original acquisition curves](archive/language_learning/acquisition_curves.md)
are archived with their protocols, results and source revisions. The
[archive index](archive/README.md#completed-study-worktrees) identifies their
full local worktrees, runs and compatibility paths.

The 16×16 implementation is commit
[`edb6b1a4`](https://github.com/rishisim/nursery/commit/edb6b1a47474bc08cd72ebc795e782b7eed0675a)
on `research/acquisition-curves`. It passed 83 software checks, not a new
scientific experiment. It is preserved for reuse and has not been merged into
this `research/embodied` checkout by the documentation/archive work.

- Keep `outputs/renders/clean_856/`, saved speech/features and past run roots in
  place. They are retained inputs/evidence, not an active experiment queue.
- Read [Past experiments](archive/language_learning/past_experiments.md) before
  designing a new comparison; [experiments.md](archive/language_learning/experiments.md)
  remains the detailed historical protocol/result record.
- Use [the learner reference](docs/learner.md), [speech evidence](docs/speech_coupling.md)
  and [design intent](archive/language_learning/design_intent.md) for existing
  implementation and weak-alignment constraints. The [archive](archive/README.md)
  also holds earlier simulator work and the historical literature review.
