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

## Immediate order

1. Preserve completed experiments and clarify historical intent before changing
   the protocol. The [archive](archive/README.md) now holds the completed record.
2. Resolve the narrower experience/window and scoring questions: contact can
   support a held-channel name across a hard clip cut that resets body memory;
   distractor scenes can also contain the target object. The audits establish
   these conditions, not their causal contribution to the failed learning runs.
   Preserve natural ambiguity and delayed naming when defining any repair.
3. Specify and validate the smallest correction, then compare the existing
   video-only and body learners on identical revised exposure and defensible
   trials. Record the protocol before new training; do not bundle architecture
   changes, new sensors or privileged filtering into this comparison.
4. Choose the next single diagnostic from that result. Separate better use of
   current signals, sensory representation learning, and simulator changes.
   Scaling across independent data amounts follows a trustworthy comparison.

This archive checkpoint changes documentation and historical aggregate locations
only. Connection/scoring repairs, new training and architecture changes are
deferred; the historical experiment queue is not an instruction to rerun it.

## Where to look

- [Completed language protocols and results, steps 1–4.13](archive/language_learning/experiments.md)
- [Earlier 90-run cue grid and 100-run confirmation](archive/language_learning/cue_experiments.md)
- [Current research synthesis and alignment audit](docs/literature/body_input_review.md)
- [Learner implementation and diagnostic flags](docs/learner.md)
- [Speech-coupling evidence and parameter provenance](docs/speech_coupling.md)
- [Older simulator and motion work](archive/README.md)
