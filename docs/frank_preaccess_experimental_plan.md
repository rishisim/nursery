# Toward a BabyView-calibrated sensorimotor cue study

**Pre-access experimental plan for discussion with Professor Michael Frank**  
**Rishi Simhadri, UT Dallas | July 2026**

## Page 1 - Scientific question and completed pre-access infrastructure

### Action-first question

**Does synchronized low-level motor experience improve action-language grounding when visual and linguistic experience is only weakly aligned?**

This is the first, deliberately narrow test in a broader program asking whether action, proprioception, and touch can scaffold grounded language learning. Physical forward prediction remains an auxiliary manipulation check that the side channels are informative; language grounding is the primary scientific outcome.

### Rationale and causal comparison

BabyView provides naturalistic egocentric vision, audio, and language, but a head camera cannot directly record motor commands, proprioception, contact force, or exact causal state. A simulator can preserve a selected set of measured BabyView distributions while generating those synchronized side channels. The key comparison is therefore not simply more inputs versus fewer inputs. It is correctly synchronized experience versus an identically structured but uninformative cue stream.

```text
BabyView aggregate measurements (after access)
                    |
                    v
       versioned distribution configuration
                    |
                    v
  raw video + timed utterances + motor/touch streams
                    |
                    v
        controlled paired training arms
                    |
                    v
 video-language test with motor input withheld
```

### What is implemented before access

The new generator is driven by a YAML configuration explicitly labeled `provisional_not_babyview_matched`. Its rates can be replaced by measured aggregates without changing the experimental logic. The current validated build produces 96 base episodes and 1,152 factorial examples with:

- Strong, weak, and episode-shuffled language alignment.
- Null, synchronized, split-local episode-shuffled, and time-shifted whole motor trajectories.
- Multiple visible objects, timestamped utterances, clean RGB frames, and low-level hand position/velocity.
- Composition- and hash-disjoint splits with all frames from an episode kept together.
- A strict model-input allowlist and physically separate oracle/causal records.

The persistent audit reports no model-input leakage, shuffled self-match, cross-split donor, split-hash overlap, or split-composition overlap. The training renderer contains no action/event/force caption and no artificial contact marker.

### Learner and training arms

A small video-text contrastive learner now contains the same video, text, and temporal motor encoders in every arm. Initialization, data order, optimizer schedule, examples, and compute are paired by seed. The motor stream contains only continuous hand `x/y/vx/vy`; categorical words such as *push* and *grasp* are never motor inputs.

| Arm | Motor during training | Motor in primary test |
| --- | --- | --- |
| Null | Masked sequence | Withheld |
| Episode-shuffled | Another same-split episode's whole trajectory | Withheld |
| Time-shifted | Correct trajectory at the wrong time | Withheld |
| Synchronized | Correct trajectory at the correct time | Withheld |

<div style="page-break-after: always;"></div>

## Page 2 - Primary test, pilot status, calibration, and confirmatory design

### Primary test

The locked primary evaluation is a balanced action-verb minimal-pair test on held-out object-action compositions. It calls only the video and text encoders. A unit test makes the run fail if the motor encoder is invoked during primary evaluation.

### Current pilot: pipeline validation, not an effect claim

All 24 repository tests pass. A three-seed weak-alignment pilot ran end to end on the provisional corpus. Synchronized motor training did not outperform the episode-shuffled or null controls on held-out action grounding. This small single-corpus run is therefore strictly infrastructure validation, not preliminary evidence that motor cues improve language grounding. Exact exploratory estimates and uncertainty intervals are retained in the reproducibility report rather than promoted as a scientific result.

### BabyView calibration after access

Subject to the data-use agreement, only non-identifying aggregate measurements needed for calibration will be exported. Each measurement will be versioned with its definition, unit, sample size, uncertainty, and limitations. The first pass will prioritize:

- **Language:** word and part-of-speech frequency, utterance rate and length, repetition, and silence.
- **Alignment:** visual-speech lag, speech near event boundaries, visible-referent rate, irrelevant speech, and ambiguity, with manual checks on a subset.
- **Vision and interaction:** candidate-object density, visibility/occlusion, object size, camera motion, activity duration, and interaction frequency where reliably measurable.

The goal is selected, transparent distribution matching, not complete ecological equivalence. Until these measurements replace the provisional values, the corpus will not be called BabyView-like or BabyView-matched.

### Confirmatory experiment and decision rule

The confirmatory unit will be the whole episode, with independent corpus seeds and paired model seeds. The primary estimand is:

**Delta = action-grounding score(synchronized) - action-grounding score(episode-shuffled).**

A language-effect claim will require all leakage and fairness gates to pass; a lift of at least 3 absolute accuracy points with a paired hierarchical 95% confidence interval above zero; synchronized performance above both shuffled and null controls; consistent direction across at least 7 of 9 paired corpus/model runs; and selectivity for action-related trials rather than matched noun/color controls. High alignment must outperform shuffled alignment as a positive control. Null or mixed results will be reported without changing the endpoint.

Secondary analyses will cover action-result grounding, video-text retrieval, frequency/alignment strata, held-out compositions, vision/text encoder quality, and the existing physical forward-prediction benchmark. Touch and proprioception will be added only after the action-first design is validated.

### Relationship to EgoBabyVLM

The released Machine-DevBench protocol and model interface have been mapped, but no official score has been reproduced locally: its pinned environment is Linux/CUDA-first rather than Apple MPS. The fixed benchmark will be used only after checking vocabulary coverage; a genuinely corpus-grounded synthetic version requires separate stimulus generation. Synthetic-trained models are not eligible for the BabyView-only challenge, so this work is framed as a complementary causal study rather than a leaderboard submission.

### Claim boundary and next decision

A future positive result would show that synchronized motor experience can improve action-language grounding in this controlled learner. It would not establish the same mechanism in infants, complete BabyView equivalence, or that synthetic data can replace natural data. The immediate post-access decision is which BabyView statistics to prioritize before scaling the paired experiment and expanding it to proprioception and touch.
