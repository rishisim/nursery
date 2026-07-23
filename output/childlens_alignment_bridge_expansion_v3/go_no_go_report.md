# ChildLens alignment-bridge measurement expansion v3

**Decision: NO-GO. Do not proceed to locked alignment scoring.**

This version added exactly ten preregistered, participant-distinct ChildLens
recordings outside all 30 prior participants. It reused the immutable v2
instruments, segmentation rules, metrics, thresholds, and eight-participant
results. It loaded no locked outcomes and ran no vision-language alignment,
simulator generation, or cue-condition training.

## What the additional recordings repaired

The combined 18-participant estimates now pass all three uncertainty gates that
failed in v2:

| Metric | v2 lower bound | v3 combined lower bound | Frozen minimum | Result |
| --- | ---: | ---: | ---: | --- |
| VAD boundary F1 | 0.527 | 0.772 | 0.650 | Pass |
| Matched character similarity | 0.256 | 0.758 | 0.400 | Pass |
| Matched embedding cosine | 0.741 | 0.865 | 0.700 | Pass |

Typical matched-segment agreement also remained strong: combined character
similarity was 0.866 and embedding cosine was 0.931. Thus, the ten-recording
expansion did resolve the original cross-participant precision concern.

## What still failed

The expanded instrument did not pass unchanged qualification gates:

- combined participants with accepted VAD speech remained below 0.900;
- combined primary and sensitivity nonempty item fractions remained below
  0.900;
- combined primary German classification remained below 0.800.

The independently reported new ten-participant cohort also failed the
preregistered replication safeguard:

- released-timing overlap recall was 0.594, below 0.600;
- primary transcript self-similarity under the frozen 250 ms expansion was
  0.605, below 0.700;
- primary embedding self-similarity under that expansion was 0.844, below
  0.850;
- primary German classification remained below 0.800.

These failures cannot be repaired by the strong combined medians or by the now
narrower confidence intervals.

## Diagnosis

- **German ASR:** still unqualified. Matched model–model content agreement is
  strong when both systems produce text, but primary language classification
  and primary boundary-shift robustness fail in the new cohort.
- **VAD/boundary estimation:** substantially improved but not fully qualified.
  Boundary-F1 median and uncertainty pass; new-cohort overlap recall falls just
  below its frozen gate and accepted speech remains absent for part of the
  combined sample.
- **ChildLens speech signal:** remains a plausible contributor because added
  participant-distinct data did not eliminate the zero-accepted-speech pattern.
- **German-speaking human validation:** remains necessary to distinguish
  genuinely non-German or unintelligible material from ASR/VAD failure.

All transcript, language, and timing findings are model–model or sensitivity
checks, never human validation, ground truth, or transcription accuracy.
Exact participant-level fractions with a nonzero complementary cell smaller
than five are suppressed in public artifacts.

## Recommendation

Stop this expansion. Do not add more recordings under v3, run locked
real-vs-shuffled/time-shifted alignment, fit a simulator, or train side-cue
conditions.

The scientifically bounded next option is a separately frozen German-speaking
human development audit of the unsupported and timing-sensitive signal. Any
later ASR/VAD change would require a new protocol and cannot reinterpret v1,
v2, or v3.
