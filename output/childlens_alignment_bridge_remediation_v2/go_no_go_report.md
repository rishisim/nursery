# ChildLens alignment-bridge measurement remediation v2

**Decision: NO-GO. Do not proceed to locked alignment scoring.**

The remediation repaired an important part of the v1 measurement failure, but it did not qualify the instrument across the full development sample. This is a development-only model–model sensitivity result, never human validation, ASR accuracy against ground truth, or a causal result.

## What changed

V1 remains untouched. V2 reused only its eight frozen development participants and loaded no locked outcomes.

The remediation froze one dedicated segmentation route—Silero VAD 6.2.1—and fed byte-identical accepted segments to:

- primary: unquantized Whisper large-v3;
- sole sensitivity: unquantized Whisper large-v3-turbo.

Both used the same pinned `whisper.cpp` build and automatic language identification. A pinned multilingual MiniLM supplied embedding agreement. No other ASR, VAD, aligner, threshold, or text model was tried.

## Separate diagnosis

### 1. Speech segmentation and boundaries

The shared VAD produced 85 accepted segments and 307.422 speech seconds. Typical timing behavior was strong:

- participant-median overlap precision with released speech-like timing: 0.980;
- participant-median overlap recall: 0.719;
- participant-median silence-padding self-consistency boundary F1: 0.911;
- median absolute duration change: 0.013 seconds;
- median perturbed/base speech-coverage ratio: 1.016.

However, boundary-F1 uncertainty was wide: the participant-bootstrap 90% interval was 0.527–0.986, below the frozen 0.65 lower-bound requirement. Not every development participant produced accepted VAD speech. The unsupported audio was not mechanically silent and had released speech-like timing support, leaving a genuine ambiguity between VAD domain failure and unsuitable/non-speech-like signal.

### 2. Transcript and language stability on matched audio

On byte-identical segments, median agreement improved markedly:

- primary and sensitivity nonempty item fractions: 0.875 each, below 0.900;
- primary German classification and model–model language agreement: 0.875 each;
- participant-median normalized character similarity: 0.819;
- character-similarity bootstrap 90% interval: 0.256–0.948, below the frozen 0.40 lower-bound requirement;
- participant-median multilingual embedding cosine: 0.922;
- embedding bootstrap 90% interval: 0.741–0.985.

Small timing-shift robustness passed. Expanding every segment by 250 ms yielded character self-similarity medians of 0.976 for large-v3 and 0.888 for turbo, with embedding self-similarities of 0.998 and 0.979.

Thus, German ASR content agreement is no longer the dominant median failure on supported segments. It is still not qualified at the participant level because coverage and the character lower bound fail.

### 3. Was v1 mostly a boundary mismatch?

Yes, boundary mismatch was a major contributor. The participant-median character similarity increased from 0.126 with independently generated v1 boundaries to 0.819 on v2 matched segments—an absolute improvement of 0.693.

It was not the whole problem. The lower confidence bound and participant coverage still fail, so matching boundaries does not justify locked scoring.

## Failure classification

- **Inadequate German ASR:** not the dominant median failure on supported segments, but the route remains unqualified because participant coverage and the character lower bound fail.
- **Inadequate VAD/boundary estimation:** partially supported. Median and perturbation behavior pass; cross-participant lower-bound stability fails.
- **Insufficient or unsuitable ChildLens speech signal:** supported as a stop reason because accepted speech is absent for part of the development sample despite non-silent audio and released speech-like timing support.
- **No German-speaking human validation:** material and unresolved. Model–model agreement cannot establish whether the unsupported audio contains intelligible German speech or whether either transcript is correct.

## Recommendation

Stop this measurement remediation. Do not run locked real-vs-shuffled/time-shifted alignment, generate simulator episodes, or train cue conditions.

The bounded next option is an authorized, separately frozen German-speaking human development audit focused on the unsupported/unstable signal. Do not begin another ASR/VAD model search under this protocol.
