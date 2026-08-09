# ChildLens v1.2 offline-instrument availability and firewall

Version: `childlens-offline-instrument-availability-v1.2.0`  
Audit date: 2026-07-21  
Scope: local package/executable inventory; no restricted input

## Disposition

Structural decode is ready. Automated ASR, alignment, language identification, diarization, voice-type classification, and reliability scoring are not ready on this host. The permitted and technically runnable v1.2 route is therefore:

1. FFmpeg/FFprobe for local structural media checks;
2. genuine qualified humans for language establishment, timing, source-language text, speaker role, and reference;
3. no automated speaker-role instrument; and
4. no external pretrained artifact crossing into learner ancestry.

This inventory did not download a package, model, weight, or dataset; accept a license or gated condition; access restricted data; or use an external API.

## Installed components

| Component | Observed state | Fixed use |
| --- | --- | --- |
| FFmpeg / FFprobe / FFplay | Version 8.0.1; binary digests match the v1.1 preflight | Local structural probe, error-fatal decode, and local playback fallback only |
| Streamlit | Version 1.56.0; matches the v1.1 pin | Local loopback human interface, subject to the separate UI security sentinel |
| Python 3.10 | Present as established in v1.1 | Candidate isolated runtime only |

The FFmpeg build enables GPL and version 3. It is treated as GPL v3-or-later for local executable use and must not be redistributed in the report bundle.

## Components absent or blocked

Targeted package metadata inspection found no local installation of faster-whisper, WhisperX, OpenAI Whisper, CTranslate2, PyTorch, torchaudio, pyannote.audio, SpeechBrain, Silero VAD, JiWER, or the Krippendorff package.

- The previously pinned faster-whisper 1.2.1 and WhisperX 3.8.5 specifications remain candidate measurement instruments, not runnable tools.
- Actual language must be established by qualified humans before any language-specific aligner is selected.
- Pyannote model acquisition remains prohibited because its model conditions require separate user action and review. No condition was accepted.
- VTC remains prohibited because code/weight licensing was unresolved in v1.1.
- Code-package licensing never establishes permission for separately distributed weights.
- No cache or filesystem-wide weight search was used in this v1.2 inventory. Absence here means the required package/runtime is not installed, not a claim that every byte on the host was searched.

No ASR or alignment output should be promised for this pilot unless a separately approved, immutable, hash-verified offline environment is actually created. The human-first workflow does not depend on it.

## Instrument-to-learner firewall

The following prohibition is unconditional for this task:

- no instrument weight, checkpoint, tokenizer, vocabulary, token ID, prompt, feature, embedding, hidden state, logit, confidence, alternative hypothesis, VAD segment, diarization cluster, detector output, or unreviewed transcript/timing/role enters learner inputs, tokenizer construction, checkpoints, causal arms, or evaluation;
- automated output, if later approved, stays in a proposal-only quarantine and is never gold;
- every future scientific text/timing/role record requires a complete qualified-human disposition against source media;
- instrument identities/settings may enter provenance receipts but not learner examples; and
- no learner, tokenizer, checkpoint, or causal arm is authorized in v1.2.

Speaker role remains the direct human label set `NON_CHILD`, `CHILD`, `OVERLAP`, `UNCERTAIN`, and `NONSPEECH`. Anonymous diarization clusters cannot establish those roles. Referential status also remains human-only under the six frozen labels and the fixed context window.

## Reliability implementation

Neither JiWER nor the Krippendorff package is installed, so the scoring implementation is not frozen. This does not block packet construction or first-pass human annotation, but public gate scoring must fail closed until exact local metric code is pinned and synthetic edge-case tests pass. A small reviewed local implementation is preferable to acquiring another unreviewed model stack; its specification must be frozen before human results are inspected.

## Provenance

The machine inventory inherits, without modifying, the v1.1 instrument-preflight and proposed annotation-instrument-amendment digests. It does not silently adopt the proposed amendment or weaken the scratch learner boundary. The current user's instruction permits local fixed instruments only when separately licensed; no absent or gated instrument qualified for execution in this workstream.

