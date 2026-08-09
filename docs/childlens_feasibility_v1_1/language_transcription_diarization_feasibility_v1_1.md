# ChildLens v1.1 language, transcription, timing, and speaker-role feasibility

Version: `childlens-transcription-feasibility-v1.1.0`  
Audit date: 2026-07-21  
Scope: measurement readiness and human-validation handoff only  
Scientific learner outcome authorized: **no**

## Updated finding

The v1.1 continuation has not produced empirical language, transcript, timing, or speaker-role evidence. No ChildLens video or audio was acquired or decoded, the actual language or languages remain unknown, no ASR/alignment model or weights are installed, and no human has reviewed the selected items. The 15-item blinded packet is an empty structural skeleton, not a validated sample. Accordingly:

- `G3_AUDIO_TIMING = UNRESOLVED`
- `G4_TRANSCRIPT_ROLE = UNRESOLVED`

This is not evidence that ChildLens audio is unusable. It is a fail-closed statement that usability has not yet been measured. It also preserves the v1 decision history: this report supplements and does not edit or retroactively pass the v1 audit.

## What v1.1 established

Local engineering readiness improved in bounded ways:

- Homebrew FFmpeg `8.0.1_4` is fixed as the local decoder. Its `ffmpeg` executable has SHA-256 `0a96da2735695308d964e25fa6f4a0db2e9d24031390360f4c5ff96a4f8938e5`. A two-pass synthetic sentinel deterministically produced mono 16 kHz `pcm_s16le` audio with matching output digests.
- Python `3.10.20` is available for a future isolated speech-instrument environment.
- `faster-whisper==1.2.1`, `whisperx==3.8.5`, and the fixed `Systran/faster-whisper-large-v3` revision remain acquisition specifications only. They and their dependencies/weights are not installed or cached.
- The proposed VTC code and weights remain excluded because explicit license metadata was not established. Pyannote remains excluded because acquisition requires user acceptance and contact-information sharing, and anonymous clusters would not establish child versus other-person role.
- A local human-only route is structurally available. It does not depend on ASR, alignment, VTC, or diarization weights.

These findings come from [instrument_preflight.md](instrument_preflight.md) and its [machine-readable receipt](../../output/childlens_feasibility_v1_1/instrument_preflight.json). They establish tooling readiness only; they do not satisfy a corpus gate.

## Current acquisition and evidence boundary

The frozen preselection contains 15 blinded items, one from each of 15 participant groups, selected without lexical, audio, visual, ASR, or learner content. Its empty human-packet skeleton has SHA-256 `ae8503fc5c21fc6df0b08c763202eab074b2526f83ef847d2faca3ae4eb217a5`.

That packet remains `SKELETON_AWAITING_MEDIA_AND_HUMANS`. Exact remote byte sizes for the selected video objects have not been established, the canonical final restricted manifest is incomplete, and acquisition has not started. Placeholder sizes are not admissible scientific evidence. The packet digest proves only that a fixed 15-item assignment skeleton existed before content inspection; it proves nothing about speech, language, intelligibility, timing, role prevalence, or lexical content.

No restricted media, decoded audio, transcript text, exact timestamp, participant identifier, or row-level annotation is included in this report.

## Fixed measurement route

### 1. Admission before content

Before any selected object is copied, the coordinator must resolve exact remote bytes, finish the deterministic restricted manifest, rerun the 20 GiB raw-pilot, 73 GiB namespace, and 50 GiB post-peak-free-space admission checks, and verify the restricted quarantine controls. Selection and assignment order cannot change in response to file content, language, speech density, lexical richness, or visible-referent yield.

### 2. Deterministic local decode

For each admitted selected object, decode the first audio stream locally with the fixed FFmpeg executable, explicit `0:a:0` selection, no video output, mono 16 kHz signed 16-bit PCM, metadata removal, and bit-exact flags. Keep source objects immutable. Media paths, exact durations, decode diagnostics tied to an item, and audio remain in the untracked restricted quarantine.

Technical decode is necessary but not sufficient. `G3` still requires at least 80% decodable sampled duration and human-correctable utterance boundaries in at least 70% of speech-bearing sampled windows. A corrected onset and offset count as usable only when each is within 500 ms of the adjudicated reference, ordered, and consistent with overlap status.

### 3. Human-led language routing

The actual language cannot be inferred from geography, release labels, annotator assumptions, or an ASR model. Two qualified humans independently assign an ISO 639 language code, `MIXED_OR_CODE_SWITCHED`, or `UNDECIDABLE` from the admitted source audio before any forced-alignment route is chosen. Machine language identification may later be recorded only as an advisory discrepancy signal.

Translation is prohibited. Source-language transcription is required. Mixed or unsupported language spans use manual utterance timing unless a separately licensed, fixed aligner is approved before the span is processed.

### 4. Human-only base path

The controlling base path is manual:

1. Independently identify language and speech/non-speech status.
2. Independently mark utterance onset/offset and overlap.
3. Transcribe intelligible source-language speech verbatim, using explicit unintelligible markers rather than guessing.
4. Assign one of `NON_CHILD`, `CHILD`, `OVERLAP`, `UNCERTAIN`, or `NONSPEECH`.
5. Lock both independent records before adjudication.

This route may pass the scientific gates without machine speech proposals if qualified labor and reliability are adequate. Codex may construct the local packet, test blinding, compute prespecified metrics, and audit privacy; Codex cannot serve as either human judge or adjudicator.

### 5. Optional fixed ASR/alignment proposals

ASR may be added only after project authority approves the proposed annotation-instrument amendment and the code and model licenses are checked and receipted. The fixed candidate remains `faster-whisper==1.2.1` with `Systran/faster-whisper-large-v3` revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`; `whisperx==3.8.5` is the conditional alignment interface. Installation and inference must occur inside a hash-verified, network-disabled local quarantine.

Any ASR text, timing, VAD segment, confidence, alternate hypothesis, tokenizer, token ID, or internal representation is a machine proposal and is never gold. Every accepted record must be fully verified or corrected against the source by a qualified human. A prespecified ASR-hidden reference subset is mandatory to measure anchoring bias. If the optional stack is not acquired, the manual route remains controlling.

## Frozen validation block and metrics

Accumulate the content-blind, stratified validation block until the first of 300 utterances or 30 speech minutes. All timing and speaker-role validation items are independently double-coded; at least 20% of the reference transcription block is hidden from any machine proposal. Stratification uses participant group, fixed speech/activity metadata, duration, and established language route—not words or apparent grounding quality.

Required evidence is:

| Measurement | Frozen disposition |
| --- | --- |
| Technical audio coverage | Pass requires at least 80% of sampled duration decodable. |
| Human-correctable timing | Pass requires at least 70% of speech-bearing sampled windows to have usable boundaries; both edges must be within 500 ms of the adjudicated reference. |
| Timing diagnostics | Median and 90th-percentile onset/offset error, insertions, deletions, overlap and censoring errors. |
| Speaker role | Production-corrected four-class macro-F1 across `NON_CHILD`, `CHILD`, `OVERLAP`, and `UNCERTAIN` must be at least 0.80; report classwise precision/recall and nonspeech false detections. |
| Transcript quality | Report language-appropriate token or grapheme error against the blinded adjudicated reference, before and after correction, plus unresolved/unintelligible coverage. Raw ASR is never the reference. |
| Human reliability and effort | Pre-adjudication agreement, boundary disagreement, adjudication fraction, corrected words per minute, correction real-time factor, language competence, training, and blinding. |

Exact metric code or package pins for token/grapheme error, macro-F1, boundary summaries, and cluster-aware uncertainty must be frozen and unit-tested before labels are received. `jiwer` and `krippendorff` were not installed at preflight; the implementation cannot be selected after agreement is visible.

## Instrument-to-learner firewall

Measurement-process provenance and scientific learner ancestry remain separate graphs. An accepted, fully human-dispositioned ChildLens record may later become a restricted ChildLens-derived scientific artifact only under an approved protocol. The following never enter learner or evaluation data: instrument code or weights, model checkpoints, tokenizers or vocabularies, prompts, token IDs, features, embeddings, hidden states, logits, scores, confidences, alternate hypotheses, VAD segments, diarization clusters, unreviewed text/timing/roles, or instrument-generated labels.

A future learner tokenizer must be initialized from scratch using only authorized, fully human-dispositioned ChildLens-local training text. Nothing in this report authorizes tokenizer construction, learner training, probing, causal-arm execution, or outcome inspection.

## Gate disposition and bounded next action

`G3` and `G4` remain `UNRESOLVED`, not `FAIL`: no admitted media or genuine human evidence exists. They may be judged only after exact-byte acquisition admission, local decode, and the blinded human block above.

The exact next measurement action is to complete exact-byte manifest admission, populate the fixed 15-item packet with local-only media windows, recruit qualified language-matched humans after initial language discovery, freeze metric implementations, and complete independent coding plus adjudication. Until those steps occur, no statement about ChildLens language, speech usability, non-child input, or transcript/role accuracy is scientifically defensible.

