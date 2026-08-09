# ChildLens v1.1 local annotation-instrument preflight

Version: `childlens-instrument-preflight-v1.1.0`  
Audit date: 2026-07-21  
Scope: local structural preflight only; no Keeper/email/restricted-data access, model acquisition, weight execution, human judgment, or learner outcome  
Decision authority: none

## Finding

The host can already pin and run deterministic local audio extraction and can host a local-only annotation interface. It cannot yet run the proposed multilingual ASR/alignment stack: the exact packages and model snapshot are not installed or cached. The host does have a compatible Python 3.10 interpreter for a future isolated WhisperX environment.

Automated speaker-role inference is not an admissible base path. VTC remains fail-closed because neither its pinned code commit nor its pinned weight repository exposes explicit license metadata. Pyannote `speaker-diarization-community-1` remains fail-closed because downloading it requires a user to accept conditions and share contact information; this task has no authority to do that, and anonymous speaker clusters would not by themselves establish child versus other-person role. The base plan is therefore genuine human language identification, timing, source-language transcription, speaker-role coding, and visible-referent coding. Codex can construct and audit a blinded packet but cannot count as the human judge.

This is a readiness finding, not ChildLens pilot evidence. It does not pass G3, G4, or G6 and does not change the preserved v1 `REVISE` decision.

## Boundary and method

This workstream did not access Keeper, email, ChildLens media or metadata, AEA artifacts, or BabyView empirical artifacts. It did not download a model or dataset, load weights, accept conditions, generate transcripts, train a tokenizer or learner, or inspect an acquisition effect. Public software/model-card metadata was used only to check fixed versions, licenses, and access conditions. The v1 files were read but not edited.

The machine-readable receipt is [instrument_preflight.json](../../output/childlens_feasibility_v1_1/instrument_preflight.json). It records the full local inventory, executable hashes, sentinel results, packet schema, firewall, and source URLs.

## What is locally pin-able now

| Stage | Local status | Fixed disposition |
| --- | --- | --- |
| Language identification | Protocol available; no automated LID model installed | Two qualified humans assign an ISO 639 code, `MIXED_OR_CODE_SWITCHED`, or `UNDECIDABLE` before alignment. Any future Whisper LID output is advisory only. |
| Audio extraction | Ready and synthetic-sentinel tested | Homebrew FFmpeg `8.0.1_4`; `ffmpeg` SHA-256 `0a96…38e5`; deterministic first-audio-stream decode to mono 16 kHz `pcm_s16le`, metadata removed. |
| Multilingual ASR | Exact upstream pin known; package and model absent | `faster-whisper==1.2.1`, wheel SHA-256 `79a6…26a7`; `Systran/faster-whisper-large-v3` revision `edaa852…a478`. No use until separately acquired into quarantine after G1. |
| Alignment | Interface pin known; package and language route absent | `whisperx==3.8.5`, wheel SHA-256 `842f…50c3`; Python `3.10.20` is present and compatible. Pin the language-specific aligner only after genuine human language identification and before alignment, otherwise time manually. |
| Speaker role | Automated tools blocked | Do not acquire/run VTC or pyannote. Humans directly label `NON_CHILD`, `CHILD`, `OVERLAP`, `UNCERTAIN`, or `NONSPEECH`. |
| Referential status | Human protocol available; no visual model approved | Humans apply the frozen ±5-second window and six-state visible/null/ambiguity ontology. No detector or ASR output selects items. |
| Annotation UI | Runtime available; app not implemented or restricted-media tested | Streamlit `1.56.0` plus Python SQLite `3.53.3`. Bind only to loopback, disable telemetry/egress/file watching/static serving/uploads, and redact logs. |

The observed FFmpeg build has `--enable-gpl --enable-version3`, so the binary reports GPL v3-or-later. It may be used locally as an executable; do not redistribute it as part of a report bundle. This software-license observation is separate from, and cannot substitute for, ChildLens data-use permission.

### Audio sentinel

Two independent executions converted a locally generated 0.25-second 48 kHz stereo sine to 16 kHz mono signed 16-bit PCM. Both outputs had SHA-256 `dc12c788…f1495a`; `ffprobe` reported `pcm_s16le`, `s16`, 16,000 Hz, one channel. No restricted media was involved.

The later restricted command must keep the source immutable, explicitly select `0:a:0`, remove metadata, use bit-exact flags, suppress path-bearing logs, and write only inside quarantine. A technically decodable file still does not pass G3; human-correctable timing on the authorized sample remains necessary.

## Automated speech instruments: exact status

The v1 pins remain usable as acquisition specifications, not as claims of local readiness:

- `faster-whisper==1.2.1` has MIT package metadata and a fixed PyPI wheel digest. The pinned large-v3 conversion has MIT model-card metadata at immutable revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`. Neither package nor snapshot is local.
- `whisperx==3.8.5` has BSD-2-Clause package metadata, Python `>=3.10,<3.14`, and a fixed wheel digest. It is not installed. The default Python 3.14 interpreter is incompatible, but local Python 3.10.20 supports an isolated future environment.
- The actual ChildLens language is deliberately unknown here. A language-specific aligner cannot be finalized until humans establish the source language. That is a frozen routing rule, not permission to search for better alignment after inspecting lexical quality.
- Whisper’s own model card reports uneven language/accent/dialect performance and possible hallucinations. Therefore automatic LID and ASR cannot establish language or serve as gold. Translation remains prohibited.

The related local dependencies are also absent: CTranslate2, PyTorch, torchaudio, Silero VAD, and pyannote/audio packages. The standard caches contain none of the pinned large-v3, VTC, pyannote, or OpenAI Whisper checkpoints. Accordingly, the upstream identifiers are an acquisition specification, not a runnable environment or dependency lock.

No ASR or aligner can execute until the coordinator has separately passed the applicable ChildLens permission and immutable-release gates, then logs a hash-verified acquisition. Network access must be disabled during inference. If that installation never happens or the language route is unsupported, the valid fallback is manual timing and source-language transcription.

## Role and referential instruments

VTC code commit `9308411…27f` still returns no GitHub license object, and pinned `coml/VTC-2` revision `6b1a955…88a` remains public/ungated but has no license metadata. Ungated availability is not permission. Both code and weights therefore remain unacquired and unexecuted.

Pyannote’s primary model card identifies CC-BY-4.0 but requires the user to accept model conditions and share contact information. No such conditions were accepted. Even if permission were later granted, anonymous diarization clusters would need independent human mapping to the frozen roles and would remain only machine proposals. It is unnecessary for the v1.1 human-only path.

No pretrained visual detector, tracker, temporal proposal tool, or external class vocabulary is approved in this preflight. Referential feasibility uses the fixed content-blind sample, the ±5-second context, and human judgments of `VISIBLE_SINGLE`, `VISIBLE_MULTIPLE`, `NULL_NOT_VISIBLE`, `IRRELEVANT`, `UNDECIDABLE`, or `UNUSABLE`. Null, ambiguous, uncertain, and unusable cases cannot be dropped to improve coverage.

## Minimal local annotation interface

Streamlit 1.56.0 imports successfully and its installed package `METADATA`/`RECORD` hashes are in the JSON receipt. SQLite is available through Python. The later app must be implemented under the untracked restricted namespace, not `docs/` or `output/`, with these fail-closed settings:

- bind `server.address` and `browser.serverAddress` to `127.0.0.1`;
- set `server.headless=true`, `server.showEmailPrompt=false`, `server.fileWatcherType=none`, `server.enableStaticServing=false`;
- keep CORS and XSRF protection enabled and allow only the loopback origin;
- set `browser.gatherUsageStats=false` and block process network egress;
- provide no upload widget, cloud deployment hook, external component, map, font, analytics, or remote asset;
- read only the content-blind packet manifest from quarantine and keep SQLite, autosaves, media, transcripts, exact times, paths, and logs there; and
- sanitize errors and logs so report artifacts receive no media path, transcript, identifier, exact timestamp, or small cell.

`ffplay` is locally pinned as the minimal playback fallback. It may be paired with a restricted SQLite form if the Streamlit app cannot meet the loopback/egress sentinel. It is not a substitute for independent human labels.

## Genuine human-validation packet

The coordinator can populate the following packet only after permission, release binding, and hash-ordered pilot selection. Until then the specification is empty and contains no ChildLens payload.

1. **Language/timing discovery.** Opaque item keys and content-blind local windows are independently labeled by qualified humans for source language, mixed/undecidable status, utterance onset/offset, intelligibility, and overlap. No machine proposal is shown.
2. **Blinded reference.** At least the frozen 20% reference subset is transcribed and role-coded by two language-matched annotators without ASR/VTC proposals or peer labels. A separate production corrector may see licensed proposals outside this layer. Every accepted learner-visible utterance still receives a full human disposition.
3. **Referential layer.** Coders view the frozen ±5-second window and record visible object/action candidates, null, ambiguity, unusability, candidate count, and the nearest visible-candidate boundary. At least 20% is independently double-coded and stratified exactly as frozen.
4. **Adjudication and effort.** Independent records are locked before a qualified person adjudicates. The packet records training, language competence, blinding, corrections, unresolved coverage, adjudication fraction, and real-time correction effort.

The timing/role validation stop rule remains the first of 300 utterances or 30 speech minutes. All timing and role validation items are double-coded. Required evidence includes decodable coverage, the 500 ms-per-edge timing criterion, four-class role macro-F1, classwise errors, nonspeech false detections, language-appropriate post-correction token/grapheme error, the ASR-hidden anchoring-bias comparison, referential agreement, uncertainty, and effort. The human record, including exact timing and text, stays restricted; only terms-permitted cell-suppressed aggregates and digests may leave.

The `jiwer` and `krippendorff` packages are not installed. Before the packet receives labels, exact package pins or a reviewed local implementation for every scoring and cluster-aware uncertainty calculation must be frozen and unit tested. Metric implementation cannot be chosen after seeing agreement or lexical quality.

Codex can create the UI, validate blinding and assignment structure, compute metrics, and audit leakage. It cannot provide genuine auditory, language, role, or referential judgment. If qualified people are not available, the correct action is to pause with G3/G4/G6 unresolved.

## Instrument-to-learner firewall

The preflight keeps five distinct zones: external code/weight cache; raw machine-proposal quarantine; blinded human reference plus adjudicated ChildLens derivatives; a future ChildLens-only learner store; and nonidentifying report exports.

Only full human-dispositioned, terms-permitted ChildLens source-language text/timing/role/referential records may leave the proposal zone, and only the later frozen protocol can authorize accepted ChildLens RGB/text into a learner store. Package/model identities and settings may enter provenance receipts. No instrument weights, checkpoints, tokenizer, vocabulary, token IDs, prompts, features, embeddings, hidden states, logits, scores, confidences, alternate hypotheses, VAD segments, diarization clusters, detector outputs, or unreviewed text/timing/roles may enter learner data. No media, transcript, identifier, exact timestamp, path, or small cell may enter report outputs.

The later tokenizer must initialize from scratch using only authorized, fully human-dispositioned ChildLens-local text. The proposed amendment truthfully records external instruments as measurement-process ancestors while prohibiting them as scientific learner ancestors. An ASR-hidden reference subset is mandatory if ASR-assisted text is used; the human-only fallback gives the cleanest ancestry.

## Gate implications and next action

- G1: no finding here; this workstream did not inspect ChildLens permission evidence.
- G3: local decode preflight passes, but the empirical audio/timing gate remains unresolved until an authorized human pilot.
- G4: the human-only route is complete on paper; ASR/alignment are not local and automated roles are blocked. Language and accuracy remain unobserved.
- G6: the packet and ontology are fixed, but no genuine human referential annotation or reliability exists.
- G10: decode and UI runtime are available; instrument installation and language-matched human staffing remain.

The exact next instrument task, after the coordinator independently establishes G1 and binds/selects the release, is: create a hash-verified isolated Python 3.10 quarantine, acquire only the already pinned ASR/alignment artifacts if their software/model permissions pass, implement the loopback UI, and hand the populated blinded packet to qualified language-matched humans. VTC and pyannote remain excluded. No learner training follows from this preflight.

## Validation run

- Parsed the machine-readable receipt with both `jq` and Python’s JSON parser.
- Ran the two-pass synthetic FFmpeg conversion and format probe described above; both byte digests matched.
- Imported the pinned local Streamlit and SQLite runtimes without restricted input.
- Rechecked the four governing v1 artifact hashes against their preflight values; all were unchanged.
- Ran `scripts/validate_childlens_feasibility_v1.py`: pass.
- Ran `tests/test_childlens_feasibility_v1.py`: 17 tests passed.
- Scanned the two new artifacts for likely restricted filenames, participant/episode keys, email addresses, and media extensions; no restricted payload was found.

## Primary metadata sources

- [faster-whisper 1.2.1 PyPI metadata](https://pypi.org/project/faster-whisper/1.2.1/) and [pinned source commit](https://github.com/SYSTRAN/faster-whisper/tree/65882eee9f5cdbeeb2d877f1131d48cf241b327d)
- [WhisperX 3.8.5 PyPI metadata](https://pypi.org/project/whisperx/3.8.5/) and [pinned source commit](https://github.com/m-bain/whisperX/tree/4a6477e5e52ad516faab5363bdafbfa9840aec5e)
- [Pinned faster-whisper large-v3 snapshot](https://huggingface.co/Systran/faster-whisper-large-v3/tree/edaa852ec7e145841d8ffdb056a99866b5f0a478)
- [OpenAI Whisper model card](https://github.com/openai/whisper/blob/main/model-card.md)
- [Pinned VTC code](https://github.com/LAAC-LSCP/VTC/tree/9308411fb47f4290dfcd84338fc84bfb7f91227f) and [pinned VTC weight repository](https://huggingface.co/coml/VTC-2/tree/6b1a95508302edc14c50f670cd9a30d66fa4f88a)
- [Pyannote community-1 model card and access conditions](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [FFmpeg licensing and build-option guidance](https://ffmpeg.org/legal.html)
- [Streamlit configuration reference](https://docs.streamlit.io/develop/api-reference/configuration/config.toml) and [telemetry configuration](https://docs.streamlit.io/develop/concepts/configuration/options)
