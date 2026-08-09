# ChildLens language, transcription, and speaker-role feasibility audit

Version: `childlens-transcription-feasibility-v1.0.0`  
Audit date: 2026-07-21  
Scope: governance and feasibility only; no media acquisition, transcription run, or learner training  
Gate status from this workstream: `G3_AUDIO_TIMING=UNRESOLVED`, `G4_TRANSCRIPT_ROLE=UNRESOLVED`

## Finding

A fixed, local, multilingual candidate workflow can be specified without assuming that ChildLens is English or German. The workflow uses a versioned ASR model only to propose source-language text, a language-specific aligner only after human language identification, and a child-centered voice-type instrument only to propose speaker roles. Humans must validate or correct language, utterance boundaries, text, and speaker role. **Unvalidated ASR is not gold.**

This is a design finding, not pilot evidence. The actual ChildLens language or languages remain `UNKNOWN_PENDING_RELEASE_OR_PERMITTED_PILOT`; no ChildLens audio or video was acquired or inspected, no ASR or diarization was executed, and no quality metric was produced. Gates G3 and G4 therefore cannot pass from this report.

The proposed voice-type instrument also has a permission caveat. The observed VTC 2.2 code repository and its public, ungated weight repository do not expose an explicit license in the GitHub license endpoint or Hugging Face model metadata. Public availability is not permission. VTC use must remain blocked until its code-and-weight permission is documented, or the role stage must use the human-only fallback described below. No license or model conditions were accepted in this audit.

## Evidence boundary

This workstream read the governing construction contract, ChildLens measurement definitions, readiness and parallel-design documents, the frozen feasibility rubric and pilot protocol, and the local EgoBabyVLM README/transcription interfaces at commit `224621caf0628270b6115845ac75a65b984234a3`.

EgoBabyVLM was consulted only as a public methodological/code-interface reference. No BabyView data, measurement, distribution, vocabulary, tokenizer, checkpoint, weight, or result was read or reused. No AEA artifact was read. The local checkout remained a reference and nothing was imported from it into a scientific data path.

The reference interface is not suitable unchanged:

- its checked-in transcription configuration defaults to English;
- when its language field is empty, the alignment loader is initialized with English before per-file language detection, which can misalign non-English speech;
- its VTC filter drops any ASR segment overlapping a key-child interval, while this audit must retain child, other-person, overlap, uncertain, and nonspeech cases for validation; and
- its manifest examples expose transcript text, media names, and timestamps, all of which are restricted here.

Only the general pattern—local ASR, forced alignment, voice-type proposals, and separate manifests—is retained.

## Fixed candidate pipeline

The word “fixed” means that code revisions, model revisions and hashes, decode settings, language routing, thresholds, and annotation rules are recorded before processing the authorized pilot. It does not mean that a currently unknown language is guessed in advance.

### 0. Preconditions and quarantine

Processing may start only after all of the following are true:

1. G1 terms explicitly permit local audio extraction, restricted derived transcripts/role annotations, human correction, retention, and the intended nonidentifying aggregate exports.
2. The exact Keeper release and participant/session grouping are pinned, and the content-blind 12–18-video pilot has been selected under `frozen_pilot_protocol_v1.json`.
3. Every code wheel, repository commit, model revision, configuration, threshold file, and model weight has a recorded SHA-256 or immutable upstream object identifier. Network access is then disabled for inference.
4. The restricted working root is outside `docs/` and `output/`, is not version controlled or synchronized to a third-party service, and uses pseudonymous internal keys. Logs suppress source filenames and transcript strings.
5. VTC permission is documented. If it is not, use the human-only speaker-role fallback; do not silently substitute a newly found model.

Raw media, extracted audio, ASR text, word/utterance times, VTC intervals, human transcripts, role labels, and model confidences all remain restricted derivatives. Report artifacts may contain only permitted, cell-suppressed aggregates and procedural receipts.

### 1. Deterministic audio decode

- Read only the predeclared pilot files.
- Record container, audio codec, channel count, sample rate, duration agreement, decode failures, discontinuities, and time-base behavior inside the restricted receipt.
- Decode to mono 16 kHz PCM WAV with a versioned `ffmpeg` binary and an explicit downmix rule. Keep the source immutable.
- Do not infer that a technically decodable track has usable speech. G3 still requires human timing evidence on speech-bearing windows.

### 2. Language identification before alignment

Language routing is human-led and is performed before transcript quality is judged:

1. Use the content-blind pilot and its predeclared speech/activity strata. Do not select windows from ASR words, apparent lexical richness, or visual content.
2. Qualified language annotators independently label sampled speech as an ISO 639 language code, `MIXED_OR_CODE_SWITCHED`, or `UNDECIDABLE`. Release documentation can establish a candidate language, but household location, annotator language, or model language-ID cannot.
3. Whisper `large-v3` language probabilities may be shown only as an advisory discrepancy flag. They cannot override the human label.
4. Freeze a route per utterance or contiguous language span before forced alignment. Never translate to English.
5. For a language without a validated aligner, or for mixed/undecidable spans, use manual utterance timing. Lack of word alignment is not fatal because the scientific measurements require validated utterance onset/offset; it does, however, increase labor and must be reflected in the resource projection.

No corpus-local tokenizer or vocabulary is created in this audit. In a later authorized task, the learner tokenizer must be initialized from scratch using only approved ChildLens human-corrected text.

### 3. ASR candidate text

Primary candidate instrument:

- `whisperx==3.8.5` for its alignment interface, PyPI wheel SHA-256 `842f9ef5c2c2d33ccb6e68f5d5b233ded9d77a7e89bba06a1d55a9704cef50c3`, upstream tag commit `4a6477e5e52ad516faab5363bdafbfa9840aec5e`;
- `faster-whisper==1.2.1`, PyPI wheel SHA-256 `79a66ad50688c0b794dd501dc340a736992a6342f7f95e5811be60b5224a26a7`, upstream tag commit `65882eee9f5cdbeeb2d877f1131d48cf241b327d`;
- `Systran/faster-whisper-large-v3` revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`, converted FP16 `model.bin` LFS SHA-256 `69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1`;
- local transcription only, `task=transcribe`, human-fixed source language, beam size 5, temperature 0, no initial prompt/hotwords, `condition_on_previous_text=false`, and no translation; and
- CPU `float32`, one process, at most four CPU threads as the conservative local default. Any `int8` or CUDA change is a protocol revision to freeze before content processing, not a runtime adaptation based on transcript quality.

The faster-whisper implementation includes a versioned Silero VAD interface. If used, its exact options are frozen before processing: threshold `0.5`, minimum speech duration `250 ms`, minimum silence duration `500 ms`, speech padding `250 ms`, and maximum speech duration `30 s`. These values are candidate engineering settings, not evidence that the resulting boundaries are scientifically usable. All final utterance boundaries are human checked. VAD-missed speech and hallucinated speech are explicit error classes.

The official Whisper model card says the model is multilingual but that accuracy varies substantially with language and data availability. The WhisperX paper motivates language-specific forced alignment for timing. Neither source validates this model on ChildLens or on child-worn recordings, so no public benchmark is used as a proxy for G3 or G4.

### 4. Conditional forced alignment

Use the language-specific alignment registry from WhisperX tag `v3.8.5`. The tag provides built-in routes for English, French, German, Spanish, and Italian and named Hugging Face routes for additional languages. The actual aligner repository revision, license, model-file hashes, character inventory, and normalization rule must be frozen **after human language identification but before any pilot alignment**.

- If the human-established language has a permitted supported route, align the source-language candidate transcript, then have humans correct words and utterance boundaries.
- If corrected text changes materially, realign the corrected text and require human acceptance of the final utterance onset/offset.
- If a word contains characters outside the aligner's dictionary, retain it with `WORD_TIME_UNAVAILABLE`; never delete it to improve apparent coverage.
- If the language is unsupported, code-switched, or aligner validation fails, do not select an aligner ad hoc after reading content. Route that item to manual utterance timing.

Word-level times are convenient diagnostics, not required truth. The final measurement unit is a human-accepted utterance interval with source-language text.

### 5. Speaker-role proposals

Preferred candidate, subject to permission resolution:

- VTC inference code version `2.2.0`, repository commit `9308411fb47f4290dfcd84338fc84bfb7f91227f`;
- model repository `coml/VTC-2` revision `6b1a95508302edc14c50f670cd9a30d66fa4f88a`;
- `model/best.ckpt` LFS SHA-256 `5c3446d037f9c6746cbe19bb77a77973354d3443556f9e2278e83cd4fe7db5a2` (reported size 1,099,159,455 bytes; not downloaded here);
- VTC `thresholds/f1.toml` SHA-256 `88e325a7a351f7484a11155e30644802ec2f3625636894b196af1bb2ff52ed3a`;
- 16 kHz mono input, `save_probs=false`, and raw role intervals retained only until adjudication; and
- isolated environment because VTC requires Python 3.13+, while WhisperX 3.8.5 requires Python below 3.14.

The code README calls this VTC 2.2 while the pinned weight-repository README calls the current repository VTC 2.1. Exact commits and the checkpoint hash, not the marketing version string, define the proposed artifact. This naming discrepancy and the missing explicit license must be resolved in the instrument receipt before use.

Provisional VTC-to-frozen-role mapping:

| VTC evidence within an utterance | Frozen role proposal |
| --- | --- |
| `KCHI` only | `CHILD` (camera-wearing child) |
| exactly one of `OCH`, `FEM`, or `MAL`, without `KCHI` | `NON_CHILD` (another person) |
| concurrent activation of two or more voice sources | `OVERLAP` |
| audible speech without a reliable source | `UNCERTAIN` |
| no speech after human review | `NONSPEECH` |

`OCH` must not be mislabeled as adult; it is another child. `NON_CHILD` here means “not the camera-wearing child,” matching the ChildLens input question, not “adult.” Internal adult/other-child evidence can be retained for QA only if terms permit, but the frozen exported role set is unchanged. Gender predictions are not a scientific endpoint.

Pyannote `speaker-diarization-community-1` is not in the base plan. Its model card requires a user to accept conditions and provide contact information, an action this audit is not authorized to take. It also returns anonymous speaker clusters, which still require human child/other role assignment. It may be reconsidered only through an explicit user-authorized protocol revision; no such revision is needed for the human-only fallback.

### 6. Human correction and restricted record

Every utterance that could become learner-visible in this bounded prototype is reviewed by a qualified human. Machine confidence cannot waive review.

The restricted final record contains an opaque utterance key, human-accepted onset/offset, source-language verbatim text, language/mixed tag, one frozen speaker role, transcription/timing validity, and adjudication status. Use explicit markers for unintelligible material and overlap; do not guess words. Store machine proposals and confidences separately from final labels and delete them under the retention policy after QA.

The final learner interface receives only approved text and RGB windows. It never receives ASR tokens, ASR/VTC/aligner embeddings, hidden states, logits, confidence scores, speaker embeddings, model-derived features, or instrument checkpoints.

## Conservative validation protocol

### Sampling and blindness

- Accumulate a stratified validation set until the first of 300 utterances or 30 speech minutes is reached, exactly as frozen in G4. Stratify by participant group, coarse speech/activity stratum, duration, and language route without using words or apparent grounding quality.
- All timing and speaker-role validation items are independently double coded, which is stricter than the rubric's minimum 20% and follows the frozen pilot protocol.
- On a predeclared 20% reliability subset, two language-matched reference annotators work from media without seeing ASR/VTC proposals. A separate production corrector edits the machine proposals. This allows corrected output to be compared with a reference that did not inherit machine suggestions.
- Annotators are blind to each other's labels and to all future learner outcomes. Disagreements are adjudicated without changing the frozen role set.

### Required metrics

Report cluster-aware point estimates and confidence intervals by resampling participant, then video/session where available. Never export item text, identifiers, or exact times.

1. **Technical audio coverage (G3):** decodable duration / sampled duration; pass requires at least 80%.
2. **Usable utterance timing (G3):** fraction of speech-bearing sampled windows with human-correctable utterance boundaries; pass requires at least 70%. A boundary is usable only when the production-corrected onset and offset are each within 500 ms of the adjudicated reference, are ordered, and preserve overlap status.
3. **Timing error:** median and 90th-percentile absolute onset and offset error, plus deletion/insertion rate. These are diagnostics in addition to the G3 usable-boundary fraction.
4. **Speaker-role quality (G4):** macro-F1 of production-corrected roles against the blinded adjudicated reference across `NON_CHILD`, `CHILD`, `OVERLAP`, and `UNCERTAIN`; `NONSPEECH` false detections are reported separately. The frozen pass threshold is at least 0.80. Also report each role's precision/recall, coverage, and raw-instrument macro-F1 so correction effort is visible.
5. **Transcript quality:** source-language lexical token error after production correction against the blinded adjudicated reference for valid `NON_CHILD` utterances, with insertions, deletions, substitutions, unintelligible-token coverage, and a participant-clustered interval. Freeze a language-appropriate orthographic tokenization rule before scoring; for languages without whitespace boundaries, require language-expert word segmentation and also report grapheme error. No external vocabulary supplies the tokens.
6. **Human reliability and effort:** pre-adjudication Krippendorff alpha or equivalent for roles, boundary disagreement, corrected words per minute, adjudication fraction, and correction real-time factor. A low raw-ASR score does not by itself fail if complete, bounded human correction meets G3/G4; unbounded or unavailable language-matched labor does.

The rubric says post-correction lexical token error must be “bounded” but does not freeze a numeric ceiling. Before the pilot begins, the coordinator must add a numeric maximum and uncertainty rule or explicitly define an all-utterance adjudication policy. This workstream does not invent a post-content threshold.

### Error taxonomy

The restricted audit must count, with small-cell suppression on export:

- absent/corrupt/undecodable audio;
- VAD deletion, VAD false alarm, merged utterances, split utterances, and timestamp drift;
- ASR substitution, insertion, deletion, hallucination, non-speech transcription, named-entity uncertainty, and unintelligible speech;
- language misroute, dialect mismatch, code-switching, and unsupported script/aligner characters;
- key-child/other-person confusion, other-child/adult confusion, overlap missed, false overlap, distant/overheard speech, and media playback speech; and
- annotator disagreement and unresolved items.

Uncertain, overlapping, and unusable items remain counted. They are not silently removed to improve accuracy or visible-referent rates.

## Fallback and revision ladder

The following corrections preserve the scientific question and must be chosen without looking at a learner outcome:

1. **Alignment unavailable but language expertise available:** retain large-v3 as a typing aid, manually transcribe and time every learner-visible utterance, and report the extra labor.
2. **VTC permission absent or role quality inadequate:** omit VTC and have language-matched annotators assign the five frozen roles directly, using the release's coarse `child talking` / `other person talking` / `overheard speech` labels only as nonlexical context when the protocol permits. Double code all validation items.
3. **ASR quality poor but speech is intelligible:** use ASR only as a blank/editing interface or remove it; manual verbatim transcription remains possible if the labor projection passes.
4. **Qualified annotators unavailable, language cannot be identified, speech is unintelligible, or bounded correction cannot meet G3/G4:** G3/G4 cannot pass. Machine-only output cannot substitute.

Changing ASR model, adding prompts/hotwords, translating, searching for a better aligner after seeing transcript content, or tuning VTC thresholds on ChildLens lexical richness is not an allowed convenience adaptation. Such changes require a documented pre-outcome protocol revision and independent validation.

## Annotation-instrument ancestry firewall

The construction contract currently prohibits external pretrained models from the ancestry of scientific artifacts. A separate amendment is therefore required before these instruments can be used. The defensible amendment should distinguish **measurement proposals** from **learner ancestors**:

- named pretrained ASR, VAD, alignment, and voice-type models may run only in a restricted measurement namespace under fixed versions and licenses;
- their proposals must be human-reviewed, and only the human-accepted source-language transcript, utterance interval, and frozen role label may cross into the ChildLens scientific annotation namespace;
- no pretrained instrument feature, embedding, hidden state, token ID, tokenizer, logit, confidence, checkpoint, or pseudolabel accepted without human review may enter tokenizer training, simulator calibration, learner training, evaluation, model selection, or claim interpretation;
- the scientific tokenizer and text/vision learner still initialize from scratch inside this exact ChildLens release instance; and
- provenance must preserve the instrument receipt and human-review receipt while keeping restricted payloads out of reports.

For the bounded prototype, requiring human review of every learner-visible utterance is the cleanest way to prevent an external ASR vocabulary or VTC decision from becoming an unexamined scientific ancestor. A later scale-up using unchecked ASR would need a new amendment and cannot inherit this feasibility finding automatically.

## Gate implications

| Gate | Evidence from this workstream | Status |
| --- | --- | --- |
| G1 terms | Required derivative types and instrument conditions are enumerated; release permission not established here | `UNRESOLVED` |
| G3 audio/timing | Decode and human timing protocol is fixed; no authorized pilot was run | `UNRESOLVED` |
| G4 transcript/role | Multilingual route, candidate instruments, validation, and fallback are specified; actual language, VTC permission, and empirical accuracy are absent | `UNRESOLVED` |
| G5 input lexicon | Corrected non-child transcript could support the audit; no lexical content or recurrence was measured | `UNRESOLVED` |
| G10 resources | Local CPU/MPS-compatible paths exist, but measured duration/correction rate and language-matched staffing are absent | `UNRESOLVED` |

This report intentionally makes no terminal ChildLens feasibility decision.

## Primary sources

- OpenAI, [Whisper repository and official model card](https://github.com/openai/whisper/blob/main/model-card.md). The card documents multilingual scope and language-dependent limitations.
- Bain et al., [WhisperX: Time-Accurate Speech Transcription of Long-Form Audio](https://arxiv.org/abs/2303.00747) and the [WhisperX repository](https://github.com/m-bain/whisperX/tree/4a6477e5e52ad516faab5363bdafbfa9840aec5e).
- SYSTRAN, [faster-whisper v1.2.1 implementation](https://github.com/SYSTRAN/faster-whisper/tree/65882eee9f5cdbeeb2d877f1131d48cf241b327d) and [large-v3 conversion card](https://huggingface.co/Systran/faster-whisper-large-v3/tree/edaa852ec7e145841d8ffdb056a99866b5f0a478).
- LAAC-LSCP, [VTC 2.2 inference repository](https://github.com/LAAC-LSCP/VTC/tree/9308411fb47f4290dfcd84338fc84bfb7f91227f) and [pinned public weight repository](https://huggingface.co/coml/VTC-2/tree/6b1a95508302edc14c50f670cd9a30d66fa4f88a). The [GitHub license endpoint](https://api.github.com/repos/LAAC-LSCP/VTC/license?ref=9308411fb47f4290dfcd84338fc84bfb7f91227f) returns no license file and the [Hugging Face model API](https://huggingface.co/api/models/coml/VTC-2) exposes no license metadata as observed on the audit date.
- Lavechin et al., [An Open-Source Voice Type Classifier for Child-Centered Daylong Recordings](https://www.isca-archive.org/interspeech_2020/lavechin20_interspeech.html). The paper motivates child-centered voice types and multi-label overlap but does not validate the later VTC artifact on ChildLens.
- Pyannote, [`speaker-diarization-community-1` model card](https://huggingface.co/pyannote/speaker-diarization-community-1), consulted only to document its access conditions and why it is excluded from the base plan.
