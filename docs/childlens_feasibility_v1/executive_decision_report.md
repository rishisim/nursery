# ChildLens governance and lexical-grounding feasibility audit v1

Terminal decision: `CHILDLENS_FEASIBILITY_REVISE`

## Decision

The exact ChildLens view currently accessible in Keeper is a plausible sole empirical anchor for the proposed naturalistic weak-alignment prototype, but it is not ready for protocol freeze or learner training. The bounded revision is to establish release-specific processing terms, pin the live release, and run the already frozen metadata-stratified feasibility pilot. This decision does not say that synchronized embodied cues help grounding, and no acquisition effect was trained, estimated, or inspected.

The positive basis is real but limited: the live library shares the public dataset identity; it exposes 192 MP4 objects, 192 final annotation objects, 192 platform-format annotation objects, participant and recording-date grouping fields, synchronized naturalistic video/audio as the documented raw modalities, coarse non-child speech contexts, and object/action activity contexts. These are enough to make a bounded lexical-grounding feasibility test plausible.

The current evidence cannot support launch. The live view has no exposed immutable revision and differs from the paper's 354-file corpus snapshot; its license field is blank; language is unknown; audio decode and utterance timing were not tested; no timed lexical transcript or dedicated speaker-role labels exist; lexical recurrence, visible referents, null/ambiguity rates, and leakage-resistant held-out support were not measured. The predeclared pilot was correctly withheld because local copying and derived processing have not been shown to be permitted.

## Release reconciliation

The public paper reports 354 video files, 108.58 hours, 62 children, and 192 annotated videos. The Keeper archive description rounds the corpus to 109 hours and the annotated portion to 54 hours. The current read-only UI contains 192 MP4 objects and two 192-object JSON annotation representations. The live-video count therefore matches the paper's annotated-video count, not its full-corpus count; an annotated-subset release is plausible but unverified until a canonical manifest links media, annotations, and participant/date groups. The earlier approximate 343-video observation is contradicted by the current UI and is retired.

The 191 videos with displayed sizes sum to about 201.33 decimal GB, while one video has no displayed size. That is a lower bound, not a complete byte manifest. The full library does not fit the host safely and must not be downloaded.

## Gate readout

| Gate | Status | Why |
| --- | --- | --- |
| Terms | UNRESOLVED | No release-specific grant was found for local copies, restricted derivatives, model processing, retention, or aggregate export. |
| Exact release and grouping | CONDITIONAL | Public/live library identity and grouping fields match, but the live revision and complete linkage are unpinned. |
| Audio and speech timing | UNRESOLVED | Audio is documented; decode coverage and utterance boundaries were not piloted. |
| Transcript and speaker role | UNRESOLVED | Language is unknown and no validated timed lexical or child/non-child role pipeline exists. |
| Non-child input and lexical recurrence | UNRESOLVED | Speech-context labels exist, but corrected transcript volume and recurrence are unmeasured. |
| Visible referents, nulls, ambiguity, lag | UNRESOLVED | The ontology is frozen; no authorized double-coded pilot was run. |
| Held-out participant/session evaluation | UNRESOLVED | Split keys exist; linkage and noun/action support after splitting remain unmeasured. |
| Strong ceiling and weak baseline | PASS | Both can later be constructed from one immutable RGB/text inventory without changing empirical exposure. |
| Simulator calibration | CONDITIONAL | Video/audio aggregates may be usable after permission; all physical side streams remain unavailable and simulator-defined. |
| Storage, labor, and compute | CONDITIONAL | Selective low/base plans fit; exact bytes, language-matched staffing, and training budget still need measurement. |

## Bounded revision package

1. The user reviews—not this audit accepts—a current DUA/access instrument explicitly covering selective local copying, local decoding, restricted transcripts and annotations, fixed local measurement instruments, learner/checkpoint treatment, retention/deletion, and nonidentifying aggregate export.
2. Bind the study to the DOI snapshot or an immutable live revision. Build a restricted canonical manifest and export only its digest and permitted aggregate reconciliation.
3. Before any further authenticated inspection, implement and sentinel-test the page-local allowlisted/redacted metadata extractor required by the privacy audit. An initial broad browser response and a malformed schema extraction transiently exposed restricted metadata to tool output; no such value was saved in the report namespace or sent externally, but zero tool-output leakage cannot be claimed.
4. If terms permit, run the frozen 12-18-video, participant-distinct, metadata-stratified pilot. Human-establish language; validate audio, utterance timing, full transcript correction, and five-way speaker role. Use human-only role annotation unless a voice-type instrument's permission and version are resolved.
5. Apply the frozen referential protocol, retain null/ambiguous/unusable cases, test reliability and visible-candidate coverage, audit lexical recurrence, and construct participant/date-disjoint support tables before selecting noun/object or verb/action endpoints.
6. Only if every essential gate then passes may a separate task freeze the scientific protocol. If a fundamental gate fails, do not mix corpora; wait for a separately initialized study rather than borrowing empirical values.

## Model and causal-plan disposition

Retain the compact temporal CLIP+ direction: a random/scratch vision tower, a fresh ChildLens-local tokenizer and scratch BERT-like text tower, InfoNCE plus matched MLM and DINO-style self-supervision trained from scratch. No off-the-shelf learner weights are allowed. LLaVA remains deferred because the audit found no evidence that its extra complexity is necessary for the bounded prototype.

Fixed pretrained ASR, alignment, voice-type, temporal-proposal, or vision systems may be used only if the proposed annotation-instrument amendment is adopted: versioned local measurement quarantine, human disposition and reliability, a separate instrument-influence graph, and no features, embeddings, weights, tokenizers, vocabularies, scores, or raw hypotheses passed to the learner. ChildLens has no observed IMU, touch/contact, proprioception, or motor stream; those remain simulator-defined counterfactual modalities and must never be described as ChildLens-measured.

The future causal design remains five paired arms with side cues absent at evaluation. This audit authorizes neither those runs nor any outcome-driven protocol change.

## Privacy, provenance, and scope

No ChildLens media, audio, frame, transcript, learner, tokenizer, checkpoint, or scientific outcome was written to this namespace. No other child corpus or adult corpus empirical artifact contributed data, measurements, priors, vocabulary, weights, or results. Public papers and code interfaces were used only as methods or release documentation. Nothing was uploaded, emailed, committed, pushed, published, or used to change a Keeper share; no license was accepted.

The browser-response incident prevents the stronger claim that restricted values never appeared in tool output. The on-disk artifacts are therefore validated separately, and further authenticated inspection is fail-closed until the redaction extractor is repaired and tested.
