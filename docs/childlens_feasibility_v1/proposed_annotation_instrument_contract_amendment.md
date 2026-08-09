# Proposed ChildLens annotation-instrument contract amendment

Version: `childlens-annotation-instrument-amendment-v1.0.0`  
Status: **PROPOSED — not effective until adopted with the later protocol freeze**  
Scope: ChildLens feasibility measurement and a possible later ChildLens-only prototype  
Scientific learner outcome authorized: **no**

## Purpose

This amendment resolves one narrow conflict in the construction contract. The current contract permits external pretrained models only as non-scientific diagnostics that can never be ancestors of scientific artifacts. ChildLens has no established timed lexical transcript, speaker-role transcript, or word–referent annotation in the evidence reviewed for this audit. Fixed pretrained speech, diarization/voice-type classification (VTC), temporal-proposal, or vision tools may therefore be useful as **annotation instruments**, but they must not become learner components or silently supply a vocabulary, representation, target, or empirical prior. Here `VTC` means **voice-type classifier**, not video–text correspondence.

The amendment changes the treatment of measurement instruments only. It does not relax the one-corpus rule, scratch-learner rule, privacy rules, outcome prohibition, or requirement to confirm release terms before processing.

## Controlling rule

For a selected, version-pinned ChildLens release, a fixed pretrained model may be used only to propose restricted intermediate annotations inside an isolated measurement quarantine. Every learner-visible or calibration-eligible derivative must be grounded in ChildLens source media, receive a recorded human disposition under a frozen protocol, and pass the applicable reliability gate. Raw instrument output is never gold. Instrument weights, hidden states, embeddings, logits, confidence scores, token IDs, tokenizer vocabulary, generated labels, prompt text, and unadjudicated transcript text are prohibited from every scientific learner input, tokenizer-training input, model checkpoint, causal-arm payload, and evaluation payload.

The exception is a measurement-process exception, not a learner-ancestry exception.

## Two distinct provenance graphs

The implementation must maintain two graphs rather than overloading one use of “ancestor.”

1. **Measurement-process provenance** records that a fixed external instrument proposed an annotation. It includes the instrument receipt, settings, raw-output quarantine, human review, reliability evidence, and final disposition. Omitting this influence would be misleading.
2. **Scientific artifact ancestry** begins with the pinned ChildLens release and accepted human-adjudicated ChildLens derivatives. It may continue through a fresh ChildLens-local tokenizer and scratch-initialized learner. External instrument representations, weights, label inventories, tokenizers, scores, and raw hypotheses have no edge into this graph.

An accepted human transcript may be bootstrapped by an ASR hypothesis only when a human verifies or corrects the full learner-visible record against the source audio and assigns an explicit disposition. An independently transcribed, ASR-hidden subset must estimate instrument-assisted transcription bias. The accepted record is treated as a restricted human-adjudicated ChildLens derivative; the original ASR hypothesis remains quarantined and is not copied into the learner store.

## Permitted instrument roles

| Instrument class | Permitted role in quarantine | Required human disposition | Prohibited transfer |
| --- | --- | --- | --- |
| ASR / speech segmentation | Propose language routing, utterance boundaries, and transcript hypotheses | Verify or correct the entire accepted utterance against audio; mark unusable/uncertain rather than inventing text | Raw hypotheses; unverified words; model tokenizer; alternate candidates; token probabilities; embeddings or hidden states |
| Speaker diarization / voice activity / voice-type classifier (VTC) | Propose speech regions, anonymous local speaker clusters, or role hypotheses | Assign only the frozen roles `NON_CHILD`, `CHILD`, `OVERLAP`, `UNCERTAIN`, or `NONSPEECH`; validate role accuracy | Voiceprints; speaker embeddings; cluster IDs; confidence scores; identity inference; unreviewed role hypotheses |
| Temporal alignment proposal tool, if separately justified | Rank windows for annotation efficiency after content-blind pilot selection | Human adjudicates timing, visible-candidate status, null status, ambiguity, and unusability | Similarity scores; learned alignments; selected-only training examples; representations or pseudo-targets |
| Vision detector / tracker | Propose boxes, masks, tracks, motion, or candidate object/action regions for measurement | Human checks the frozen measurement on a reliability sample and adjudicates any scientific label | Detector class vocabulary; features; logits; tracks/boxes as learner channels; detector-selected evaluation items |

Instrument outputs may help schedule blinded human work only after the pilot sample is fixed from metadata. They may not select episodes because they look lexically rich, visually salient, or easy to ground.

## Quarantine and one-way release

The annotation workspace must be restricted and release-specific. Raw ChildLens media, raw and corrected transcript text, instrument outputs, frames, identifiers, exact timestamps, local speaker clusters, and row-level annotations stay there. Report namespaces receive only terms-permitted, nonidentifying aggregates with coverage, uncertainty, and small-cell suppression.

Release from quarantine is field-whitelisted and one-way:

- A later learner store may receive raw ChildLens vision windows and fully human-dispositioned corpus-local text/timing records only if the data-use terms and frozen scientific protocol permit them.
- Timing is used internally for window construction but is never exported as an exact timestamp.
- Referential and event annotations may define human-adjudicated alignment transformations or evaluation labels, but hidden targets remain physically separated from model-visible records.
- Instrument artifact IDs may appear in provenance receipts, never in example records presented to the learner.
- No instrument cache is reused across corpus instances. No annotation instrument is fine-tuned on ChildLens during this feasibility audit.
- Quarantined intermediates are deleted according to a documented retention rule only after accepted derivatives, receipts, and audit checks are complete; reports never contain recoverable text or media fragments.

## Human validation and reliability gates

Before any instrument-assisted derivative is used for calibration or a later learner:

1. Identify the actual language or languages from release-specific evidence or a permitted pilot. Freeze language routing and instrument versions before judging quality.
2. Validate utterance timing and speaker role on at least 300 utterances or 30 speech minutes, whichever comes first. All timing/speaker validation items are independently double-coded before adjudication.
3. Require non-child versus child/uncertain speaker-role macro-F1 of at least 0.80. Report overlap, nonspeech, and uncertain errors separately; do not convert uncertainty into non-child input.
4. Measure transcript error against human-adjudicated text, stratified by language, role, activity/speech stratum, and acoustic condition when reportable. Never treat unvalidated ASR as ground truth. Preserve a blinded, ASR-hidden transcription subset to bound anchoring bias.
5. Independently double-code at least 20% of referential items, stratified by participant group and speech/activity stratum. Require Krippendorff alpha or a prespecified equivalent of at least 0.67 for the frozen referential status ontology, with bounded adjudication.
6. Report coverage and uncertainty, including unusable, undecidable, overlap, and null cases. Reliability failures cannot be repaired by selecting easier content or inspecting a learner effect.

The frozen feasibility rubric remains controlling. A failed threshold produces the rubric-defined `CONDITIONAL`, `UNRESOLVED`, or `FAIL` disposition; it does not authorize model or sample tuning.

## Minimum receipt for every instrument

Each fixed instrument must have a machine-readable receipt containing:

- receipt ID, purpose, instrument class, and quarantine namespace;
- software package, source repository, immutable revision, model card, model/weight artifact identity and cryptographic digest;
- license and permitted-use review for both code and weights, with the reviewer/date and unresolved clauses; an instrument such as VTC whose code or weight permission is not explicit remains `NOT_ACQUIRED`/fail-closed and cannot execute;
- acquisition source/date, local cache identity, execution environment, device, dependency lock or container digest;
- exact inference settings, prompts if any, language-routing rule, random seeds, determinism limitations, and failure handling;
- input field allowlist and output field inventory;
- proof that features, embeddings, logits, scores, weights, token IDs, class vocabulary, raw hypotheses, and speaker clusters are absent from released learner/evaluation artifacts;
- human annotation protocol version, annotator training, blinding, double-code sample construction, adjudication rule, reliability results, coverage, and error taxonomy;
- restricted retention/deletion status and aggregate export review; and
- hashes of the frozen measurement protocol and accepted derivative manifest, without restricted paths, filenames, IDs, or exact timestamps in reportable receipts.

A missing or mutable receipt fails closed.

## Learner boundary after amendment

The proposed learner remains a compact temporal CLIP+-style architecture with a random/scratch vision tower, a fresh ChildLens-local tokenizer and scratch BERT-like text tower, and scratch self-supervised/contrastive objectives. It receives no off-the-shelf learner weights. The tokenizer is trained only on fully human-dispositioned ChildLens-local text authorized by the release terms. No ASR tokenizer, external vocabulary, detector label list, public benchmark text, or other corpus contributes tokens or initialization.

Simulator-only IMU, contact/touch, proprioception, and motor streams remain counterfactual modalities. An annotation instrument cannot make an absent physical stream “measured,” and no instrument may estimate one as if it came from ChildLens.

## Required contract edits if adopted

If this proposal is adopted, replace only the current sentence that limits external pretrained models to non-scientific diagnostics with the controlling rule above, and add the two-graph provenance requirement. All other clauses of `child-only-scientific-contract-v1.0.0` remain in force. Adoption must be versioned, reviewed before any bulk or derived processing, and included in the later protocol digest. This feasibility task does not itself adopt the amendment or authorize acquisition, annotation, or learner training.
