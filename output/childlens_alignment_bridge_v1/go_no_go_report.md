# ChildLens distribution-matched simulator bridge v1 — bounded preflight decision

**Decision: NO-GO. Stop before utterance–frame alignment scoring, locked evaluation, simulator fitting, or learner training.**

This is a measurement-pipeline stop. It is not human validation, ground truth, a causal null, or evidence that synchronized training-only side modalities cannot help. ChildLens remains a provisional developmental calibration for ages 3–5, never infant calibration.

## What was frozen

The protocol froze the participant-level split, measurement route, utterance/frame rules, shuffled and 15-second time-shifted controls, model-independent and model-based measurement vector, participant-clustered uncertainty, K≥5 reporting, pass/fail gates, and stop rules before the new development calculation.

The existing participant-distinct 30-recording sample was split deterministically into 8 development and 22 locked participants, balanced 4/4 across the historical acquisition cohorts in development. Cohort-specific development counts are not reported as results because each cell is below K=5. Locked outcomes were not analyzed.

Primary measurement route: pinned local Qwen3-ASR-1.7B plus Qwen3 ForcedAligner for German. Sole sensitivity: pinned local `whisper.cpp` large-v3-turbo DTW. Both are fixed measurement instruments, not acquisition learners. Their outputs are model hypotheses, not transcripts. The planned continuous alignment scorer retains German text and uses normalized multilingual OpenCLIP cosine through an ID-preserving EgoBabyVLM-compatible interface; it was not activated because the earlier gates failed.

## Development observations

Coverage was mixed. The sensitivity route was nonempty for all 8 development participants, and the primary route yielded 72 usable utterance hypotheses (94.7% of its proposed utterances). But primary item coverage was 87.5%, below the frozen 90% minimum.

Model–model stability failed by a wide margin:

- Primary German classification: 75.0% versus an 80% minimum.
- Primary/sensitivity language agreement: 75.0% versus an 80% minimum.
- Participant-median normalized character similarity: 0.126 versus 0.55; participant-bootstrap 90% interval 0.036–0.657, with a frozen lower-bound requirement of 0.40.
- Participant-median utterance-boundary F1: 0.058 versus 0.40; participant-bootstrap 90% interval 0.006–0.171, with a frozen lower-bound requirement of 0.25.

An initial derived receipt compared the literal labels `German` and `de` and was mechanically wrong. It is preserved but superseded. Version 1.0.1 normalizes only language-label aliases; it changes no split, text, timing, thresholds, or decision. Character and boundary gates fail independently.

## Why the stop is scientifically necessary

Real-versus-shuffled or time-shifted cosine lift would not be interpretable when the selected German text/timing route is this unstable across two fixed local instruments. Proceeding would turn ASR/timing idiosyncrasy into an unquantified part of the alignment effect. The frozen order therefore blocks development alignment scoring and all locked access.

The failed 30-video Qwen3-VL/Gemma branch remains unchanged: it stopped before causal testing because Gemma abstention was 80–90% and the privacy-safe null/irrelevant envelope was suppressed. This preflight did not reuse categorical referential pseudo-labels, relax those gates, or use Qwen/Gemma as the acquisition learner.

## External volume and expansion

The only mounted external physical volume is a 249.8 GB ExFAT device with about 131.9 GB free. Its recording payload is 40 VRS recordings totaling 114.16 GB, with AEA provenance—not the selected 30 ChildLens MP4 recordings. It also lacks owner enforcement, no-index controls, and evidence of a restricted encrypted container. It was excluded; no files were copied, modified, or broadly hashed.

The current ChildLens sample remains intact in the owner-private quarantine: 15 original full recordings and 15 bounded extension clips, all mode 0600 and participant-distinct. The restricted release manifest has 192 media rows; a historical content-blind receipt identified 121 unused media across 42 unused participants. No additional full ChildLens media are presently copied, and no expansion is justified after this gate failure.

## Recommendation

Do not proceed to full ChildLens lexical calibration or expand the sample now. The preferred disposition is to stop this lexical alignment bridge.

The bounded future options are:

1. retain only model-independent ChildLens distribution calibration under a separately scoped protocol;
2. obtain authorized qualified German validation and then freeze a new protocol;
3. preregister exactly one new German ASR/alignment route in a new protocol, with no post-hoc model shopping; or
4. stage a ChildLens-only expansion only if a new protocol establishes a specific identifiability or K-safe aggregation requirement.

Do not substitute synthetic-only evidence. BabyView remains a future confirmation study only.
