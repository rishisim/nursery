# ChildLens 30-video calibration extension v1.8

This additive protocol implements the user-selected one-time expansion from 15
to 30 participant-distinct ChildLens videos. The original selection, outputs,
and failed K=5 calibration disposition remain immutable.

The extension is intentionally narrow. It adds exactly 15 unused participant
groups using release metadata, verified official speech-presence annotations,
and deterministic hash ordering. It cannot inspect lexical or visual content,
model predictions, confidence, agreement, or causal outcomes during selection.
It cannot expand again.

The original 20-GiB raw-media limit remains the maximum simultaneous raw
footprint. Because the host cannot retain 15 additional full source videos
while preserving the 50-GiB free-space floor, each selected source is
transferred once, verified, converted to one deterministic bounded local
calibration clip, and removed before the next transfer. The extension's
cumulative transfer volume is recorded separately; retained derived clips are
limited to 4 GiB and remain in the owner-private quarantine.

The scientific gates do not change. The two visual paths remain Qwen3-VL and
Gemma 4 E4B; the speech paths remain Whisper large-v3-turbo and Qwen3-ASR with
the forced aligner. Model agreement is provisional triangulation, never human
truth. A combined calibration passes only if every requested dimension has two
model-specific estimates and a publishable conservative envelope after K=5
cluster suppression. Failure at 30 videos closes this route without a third
model, another expansion, or an empirical-free fallback.
