# ChildLens calibration recovery v5 — prospective protocol

## Scientific role and boundary

V5 is one bounded development-only attempt to test whether the same 18
ChildLens development participants can support a proper transcription-free
audiovisual calibration when duration, speech representation, and the temporal
estimand are corrected. ChildLens ages 3–5 remains a provisional developmental
calibration, never infant calibration. V1–v4 and their gates remain immutable.

This instantiates Michael Frank’s bridge prospectively: measure selected
natural temporal, speech-context, activity, audio-quality, visual-dynamics, and
audiovisual-lag distributions; later reproduce those measured distributions in
simulation while generating otherwise unavailable physical side streams; and
then apply identical instruments, preprocessing, learner, and training
construction to estimate simulated synchronized-side-cue lift. V5 itself stops
at a development calibration decision. It does not generate simulation, train
side-cue arms, or make a causal cue-lift claim.

## Outcome-blind Stage 0

An input may enter selection only if an outer receipt proves it contains
exactly the immutable 18 development participants and zero locked
participants. Mixed-scope manifests may not be opened or filtered internally.
From released availability alone, choose one common duration: prefer 15
minutes per participant, allow 10–20 minutes, round down to a minute, and stop
unless all 18 support at least 10 minutes and at least three total hours.

Representative windows are primary and are deterministically spread across
released speech-like timing and recording-time quantiles. The preregistered
enriched activities are book reading, object play, drawing/crafting, pretend
play, and making music. Prior outcome-blind v4 availability established that
timed activity subevents are not uniformly bound, so the frozen fallback is
not to oversample within recordings: report a conditional recording-level
coarse-activity subset when K-safe, and recover natural mixture weights from
all representative observation time.

Before new media decoding, freeze exact private source bindings and bounded
intervals, their hashes, three six-participant evaluation folds, windows,
controls, models, learner, seeds, uncertainty, privacy, gates, and stop rules.
No held-out audiovisual outcome can select any of them.

## Frozen single route

Raw 16 kHz speech passes through frozen
`facebook/wav2vec2-xls-r-300m` at revision
`1a640f32ac3e39899438a2931f9924c02f080a54`; no tokenizer, CTC head,
decoder, transcript, ASR, translation, or language identifier is loaded.
Frames pass through frozen `facebook/dinov2-small` at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`.

Each modality uses masked temporal mean-plus-standard-deviation pooling,
LayerNorm, a 256-unit GELU layer, 0.10 dropout, and a 128-dimensional
L2-normalized projection. The two projectors have 792,832 trainable parameters
in total. Training is symmetric InfoNCE with fixed AdamW settings, 80 epochs,
three seeds, and no model selection. Frontends never update. No alternative
architecture or model may be tried after ChildLens outcomes.

## Cross-validation and lag estimand

Three participant-disjoint folds hold out six and train on twelve participants
each; every participant is evaluated exactly once. Assignment is deterministic
and stratifies only immutable cohort and released coarse metadata. Inference
first averages within participant, retains fold estimates, and reports
cross-fold I² and the fold range.

Window durations are 2, 6, and 18 seconds. For each scale, the signed video
onset lags are −4×, −2×, −1×, 0, +1×, +2×, and +4× the window duration.
Nonzero controls may not overlap the real window or wrap. Matching uses fixed
duration, released speech support, activity/location, audio energy, motion,
and visual persistence. The primary contrast is zero lag minus the mean of
signed 2× and 4× lags. Signed 1× lags, asymmetry, curve amplitude, and a
participant-excluding nuisance-matched shuffle are secondary. A shuffled
advantage without a reproducible within-recording curve is not evidence.
Short-lag scene persistence is reported explicitly.

## Frozen gates and decisions

The synthetic embedding-level positive control must recover an injected shared
relationship with mean aligned-minus-lagged cosine at least 0.10 and a
participant-clustered 90% lower bound at least 0.05. It is an instrument check,
not ChildLens evidence. It trains the exact 792,832-parameter projector and
frozen optimizer schedule on 12 synthetic participants, then scores six
participant-disjoint synthetic participants against an inventory-preserving
cyclic lag.

Shared pass gates require: clean governance; all 18 participants with the
uniform duration and at least 40 matched rows per participant-scale; at least
98% embedding completion without substitution; interpretable lag and
persistence controls; primary 90% interval width at most 0.04; fold I² at most
50%; and fold range at most 0.03.

`PASS_DETECTABLE_STRUCTURE` additionally requires primary lift at least 0.02,
a 90% lower bound above zero, at least 12 positive participants, all three
positive fold contrasts, and each held-out fold curve correlating at least
0.50 with the curve pooled from the other folds.

`PASS_PRECISE_WEAK_OR_FLAT` instead requires the primary 90% interval wholly
inside ±0.02 and the 90% upper bound on curve amplitude at most 0.02. This
would calibrate to reproducibly weak or undetectable alignment; it would not
prove that true grounding is absent.

Every other result is `NO_GO_UNINFORMATIVE`. Gates may not be relaxed. No v6,
fallback model, locked confirmation, simulator, or side-cue arm starts
automatically.

## Privacy and acquisition

Only frozen bounded intervals may enter the existing owner-private ChildLens
quarantine. Transfer is one source object at a time; each transient full source
must be deleted and deletion verified before continuing, and temporary
read-only credentials must be deleted afterward. The external AEA volume is
inadmissible for both data and storage.

Public results use participant clusters, K≥5, and complementary suppression.
Paths, exact intervals, identifiers, embeddings, row scores, learned weights,
media, and credentials remain private and uncommitted.

## Stage 0 disposition for this run

This run stopped before selection freeze because the inventory process parsed
legacy mixed-scope manifests. Even though only schema/count information was
queried and no identifiers, media, intervals, or outcomes were exposed, zero
locked-row loading cannot be certified under v5’s literal rule. Therefore no
new media was decoded or acquired and no ChildLens embedding, training, or
outcome analysis ran. The terminal development decision is
`NO_GO_UNINFORMATIVE`.
