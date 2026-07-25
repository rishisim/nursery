# ChildLens audiovisual calibration support repair — fail-closed protocol

## Scope

This is the single authorized support-repair attempt after the terminal V5
`NO_GO_UNINFORMATIVE` decision. It may reuse only the existing 18 development
participants, 4.5-hour bounded sample, frozen folds, windows, lag grid,
frontends, projector, seeds, nuisance variables, scores, and calibration
records. All 22 locked participants remain sealed. No acquisition, model
change, retraining, transcript-like processing, simulator work, or side-cue
condition is permitted.

V1–V5 records and decisions are immutable. The repair may replace only V5's
all-or-nothing row-level nuisance calipers with one outcome-blind,
distribution-level balancing estimator. Estimator choice, category coarsening,
common-support rules, balance targets, effective-sample-size and weight
concentration gates, participant and lag coverage, and failure rules must be
frozen from a nuisance-only feasibility table before any alignment score is
read.

## Required input binding

Feasibility requires a private, owner-only, development-only V5 feature
manifest and hash-bound feature arrays containing the original request/row
bindings and preregistered nuisance covariates. A restricted score result or
projection weights may be used only after feasibility and the estimator freeze.
Deterministic regeneration is allowed only after its cost is reported and only
from retained bounded clips with the exact V5 models and configuration.

If these inputs are absent or cannot be verified, the attempt stops before
scoring. Public V5 aggregate outcomes may document the prior failure but may
not select the estimator or its thresholds.

## Frozen analysis sequence

1. Verify the development-only scope receipt, private permissions, V5 hashes,
   and zero locked/excluded-corpus access.
2. Read identifiers and nuisance covariates only. Produce a privacy-safe
   feasibility table for every participant, duration, and signed lag.
3. If defensible common support exists, prospectively freeze exactly one
   standard balancing estimator and every diagnostic/failure threshold.
4. Hash the protocol, configuration, feasibility receipt, feature manifest,
   and arrays before allowing score access.
5. Apply the frozen estimator exactly once. Preserve participant-level
   contrasts, three-fold reporting, V5 effect/equivalence, positive-control,
   precision, heterogeneity, and shortcut-interpretability gates.

## Terminal states

The only outcomes are `PASS_DETECTABLE_STRUCTURE`,
`PASS_PRECISE_WEAK_OR_FLAT`, and `NO_GO_UNINFORMATIVE`. Either pass may only
recommend separately authorized locked confirmation. It cannot unlock data or
start simulation. Missing or unverifiable required inputs yields
`NO_GO_UNINFORMATIVE`.

## Disposition

The V5 private feature manifest, feature arrays/shards, and restricted result
were not discoverable, and the original attested-runtime discovery failed
closed. The empty owner-private quarantine directory is insufficient to bind
the prior 18-participant development run. No nuisance-only feasibility table
can therefore be constructed, and no estimator or data-dependent thresholds
can be defensibly frozen.

This attempt stops before score access, estimator application, regeneration,
locked inspection, or simulator work with terminal state
`NO_GO_UNINFORMATIVE`.
