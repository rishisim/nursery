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

## Frozen estimator

After the owner explicitly authorized read-only recovery from the connected
external volume, the exact V5 artifacts were found inside the encrypted
ChildLens bundle and matched every V5 public validation hash. The initial
missing-input receipt remains preserved in Git history and as an administrative
record of the disconnected-volume stop.

The sole repaired estimator is stable approximate balancing weighting:
separately within every participant, duration, and signed-lag arm, minimize
squared deviation from uniform weights subject to balancing the preregistered
nuisance means to the pooled seven-arm participant-duration target. This is the
minimum-variance approximate-balancing construction described by Zubizarreta
(2015, DOI 10.1080/01621459.2015.1023805). It controls the arm-level nuisance
distribution without requiring every individual row to satisfy all pairwise
calipers.

Continuous covariates are standardized within participant-duration. An
activity or location category remains exact only if every lag arm has at least
five rows; all other categories map deterministically to `__OTHER__`. No
outcome informed this rule. No additional trimming is allowed: a cell fails if
bounded weights cannot meet balance and support gates.

The nuisance-only table supported prospectively frozen gates of absolute SMD
at most 0.10, ESS at least 30 per arm, maximum weight at most 0.05, top-10
weight share at most 0.40, at least 40 raw rows per arm, all 18 participants,
all three durations, and all 378 participant-duration-lag arms. Any failure is
`NO_GO_UNINFORMATIVE`.

The estimator configuration, implementation, tests, and outcome-blind
feasibility receipt are frozen in Git before projection weights or repaired
alignment scores may be loaded.
