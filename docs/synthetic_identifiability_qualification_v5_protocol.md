# Identifiability-safe synthetic benchmark qualification v5

This protocol qualifies a package only. It generates no development or confirmation outcome and cannot authorize a scientific GO.

V5 uses fresh, mutually disjoint fixture, excluded-smoke, future-development, and confirmation registries. Every identifier found in the v1-v4 registry sources is blocked for all operations. The excluded smoke may run once after freeze; failure is retained.

The benchmark crosses target presence, candidate count, action visibility, speech/action lag, repetition, sensor informativeness, dropout, noise, and false-positive activity. It qualifies synchronized evidence, an exact matched randomized shuffle, preregistered shifts of -8 and +8 samples, a deliberately uninformative stream, an absent sensor channel, and a corrupted-sensor control. Action and noun episodes share the factor schedule. Noun sensor weight is exactly zero, while sensor ownership and noun target identity/position are factorially independent.

The runtime must instantiate the actual frozen v2 `SensorEventDetector` and call its `candidate_evidence`. Detector top-event-plus-null accuracy must satisfy predeclared overall and stratum bounds: it must be informative in the informative stratum, imperfect overall, non-oracular in zero-information and corrupted strata, and not chance everywhere.

The learner is a joint iterative cross-situational model. Candidate responsibilities are shared across component words; vocabulary-column exclusivity changes the numeric update. Sensor logits enter the same fitting rule in every condition and never enter noun updates. No learned parameter is projected, canonicalized, answer-key aligned, or manually replaced after fit.

Evaluation is language-only. Prompts contain tokens and candidate visual observations, never raw sensor data, owners, targets, or keys. Exact ties receive fractional top credit and softmax probabilities are scored with log loss and multiclass Brier score. The same rule applies to primitive, manner, compositional action, noun, and null prompts. Candidate reordering with only sealed key-index updates must leave every metric and decision byte-identical. Global-label and best-global-permutation diagnostics are reported separately.

All visible condition inputs, permitted derived evidence, language prompts, and sealed keys are persisted separately. A new Python process launched from the frozen source snapshot verifies its own source and input manifests, reloads only those persisted inputs/config/evidence/keys, and recreates deterministic results in another directory. Mutation of either one input byte or one frozen source byte must fail verification.

The official test report uses the frozen repository interpreter and exactly two commands, v5 tests first and then `tests/`. It is the only test report admissible to the adjudicator. Traceability resolves real AST symbols, actually collected test node IDs, artifact paths, and gate fields. The final decision is regenerated in a clean staging directory from frozen official artifacts and must be byte-identical.

`V5_READY` requires every frozen gate. `REVISE` is used for a repairable package gate. `STOP` is used when the benchmark is structurally unidentifiable or requires oracle leakage or manufactured scoring. None authorizes a future-development or confirmation outcome.

