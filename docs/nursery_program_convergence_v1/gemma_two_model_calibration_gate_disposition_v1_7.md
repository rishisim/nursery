# ChildLens two-model calibration gate disposition v1.7

## Terminal state

`GENUINE_USER_ONLY_BLOCKER`

The bounded local Gemma path completed, but the frozen two-model calibration did not pass. The sole failed gate is `ENVELOPE_NULL_OR_IRRELEVANT`. The no-automatic-fallback override therefore prohibits a Qwen-only calibration, a third visual model, a conservative-unbounded/full-domain substitution, an empirical-free automatic run, or any causal endpoint.

## Evidence

- The frozen sample remained unchanged: 15 participant-distinct items, 900 speech-seconds, and 137 candidate windows.
- Gemma and the read-only Qwen3 visual path both produced schema-valid aggregate diagnostics for all bounded windows. Both schema-validity intervals and item coverage are `[1.0, 1.0]`.
- Referential-status agreement is reported only as a model-model diagnostic and is outward-rounded to `[0.8, 0.8]`; it is not human reliability or ground truth.
- For `null_or_irrelevant`, both model-specific paths, their intersection, union, and conservative envelope are all `SUPPRESSED_K5`. The frozen gate requires a published conservative envelope, so it fails. No exact sub-K cell count is exported.
- Zero parser corrections were used, no semantic prompt tuning occurred, no third model was tried, and no sample was reselected.
- Restricted Gemma inference completed locally under network denial. Its hypotheses remain quarantined. Only the K=5-suppressed, outward-rounded aggregate receipt was exported.
- No pre-outcome receipt, one-shot authorization, causal output, learner outcome, or scientific endpoint exists.

The aggregate calibration receipt is bound by SHA-256 `2918e5094071e003931f79286945ad515ad41a2c5987eee3e98b14ea398f5b56`. The no-fallback override is bound by SHA-256 `cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7`.

## Bounded technical corrections attempted

1. Changed only the public TTS temporary transport container from AIFF to CAF while preserving the canonical WAV.
2. Relocated the rendered audio placeholder ahead of the five image placeholders without changing the declarative prompt or schema.
3. Removed redundant mandatory offline flags from the public canary subprocess extras.
4. Passed one immutable, parser-exact precomputed seal through the nested public canary checks.
5. Removed the same redundant flags from the restricted worker extras while retaining the two frozen thread controls and quarantine-local `TMPDIR`.
6. Restored the original frozen causal runner's planned aggregate-receipt path through a path-only compatibility erratum.

These corrections resolved transport and implementation failures. They did not change the final scientific gate result.

## User decision options

| Option | Cost | Main risk | Expected value |
|---|---|---|---|
| **A. Freeze one new content-blind ChildLens calibration extension** whose only purpose is to determine whether the K=5 null/irrelevant envelope becomes publishable; keep all model, schema, thresholds, and causal endpoints closed. | Moderate: additional deterministic selection, local preprocessing, and another comparable offline inference pass. | This is a post-failure sample extension and must be disclosed; it could still remain suppressed. | Highest chance of preserving the intended two-model ChildLens anchor without relaxing the failed gate. |
| **B. Authorize a new partially calibrated protocol** that treats null/irrelevant as unavailable and prespecifies how that missing dimension is handled before any endpoint. | Low to moderate. | Post hoc protocol relaxation weakens the calibration claim and must not be presented as the originally frozen test. | Could yield a small directional demo, but with a materially weaker naturalistic envelope. |
| **C. Authorize a separate empirical-free synthetic sensitivity study.** | Low. | Removes the ChildLens naturalistic anchor and is only a mechanism/engineering demonstration. | Fastest route to a toy nudge, but least aligned with the selected two-layer prototype. |
| **D. Stop this prototype lane and wait for a separately initialized BabyView-only study.** | High delay and governance cost. | Months of delay and uncertain access. | Best future naturalistic validation, but no immediate Michael-facing causal figure. |

## Recommendation

Choose **Option A** if a ChildLens-calibrated Michael-facing nudge remains the priority. It preserves the original scientific structure and changes only the amount of content-blind calibration evidence. Freeze a single deterministic extension size and a no-further-expansion stop rule before opening any new pseudo-label aggregate. If that resource cost is not worthwhile, choose **Option C** explicitly rather than weakening the current gate silently.

No branch is authorized until the user chooses.
