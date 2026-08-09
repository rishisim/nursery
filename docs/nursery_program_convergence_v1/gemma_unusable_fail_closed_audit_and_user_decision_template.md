# Gemma-unusable fail-closed audit and user-decision template

Date: 2026-07-22

Status: `PUBLIC_NONEMPIRICAL_AUDIT_NO_BRANCH_SELECTED`

Scope: public contracts, aggregate-safe receipts, code structure, and synthetic tests only

## Result

The revised CAF/Gemma chain fails closed before restricted calibration and
before every scientific endpoint if the exact Gemma path remains unusable.
There is no automatic Qwen3-only, conservative-unbounded, empirical-free,
third-model, or altered-claim fallback in this chain. No such fallback is
prepared or authorized by this audit.

The current public state is intentionally incomplete:

- the opaque-provenance activation receipt records that the provenance branch
  is active while the full instrument remains pending;
- the reusable resource receipt is admitted;
- the full Gemma activation receipt is absent;
- the exact public common-schema canary receipt is absent;
- the distinct Gemma-substitution calibration receipt is absent;
- the symbolic pre-outcome receipt is absent; and
- the one-shot scientific authorization seal is absent.

These are ordered prerequisites, not interchangeable evidence. Preliminary
provenance or resource admission cannot substitute for full activation or the
exact bound canary.

## Fail-closed chain

| Stage | Required evidence | Fail-closed behavior if Gemma is unusable |
|---|---|---|
| Public provenance and resource admission | Exact immutable public receipts | Preliminary admission does not activate restricted inference. |
| Full Gemma activation | Exact activation schema, artifact/runtime hashes, permissions, effective runtime pin, read-only Qwen3 binding, and no semantic tuning | A missing or invalid activation stops the public coordinator before quarantine discovery. |
| Public joint-modality canary | Exact activation-bound raw-audio plus five-frame canary with the frozen schema | A missing, mismatched, or invalid canary prevents the activation seal. The canary cannot itself authorize restricted access. |
| Restricted substitution calibration | Valid public seal, local network denial, frozen sample/window binding, read-only Qwen3 outputs, and complete Gemma output | Restricted calibration is unreachable without the public seal. Incomplete or invalid Gemma output cannot emit a passing receipt. |
| Distinct aggregate receipt | New immutable receipt with exact replacement paths and `CALIBRATION_PASS` after all unchanged gates | `CALIBRATION_REVISE`, a missing receipt, the historical receipt, suppressed required ranges, or an unsafe public payload cannot pass pre-outcome gates. |
| Symbolic pre-outcome | Exact contract/erratum/runner/calibration binding and every construction, privacy, ancestry, cue-withholding, leakage, and confidence-only gate | No hand-authored or mismatched pre-outcome document is accepted. Scientific endpoints remain unopened. |
| One-shot outcome authorization | Owner-private exact seal binding protocol, execution contract, erratum, passed calibration, passed pre-outcome receipt, and runner bytes | A missing, public, stale, extra-field, or mismatched seal blocks execution. |
| Publication and replay | Complete paired corpus-seed bundles, external private consumption registry, deterministic merge | Interrupted work can resume only under the same unconsumed authorization. A completed authorization cannot replay, including after the published tree is moved. |

Current operational disposition:

`FAIL_CLOSED_NO_GEMMA_CALIBRATION_NO_SCIENTIFIC_OUTCOME`

This is an engineering/governance disposition, not a scientific endpoint or a
claim about whether the causal hypothesis is true.

## Neutral user-decision template

No option below is selected, recommended, authorized, or launched by this
document.

| Option | Direct cost and time | Principal risks | Potential value | New authority needed |
|---|---|---|---|---|
| A. Pause and preserve the current state | Low immediate cost; indefinite delay | The extension remains unanswered and technical dependencies may age | Preserves governance clarity and avoids spending against an unresolved instrument | A later explicit instruction to resume |
| B. Continue resolving the same frozen Gemma chain | Medium-to-high engineering time; possible storage/runtime or licensing work | Exact artifacts, permissions, runtime, or common-schema behavior may still fail; sunk cost is possible | If every unchanged gate passes, preserves the already frozen two-instrument calibration design | Explicit approval for the next bounded prerequisite action; restricted execution remains separately gated |
| C. Close this pending extension without an outcome | Low additional cost | Leaves the planned sensitivity question unanswered and retains only earlier evidence within its existing scope | Produces a clean stopping point with no claim inflation or further restricted processing | Explicit instruction to close/archive the branch |
| D. Obtain an independent governance/method review before choosing | Moderate review cost and delay | Review may not resolve the technical dependency and may request more evidence | Clarifies acceptable evidence, budget, and stopping rules before further expenditure | Explicit approval to prepare a public-only review packet or involve an authorized reviewer |

Suggested user response fields:

1. option letter;
2. maximum additional engineering time or spend;
3. acceptable evidence for resolving the option;
4. a stop date or stopping condition; and
5. whether any future restricted action must return for separate approval.

## Verification

The public/synthetic validation suite covers:

- immutable prior-stop and frozen-gate bindings;
- incomplete provenance activation cannot fully activate the instrument;
- exact artifact, license, runtime, canary, and byte-binding requirements;
- one parser/transport correction only and no semantic tuning;
- public activation failure before quarantine discovery;
- private read-only Qwen3 reuse and canonical-prefix checkpoint recovery;
- exact 15-item, 900-second, 137-window job construction without transcript input;
- `CALIBRATION_PASS` versus `CALIBRATION_REVISE` mapping;
- no cloud or third-model path;
- distinct-receipt, public-privacy, pre-outcome, and authorization enforcement;
- complete paired bundles, deterministic merging, fail-closed publication, and replay denial.

No restricted ChildLens payload, calibration effect, learner, causal arm, or
scientific endpoint was inspected or executed for this audit.
