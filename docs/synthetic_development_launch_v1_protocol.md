# Synthetic development launch protocol v1

This package authorizes only a later, separate, one-shot synthetic development cohort. It does not execute development during sealing and never authorizes confirmation.

The claim is narrow: under the frozen synthetic observation geometry, raw sensor evidence may assist weak lexical/action grounding while language-only held-out prompts remain the evaluation surface. No infant-learning, developmental, ecological-validity, or real-data claim is authorized.

The package carries forward V8's untouched development reserve: 40 corpus seeds (`281001` through `281079`, odd) and three stochastic model seeds (`281101`, `281103`, `281107`). It carries the untouched 40-by-3 confirmation reserve under `289xxx` identifiers, but no confirmation command exists. Statistical inference genuinely uses development inference identifier `281201`; confirmation identifier `289201` remains blocked. Fixture and excluded rehearsals use fresh `290xxx` identifiers only.

Each corpus independently regenerates V8's exact noun independence, action relation/position balance, seed-invariant nuisance schedule, synchronized condition, matched randomized shuffle, signed shifts, uninformative stream, absent channel, corrupted sensor, and informative/weak/zero-information/corrupted detector strata. The actual frozen V2 detector runs. Training uses the V8 joint cross-situational learner without projection, key alignment, truth-conditioned canonicalization, condition-specific replacement, or candidate-order dependence.

Evaluation is cue-free and language-only. Held-out prompt instances are never training rows. Primitive and manner prompts are component endpoints; joint action prompts are the compositional endpoint; noun prompts are evaluated with exactly zero sensor weight. Scoring is the same permutation-invariant fractional-top, correct-probability, log-loss, multiclass-Brier, calibration, and tie rule in every condition.

Corpus seed is the independent unit. The three stochastic model replicates are averaged within corpus before inference. The primary endpoint is held-out action-present mean correct probability. The two co-primary paired contrasts are synchronized minus absent channel and synchronized minus matched randomized shuffle. The intersection-union rule requires both to pass: mean effect and one-sided 95% paired-bootstrap lower bound must exceed `0.005`, the exact one-sided sign-test p-value must be at most `0.05`, and at least 60% of corpus effects must be positive.

Inference uses 20,000 deterministic paired-corpus bootstrap resamples plus an exact sign test. It reports bounded support, zero mass, boundary mass, median, range, positive fraction, and an explicit degenerate mode. All-zero effects fail. A positive point mass above threshold can pass only with its exact sign test. Non-degenerate inference with fewer than eight nonzero corpora fails.

Present and null learnability never aggregate. Every corpus's replicate-averaged primitive, manner, action, and noun subgroup must independently pass the frozen chance-margin, probability, log-loss, Brier, and tie gates in both synchronized and absent-channel conditions. Action-null synchronized performance must also be noninferior to absent and shuffled comparators at margin `-0.02`. Removing target-absent training or corrupting target-absence evidence must reduce every null subgroup by at least `0.20`.

Secondary controls include both signed shifts, uninformative, corrupted, noun, zero-information, primitive, manner, oracle, direct-capacity, factor-refit, detector-capacity, side-modality, and order-invariance audits. Corrupted sensor gain is capped at `0.03`; noun synchronized-versus-absent probability must be identical to numerical tolerance.

The exact development runner refuses to start unless its output directory is absent, the frozen source manifest verifies, and `DEVELOPMENT_LAUNCH_READY.json` validates every frozen provenance hash and authorization digest. Creating the output directory is the one-shot marker: a crash or completed run both prohibit silent rerun. The package contains no confirmation command.

The frozen one-shot command is:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python output/synthetic_development_launch_package_v1/frozen_source_snapshot/scripts/run_synthetic_development_launch_v1.py run-development --snapshot-root output/synthetic_development_launch_package_v1/frozen_source_snapshot --authorization output/synthetic_development_launch_package_v1/DEVELOPMENT_LAUNCH_READY.json --output-root output/synthetic_development_v1_one_shot
```

Development terminal behavior is exact:

- `DEVELOPMENT_PASS`: every frozen gate and both co-primary contrasts pass.
- `DEVELOPMENT_NO_GO`: any frozen development gate fails, including a degenerate or underpowered observed distribution.
- A runtime/integrity failure is not interpreted as a scientific result and leaves the one-shot directory in place for forensic disposition.

Confirmation always requires a later, separately authorized package and task.

