# ChildLens v1.3 model-assisted feasibility decision

## Terminal state

`CHILDLENS_AUTHOR_AUDIT_READY`

Decision basis: `FULL_LOCAL_PSEUDO_ANNOTATION_AND_BLINDED_AUTHOR_PACKET_READY`.

This state means that local preparation is complete and the qualified author's
blinded judgment is the remaining feasibility gate. It is not a finding that
the lexical-grounding hypothesis is supported, and it is not permission to
train a learner or run a causal arm.

## Completed evidence

- The v1.3 primary author sample is frozen at exactly 900 speech-seconds across
  all 15 participant-distinct selected items. A disjoint 900-second reserve is
  sealed for at most one post-lock borderline escalation.
- Selection used only the frozen release binding and official speech-presence
  windows. Model outputs, confidence, lexical content, visual salience, and
  apparent success were unavailable to the sampler.
- Fixed local instruments completed all 60 scheduled work units over 912
  candidate windows and 135.25 speech-minutes. No work unit is pending or
  failed.
- Restricted inference ran with network denial, scrubbed subprocess
  environments, two CPU workers, at most one MPS-heavy process, checkpointed
  resume, quarantine-only outputs, and the inherited 73-GiB namespace / 50-GiB
  free-space controls.
- The blinded `AUTHOR_AUDIT_A` app is initialized for 15 items / 15
  speech-minutes, loopback-only, owner-private, immediately autosaved, and
  resumable. Its pre-use workload estimate is 75 minutes.
- The post-lock comparator is prepared for all six frozen agreement gates,
  10,000 whole-item bootstrap replicates, 90% bounds, leave-one-item-out
  influence, and the one-time reserve rule.
- Historical v1, v1.1, and v1.2 artifact-set digests remain unchanged. The
  repository privacy and ancestry validator reports no issue.

One content-independent engineering correction was made before any usable
visual pseudo-label existed: every fixed frame is deterministically resized to
a maximum 448-pixel edge to bound MPS memory. Frame times, item selection,
scientific thresholds, and author sampling did not change.

## Scientific and privacy interpretation

All machine annotations are local measurement-instrument pseudo-labels, never
ground truth. Only the irreversibly locked author-audited subset can become
human-labeled ChildLens evidence. Inter-human reliability is unavailable in
v1.3; the permitted comparison is model--human agreement only. Unaudited
pseudo-labels may support nonidentifying aggregate simulator calibration and
candidate generation, but cannot be primary evaluation truth. Any later causal
evaluation must use simulator oracle labels as its primary truth.

GPT-5.6 Luna, Codex, hosted APIs, and cloud models were excluded from ChildLens
content annotation. They received no restricted frame, audio, transcript,
identifier, timestamp, or model-label payload. No learner, corpus tokenizer,
checkpoint, causal arm, or scientific acquisition outcome was created.

## Exact next task

`QUALIFIED_AUTHOR_COMPLETES_AND_IRREVERSIBLY_LOCKS_AUTHOR_AUDIT_A_THEN_RUNS_MODEL_HUMAN_COMPARISON`
