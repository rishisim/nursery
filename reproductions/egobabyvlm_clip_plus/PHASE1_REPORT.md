# Phase 1 source/data audit report

## Status: blocked by external evidence

The exact BabyView 2025.1 subset used for the published BabyView CLIP+ row is
not reproducibly identifiable from the official public sources or pinned code.
The Databrary release metadata is reported as 894 hours; the pinned EgoBabyVLM
README describes approximately 863 hours. An official BabyView project analysis
of release 2025.1 separately reports 868 analyzed hours (and 3,163,675 frames
sampled at 1 Hz). Public evidence does not define what differs among these
accounting totals. The pinned preprocessing implementation
recursively accepts all media supplied to it, but publishes neither an
inclusion list nor a release-to-training selection rule. Therefore the roughly
31-hour difference—and the intervening 868-hour figure—is **not reconciled**,
and no exclusion causes are inferred.

The minimum sufficient external artifact is an author-provided privacy-safe
inclusion-manifest digest plus a governed list that can be matched to authorized
release assets (or an equivalent deterministic selection rule and excluded
duration by reason). A populated local audit ledger alone could establish the
894-hour inventory, but could not identify the paper's 863-hour subset.

## Aggregate audit

- Complete authorized 2025.1 inventory present: not assessed (governed storage unavailable)
- Total recordings and duration: not assessed
- Included/excluded duration: not assessed
- Missing, corrupt, or audio-unusable totals: not assessed
- Release membership evidence: official Databrary volume metadata only; no local asset-level comparison
- 894 → ~863 reconciled: no

## Primary evidence inspected

- BabyView 2025.1 controlled release: <https://databrary.org/volume/1882>
- Pinned EgoBabyVLM source and ≈863-hour statement:
  <https://github.com/facebookresearch/egobabyvlm/tree/224621caf0628270b6115845ac75a65b984234a3>
- Pinned preprocessing documentation and implementation:
  <https://github.com/facebookresearch/egobabyvlm/tree/224621caf0628270b6115845ac75a65b984234a3/apps/data_preprocessing>
- Official BabyView project release-2025.1 analysis reporting 868 hours:
  <https://babyview-project.github.io/assets/pdf/BV_Objects_CCN_2025.pdf>
- EgoBabyVLM paper v1: <https://arxiv.org/abs/2605.19130v1>

Only this aggregate status is tracked. Detailed ledger rows, source paths,
references, and hashes are restricted to governed storage.
