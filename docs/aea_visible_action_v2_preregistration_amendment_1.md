# AEA visible-action v2 preregistration amendment 1

**Frozen:** 2026-07-15T00:07:49Z  
**Parent protocol:** `aea-visible-action-v2`  
**Parent protocol SHA-256:** `b17ae9869461f508f7b63d7d90ad69ab4779554bd800af1f3a79fc9addf147ab`

## Reason and ordering

The first metadata-only implementation preflight, before any new RGB/IMU access, v2 annotation, split result, or model result, proved that the original instruction to fill every development event group to four sampled rows is impossible: six development event groups contain only one or three total windows. No media path was constructed or opened. The original preregistration and receipt remain unchanged.

This amendment changes only the deterministic allocation of the 24 expansion rows. Sample size, the 48 carried-forward prelabel IDs, every ontology definition, blinding rule, dense-evidence rule, support threshold, model budget, gate, decision, and prohibition remain unchanged.

## Effective allocation

1. Begin with all 48 v1 prelabel-manifest IDs, with labels and rationales absent.
2. For every development event group, fill to `min(4, total development windows in that group)`, using the original within-group candidate order: lowest current ASR-action sample count, lowest total development support for that ASR action, salted example-ID hash, then example ID.
3. Until exactly 72 unique rows are selected, repeatedly choose among groups with an unselected row by: lowest current selected count, then lowest selected/available coverage fraction, then SHA-256 of `aea-visible-action-v2-group|event_group`, then group ID. Choose the next row inside that group with the same original within-group candidate order.

This remains group- and support-aware, includes every development group, and depends only on frozen development metadata. There is no retry or alternate sample.
