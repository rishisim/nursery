# ChildLens v1.3 author-audit sampling contract

Status: frozen before author labels  
Scope: restricted packet preparation only

## Purpose and boundaries

The v1.3 audit uses exactly 15 minutes for the primary single-author audit and
predeclares one disjoint 15-minute reserve that may be activated once, only for
a frozen borderline result and only after the primary author record is locked.
The original v1.2 two-human workflow remains unchanged as the historical
gold-standard design.

The sampler accepts only the frozen v1.2 selection digest, 15 opaque selected
item keys, content-addressed local-media bindings, reference media durations,
and official speech-presence windows. It never discovers a quarantine, opens
media, or accepts model predictions, confidences, transcript text, lexical
labels, speaker labels, frames, or visual measures. Hosted models therefore
cannot affect sample inclusion or placement.

## Deterministic allocation

Times are placed on an integer-millisecond grid: starts round inward by ceiling,
ends round inward by floor, and ends are clipped to the declared media duration.
Official windows are sorted and coalesced. The preferred allocation is 60,000
milliseconds in every one of the 15 participant-distinct selected items for the
primary sample and another disjoint 60,000 milliseconds per item for the
reserve.

If an item has less than 120 seconds of unioned official speech coverage, its
available coverage is split evenly between primary and reserve. The resulting
deficits are assigned to other selected items in a SHA-256 order bound to the
frozen v1.2 selection digest, this sampler version, an explicit allocation role,
and the opaque item key. Every item must retain at least one second in both
samples, and the collection must support exactly 900 seconds in each sample.
Either shortfall fails closed. The public receipt records whether the preferred
one-minute-per-item design was achieved; it does not expose which item required
redistribution.

Within each item, SHA-256-derived leading slack, middle gap, and orientation
place the two allocated blocks in unioned official-speech time. Mapping a block
back to the recording may yield multiple physical segments when official
windows are separated. The primary and reserve physical segments are verified
disjoint.

## Restricted and public outputs

The sampler writes two owner-only immutable restricted files. The primary file
contains opaque audit and prediction-join keys, content-addressed media
bindings, and exact audit segments. The separately sealed reserve file contains
its disjoint segments and the activation condition
`BORDERLINE_ONLY_AFTER_PRIMARY_AUTHOR_RECORD_LOCK`. Keeping the reserve separate
prevents the normal author route from loading escalation material.

The repository-safe receipt contains only aggregate durations/counts, boolean
properties, and SHA-256 bindings. It contains no item key, media path, exact
interval, filename, timestamp, transcript, prediction, or item-level result.

The implementation intentionally has no command-line entry point. A trusted
zero-argument quarantine orchestrator must call it with already-resolved paths,
which avoids leaking restricted locations through argv and prevents broad
filesystem discovery.
