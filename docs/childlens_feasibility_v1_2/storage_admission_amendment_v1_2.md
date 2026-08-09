# ChildLens v1.2 conservative selected-transfer admission amendment

Date frozen: 2026-07-21  
Scope: storage/transfer engineering only  
Selection or scientific threshold changed: no

## Reason

The authenticated Keeper video table exposes rounded display sizes but no exact-byte attribute. Exact remote size metadata could not be obtained without a separate native authenticated route. The user explicitly authorized one bounded conservative amendment before media inspection. This amendment does not alter the frozen 15-item selection, scientific gates, endpoints, roles, referential ontology, or stopping rules.

## Frozen calculation

The synthetically tested page-local extractor matched all 15 frozen source locators to exactly one video row each without returning a filename, locator, row, identifier, timestamp, or per-item size.

For every displayed size, the admission controller adds one complete least-significant displayed unit, covering either nearest rounding or truncation at the displayed precision. It then adds transfer/storage overhead equal to 1% of the summed rounding upper bounds plus 16 MiB per selected object.

Aggregate receipt:

- selected rows requested and matched: 15 of 15;
- exact-byte attributes exposed: 0;
- display-parsed rows: 15;
- displayed parsed total: 14,071,800,000 bytes;
- rounding upper-bound total: 14,572,800,000 bytes;
- fixed and percentage overhead: 397,386,240 bytes;
- conservative admission total: 14,970,186,240 bytes;
- admission ceiling: 18 GiB = 19,327,352,832 bytes;
- hard raw cap: 20 GiB = 21,474,836,480 bytes.

The conservative total is 4,357,166,592 bytes below the 18-GiB ceiling and 6,504,650,240 bytes below the hard raw cap. It therefore preserves more than the required 2-GiB margin.

## Transfer controller requirements

1. Transfer only the unchanged frozen 15-item plan whose canonical digest is already receipted in v1.1.
2. Download sequentially, with at most one full object in flight and no duplicate full-resolution copy.
3. Stream directly into the admitted restricted quarantine through a mode-`0600` temporary file.
4. Count actual bytes while streaming and abort before the cumulative total reaches 21,474,836,480 bytes; the implementation limit is 21,474,836,479 bytes.
5. Before every object, recheck the 73-GiB namespace peak and 50-GiB projected free-space floor.
6. Verify final byte count and local SHA-256 before atomic promotion; retain exact per-object metadata only in the restricted manifest.
7. Export only aggregate counts, total bytes, digest receipts, and fixed status codes.

The bound permits transfer engineering to proceed. It is not evidence about audio, language, speech, lexical content, referential visibility, or any scientific effect.
