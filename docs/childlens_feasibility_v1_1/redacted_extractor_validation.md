# ChildLens Keeper redacted extractor validation v1.1

Version: `childlens-keeper-redacted-extractor-validation-v1.1.7`  
Validation scope: synthetic, local, unauthenticated  
Scientific outcome: none

## Disposition

The fail-closed page-local extractor required by the v1 privacy audit is implemented at `scripts/childlens_keeper_redacted_extractor_v1_1.mjs`. It was tested without Keeper, browser state, email, restricted data, or network access. This validation does not establish release identity, permission, corpus counts, schema coverage, or any scientific feasibility gate; it establishes only the extractor's synthetic safety behavior.

The function is self-contained and suitable for direct serialization by Playwright `page.evaluate`. The caller selects one of four fixed profiles: `release`, `inventory`, `schema`, or `combined`. Any other configuration, including extra keys, fails with a constant code.

## Fixed export contract

Success and failure receipts have the same exact nested key shape. The only exported value classes are:

- fixed technical status/profile/version constants and constant error codes;
- the already-public DOI, Keeper library UUID, and public snapshot commit;
- booleans indicating whether those public identities were observed;
- nonnegative safe-integer object counts and known-byte totals by broad class;
- booleans for broad schema categories; and
- fixed control booleans documenting page-local aggregation, key validation, size validation, and the absence of raw/fallback return fields.

The extractor contains no return path for a filename, path, source identifier, exact timestamp, row, header string, exception object/message/stack, DOM fragment, response body, CSV content, or generic source serialization. Internal source strings are used only to increment fixed counters or set fixed booleans. Parse failure never falls back to source material. Version 1.1.7 additionally reports fixed numeric scroll geometry, public folder ordinals, and separate exact-attribute versus display-parsed byte provenance; these remain bounded numeric aggregates under the same fixed-key/value checks.

Before a success receipt returns, the function checks the exact keys at every nesting level, rejects any string outside a fixed literal allowlist, rejects unsafe or negative numbers, and checks a 4,096-byte serialized-output ceiling. Failure receipts are constructed only from fixed literals and zeroed aggregates. Authentication, navigation, and permission checks remain outside this function and must still fail closed.

## Synthetic sentinel coverage

The Node test suite places synthetic secret markers in page-local filename/path-like labels, row text, a date-time-like string, schema-header text, caller configuration, thrown exception messages, and throwing DOM getters. Assertions serialize each returned receipt and prove those markers are absent on both success and failure paths.

The suite also verifies:

- exact top-level and nested output key allowlists;
- aggregate counts and byte arithmetic without source-text return;
- fixed release-identity matching;
- constant codes for missing documents, wrong identity, missing inventory rows, missing schema headers, and invalid configuration;
- explicit exercise of the output-size failure path;
- no dependency on module closure state after function serialization; and
- the fixed error-code allowlist.

## Invocation contract

Import the function in a local Playwright harness and pass it directly to `page.evaluate` with exactly one fixed profile key. Inspect and approve the script itself before use. The browser call must return only the receipt; callers must not request a DOM snapshot, table text, directory listing, response body, CSV rows, console capture, or exception details as a fallback.

Example, shown without navigation or authentication:

```js
import { childLensKeeperRedactedExtractorV11 } from "./scripts/childlens_keeper_redacted_extractor_v1_1.mjs";

const receipt = await page.evaluate(childLensKeeperRedactedExtractorV11, {
  profile: "inventory",
});
```

If the extractor returns an error receipt or a page layout is unsupported, stop and revise this locally tested extractor. Do not switch to a broader browser capture.

## Validation command

```text
node --test tests/test_childlens_keeper_redacted_extractor_v1_1.mjs
```

Test receipt on 2026-07-21 with Node.js v25.8.1: **PASS, 9/9 tests; exit status 0**. No authenticated page was accessed by this workstream.
