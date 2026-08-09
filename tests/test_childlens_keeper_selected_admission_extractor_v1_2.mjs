import assert from "node:assert/strict";
import test from "node:test";

import { childLensKeeperSelectedAdmissionExtractorV12 } from "../scripts/childlens_keeper_selected_admission_extractor_v1_2.mjs";

const LIBRARY = "c8ed0104-b793-4c35-817e-302afd4e036b";

function node({ attributes = {}, text = "", one = {}, many = {} } = {}) {
  return {
    getAttribute(name) {
      return Object.hasOwn(attributes, name) ? attributes[name] : null;
    },
    get textContent() {
      return text;
    },
    querySelector(selector) {
      return one[selector] ?? null;
    },
    querySelectorAll(selector) {
      return many[selector] ?? [];
    },
  };
}

const ROW_SELECTOR = "[data-childlens-file-row], [data-file-row], table tbody tr, [role='row'][data-file], [role='row'][data-name]";
const SIZE_SELECTOR = "[data-size-bytes], [data-file-size], .file-size, .size-column, [data-column='size']";

function fixture(rows) {
  return node({ many: { [ROW_SELECTOR]: rows, "a[href]": [] } });
}

function row(name, size, exact = false) {
  const sizeNode = node({ attributes: exact ? { "data-size-bytes": size } : {}, text: size });
  return node({
    attributes: { "data-file-name": name },
    one: { [SIZE_SELECTOR]: sizeNode },
  });
}

function withPage(documentValue, callback) {
  const oldDocument = globalThis.document;
  const oldLocation = globalThis.location;
  globalThis.document = documentValue;
  globalThis.location = { href: `https://keeper.invalid/library/${LIBRARY}/` };
  try {
    return callback();
  } finally {
    if (oldDocument === undefined) delete globalThis.document;
    else globalThis.document = oldDocument;
    if (oldLocation === undefined) delete globalThis.location;
    else globalThis.location = oldLocation;
  }
}

test("returns only aggregate conservative admission values", () => {
  const secretA = "SYNTHETIC_PRIVATE_ALPHA.mp4";
  const secretB = "SYNTHETIC_PRIVATE_BETA.mp4";
  const receipt = withPage(fixture([
    row(secretA, "1.2 GB"),
    row(secretB, "800 MB"),
    row("UNSELECTED_PRIVATE.mp4", "9 GB"),
  ]), () => childLensKeeperSelectedAdmissionExtractorV12({ sourceLocators: [`videos/${secretA}`, `videos/${secretB}`] }));
  assert.equal(receipt.status, "ok");
  assert.equal(receipt.requestedCount, 2);
  assert.equal(receipt.matchedCount, 2);
  assert.equal(receipt.displayParsedCount, 2);
  assert.equal(receipt.exactByteCount, 0);
  assert.equal(receipt.displayedParsedBytes, 2_000_000_000);
  assert.equal(receipt.roundingUpperBoundBytes, 2_101_000_000);
  assert.equal(receipt.transferOverheadBytes, 54_564_432);
  assert.equal(receipt.conservativeAdmissionBytes, 2_155_564_432);
  assert.equal(receipt.admissionCeilingPass, true);
  const serialized = JSON.stringify(receipt);
  assert.equal(serialized.includes(secretA), false);
  assert.equal(serialized.includes(secretB), false);
  assert.equal(serialized.includes("UNSELECTED_PRIVATE"), false);
  assert.ok(Buffer.byteLength(serialized) <= 2048);
});

test("accepts exact-byte attributes but does not emit per-item values", () => {
  const receipt = withPage(fixture([row("selected.mp4", "12345", true)]), () =>
    childLensKeeperSelectedAdmissionExtractorV12({ sourceLocators: ["videos/selected.mp4"] }),
  );
  assert.equal(receipt.status, "ok");
  assert.equal(receipt.exactByteCount, 1);
  assert.equal(receipt.displayParsedCount, 0);
  assert.equal(receipt.roundingUpperBoundBytes, 12345);
  assert.equal(JSON.stringify(receipt).includes("selected.mp4"), false);
});

test("fails closed on missing selected row", () => {
  const receipt = withPage(fixture([row("other.mp4", "1 GB")]), () =>
    childLensKeeperSelectedAdmissionExtractorV12({ sourceLocators: ["videos/missing.mp4"] }),
  );
  assert.equal(receipt.status, "error");
  assert.equal(receipt.errorCode, "E_SELECTION_INCOMPLETE");
  assert.equal(receipt.matchedCount, 0);
});

test("fails closed on duplicate selected row", () => {
  const receipt = withPage(fixture([row("same.mp4", "1 GB"), row("same.mp4", "1 GB")]), () =>
    childLensKeeperSelectedAdmissionExtractorV12({ sourceLocators: ["videos/same.mp4"] }),
  );
  assert.equal(receipt.errorCode, "E_SELECTION_DUPLICATE");
});

test("fails closed on unparseable size", () => {
  const receipt = withPage(fixture([row("same.mp4", "unknown")]), () =>
    childLensKeeperSelectedAdmissionExtractorV12({ sourceLocators: ["videos/same.mp4"] }),
  );
  assert.equal(receipt.errorCode, "E_SIZE_MISSING");
});

test("rejects invalid, URL, or duplicate locator configuration", () => {
  const invalid = childLensKeeperSelectedAdmissionExtractorV12({ sourceLocators: ["https://example.invalid/a.mp4"] });
  const duplicate = childLensKeeperSelectedAdmissionExtractorV12({ sourceLocators: ["a/x.mp4", "b/x.mp4"] });
  assert.equal(invalid.errorCode, "E_CONFIG");
  assert.equal(duplicate.errorCode, "E_CONFIG");
});

test("serialized function executes without closure state", () => {
  const reconstructed = Function(`return (${childLensKeeperSelectedAdmissionExtractorV12.toString()});`)();
  const receipt = withPage(fixture([row("selected.mp4", "1 GB")]), () =>
    reconstructed({ sourceLocators: ["videos/selected.mp4"] }),
  );
  assert.equal(receipt.status, "ok");
  assert.equal(receipt.matchedCount, 1);
});

test("wrong identity and absent document fail with constants", () => {
  const oldDocument = globalThis.document;
  const oldLocation = globalThis.location;
  globalThis.document = fixture([row("selected.mp4", "1 GB")]);
  globalThis.location = { href: "https://example.invalid/other" };
  try {
    const wrong = childLensKeeperSelectedAdmissionExtractorV12({ sourceLocators: ["videos/selected.mp4"] });
    assert.equal(wrong.errorCode, "E_IDENTITY_MISMATCH");
  } finally {
    if (oldDocument === undefined) delete globalThis.document;
    else globalThis.document = oldDocument;
    if (oldLocation === undefined) delete globalThis.location;
    else globalThis.location = oldLocation;
  }
  const absent = withPage(undefined, () => childLensKeeperSelectedAdmissionExtractorV12({ sourceLocators: ["videos/selected.mp4"] }));
  assert.equal(absent.errorCode, "E_DOCUMENT_UNAVAILABLE");
});
