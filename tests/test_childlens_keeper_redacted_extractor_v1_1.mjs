import assert from "node:assert/strict";
import test from "node:test";

import {
  childLensKeeperRedactedExtractorV11,
} from "../scripts/childlens_keeper_redacted_extractor_v1_1.mjs";

const PUBLIC_LIBRARY = "c8ed0104-b793-4c35-817e-302afd4e036b";
const PUBLIC_SNAPSHOT = "4856662653b2fa183e53268d088cffba02a33443";
const OUTPUT_LIMIT = 4096;

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

function documentFixture({ rows = [], headers = [], anchors = [] } = {}) {
  return node({
    many: {
      "a[href]": anchors,
      "[data-childlens-file-row], [data-file-row], table tbody tr, [role='row'][data-file], [role='row'][data-name]": rows,
      "table thead th, [role='columnheader'], [data-schema-field]": headers,
    },
  });
}

function withPageGlobals(pageDocument, href, callback) {
  const oldDocument = globalThis.document;
  const oldLocation = globalThis.location;
  globalThis.document = pageDocument;
  globalThis.location = { href };
  try {
    return callback();
  } finally {
    if (oldDocument === undefined) {
      delete globalThis.document;
    } else {
      globalThis.document = oldDocument;
    }
    if (oldLocation === undefined) {
      delete globalThis.location;
    } else {
      globalThis.location = oldLocation;
    }
  }
}

function assertNoSentinels(receipt, sentinels) {
  const serialized = JSON.stringify(receipt);
  for (const sentinel of sentinels) {
    assert.equal(serialized.includes(sentinel), false);
  }
  assert.ok(Buffer.byteLength(serialized, "utf8") <= OUTPUT_LIMIT);
}

function assertExactTopLevelShape(receipt) {
  assert.deepEqual(Object.keys(receipt).sort(), [
    "controls",
    "errorCode",
    "extractorVersion",
    "inventory",
    "publicReleaseIdentity",
    "requestedProfile",
    "schemaCategories",
    "status",
  ]);
  assert.deepEqual(Object.keys(receipt.publicReleaseIdentity).sort(), [
    "doi",
    "doiObserved",
    "libraryObserved",
    "libraryUuid",
    "snapshotCommit",
    "snapshotObserved",
  ]);
  assert.deepEqual(Object.keys(receipt.controls).sort(), [
    "constantErrorCodes",
    "exactTimestampReturned",
    "fallbackPayloadReturned",
    "fixedKeyAllowlist",
    "outputSizeCheckPassed",
    "outputSizeLimitBytes",
    "pageLocalAggregation",
    "rawTextReturned",
    "restrictedIdentifierReturned",
    "rowDataReturned",
  ]);
  assert.deepEqual(Object.keys(receipt.inventory).sort(), [
    "annotationCount",
    "annotationFolderCount",
    "annotationKnownBytes",
    "audioCount",
    "audioKnownBytes",
    "byteKnownCount",
    "byteMissingCount",
    "certificateCount",
    "displayParsedByteCount",
    "documentKnownBytes",
    "exactByteAttributeCount",
    "fileObjectCount",
    "finalRepresentationFolderOrdinalPlusOne",
    "folderCount",
    "imageCount",
    "imageFolderCount",
    "imageKnownBytes",
    "metadataCount",
    "metadataKnownBytes",
    "otherCount",
    "otherKnownBytes",
    "pageRowCount",
    "platformRepresentationFolderOrdinalPlusOne",
    "primaryScrollRange",
    "primaryScrollX",
    "primaryScrollY",
    "readmeCount",
    "scrollableContainerCount",
    "totalKnownBytes",
    "videoCount",
    "videoFolderCount",
    "videoKnownBytes",
  ]);
  assert.deepEqual(Object.keys(receipt.schemaCategories).sort(), [
    "activity",
    "location",
    "mediaAnnotationLinkage",
    "participantGrouping",
    "sessionOrDateGrouping",
    "speakerRole",
    "speechPresence",
    "unknownFieldPresent",
    "utteranceOrFrameTiming",
  ]);
}

test("success path emits only fixed aggregate fields and never synthetic sentinels", () => {
  const sentinels = [
    "SYNTHETIC_PERSON_KEY_X9Q7",
    "2099-12-31T23:59:58.777Z",
    "private-example-video-X9Q7.mp4",
    "/restricted/synthetic/path/X9Q7",
  ];
  const sizeCell = node({ text: "1.5 GiB" });
  const videoRow = node({
    attributes: {
      "data-file-name": `${sentinels[3]}/${sentinels[2]}`,
      "data-size-bytes": "1500",
    },
    text: `${sentinels.join(" ")} 1500 B`,
  });
  const metadataRow = node({
    attributes: { "data-file-name": `${sentinels[0]}.csv` },
    text: `${sentinels[0]}.csv ${sentinels[1]}`,
    one: {
      "[data-size-bytes], [data-file-size], .file-size, .size-column, [data-column='size']": sizeCell,
    },
  });
  const folderRow = node({
    attributes: { "data-file-name": sentinels[0], "data-type": "folder" },
    text: sentinels.join(" "),
  });
  const headers = [
    node({ text: `participant ${sentinels[0]}` }),
    node({ text: `recording_date ${sentinels[1]}` }),
    node({ text: "activity" }),
    node({ text: `unrecognized_${sentinels[0]}` }),
  ];
  const pageDocument = documentFixture({
    rows: [videoRow, metadataRow, folderRow],
    headers,
  });
  const href = `https://keeper.invalid/library/${PUBLIC_LIBRARY}/${PUBLIC_SNAPSHOT}/`;

  const receipt = withPageGlobals(pageDocument, href, () =>
    childLensKeeperRedactedExtractorV11({ profile: "combined" }),
  );

  assert.equal(receipt.status, "ok");
  assert.equal(receipt.errorCode, null);
  assert.equal(receipt.publicReleaseIdentity.libraryObserved, true);
  assert.equal(receipt.publicReleaseIdentity.snapshotObserved, true);
  assert.equal(receipt.inventory.pageRowCount, 3);
  assert.equal(receipt.inventory.fileObjectCount, 2);
  assert.equal(receipt.inventory.folderCount, 1);
  assert.equal(receipt.inventory.videoCount, 1);
  assert.equal(receipt.inventory.metadataCount, 1);
  assert.equal(receipt.inventory.byteKnownCount, 2);
  assert.equal(receipt.inventory.totalKnownBytes, 1_610_614_236);
  assert.equal(receipt.schemaCategories.participantGrouping, true);
  assert.equal(receipt.schemaCategories.sessionOrDateGrouping, true);
  assert.equal(receipt.schemaCategories.activity, true);
  assert.equal(receipt.schemaCategories.unknownFieldPresent, true);
  assert.equal(receipt.controls.outputSizeCheckPassed, true);
  assert.equal(receipt.controls.rawTextReturned, false);
  assert.equal(receipt.controls.rowDataReturned, false);
  assertExactTopLevelShape(receipt);
  assertNoSentinels(receipt, sentinels);
});

test("source exception becomes a constant failure receipt without exception text", () => {
  const sentinel = "SYNTHETIC_EXCEPTION_SECRET_Q4M2";
  const pageDocument = {
    querySelectorAll() {
      throw new Error(sentinel);
    },
  };

  const receipt = withPageGlobals(pageDocument, `https://example.invalid/${PUBLIC_LIBRARY}`, () =>
    childLensKeeperRedactedExtractorV11({ profile: "inventory" }),
  );

  assert.equal(receipt.status, "error");
  assert.equal(receipt.errorCode, "E_NO_ROWS");
  assert.equal(receipt.inventory.fileObjectCount, 0);
  assert.equal(receipt.controls.fallbackPayloadReturned, false);
  assertExactTopLevelShape(receipt);
  assertNoSentinels(receipt, [sentinel]);
});

test("throwing source getters cannot escape through a success receipt", () => {
  const sentinel = "SYNTHETIC_GETTER_SECRET_W8N3";
  const row = {
    getAttribute(name) {
      if (name === "data-file-name") {
        return "aggregate-test.mp4";
      }
      if (name === "data-size-bytes") {
        throw new Error(sentinel);
      }
      return null;
    },
    get textContent() {
      throw new Error(sentinel);
    },
    querySelector() {
      throw new Error(sentinel);
    },
  };
  const receipt = withPageGlobals(
    documentFixture({ rows: [row] }),
    `https://example.invalid/${PUBLIC_LIBRARY}`,
    () => childLensKeeperRedactedExtractorV11({ profile: "inventory" }),
  );

  assert.equal(receipt.status, "ok");
  assert.equal(receipt.inventory.videoCount, 1);
  assert.equal(receipt.inventory.byteMissingCount, 1);
  assertNoSentinels(receipt, [sentinel]);
});

test("invalid configuration containing a sentinel fails with a constant code", () => {
  const sentinel = "SYNTHETIC_CONFIG_SECRET_R2P6";
  const receipt = childLensKeeperRedactedExtractorV11({
    profile: "inventory",
    unrestrictedFallback: sentinel,
  });

  assert.equal(receipt.status, "error");
  assert.equal(receipt.errorCode, "E_CONFIG");
  assert.equal(receipt.requestedProfile, "combined");
  assertExactTopLevelShape(receipt);
  assertNoSentinels(receipt, [sentinel]);
});

test("missing document and wrong release identity fail closed", () => {
  const noDocument = withPageGlobals(undefined, "", () =>
    childLensKeeperRedactedExtractorV11({ profile: "release" }),
  );
  assert.equal(noDocument.errorCode, "E_DOCUMENT_UNAVAILABLE");

  const wrongIdentity = withPageGlobals(documentFixture(), "https://example.invalid/other", () =>
    childLensKeeperRedactedExtractorV11({ profile: "release" }),
  );
  assert.equal(wrongIdentity.errorCode, "E_IDENTITY_MISMATCH");
  assertExactTopLevelShape(noDocument);
  assertExactTopLevelShape(wrongIdentity);
});

test("inventory and schema profiles require their page-local evidence", () => {
  const inventory = withPageGlobals(
    documentFixture(),
    `https://example.invalid/${PUBLIC_LIBRARY}`,
    () => childLensKeeperRedactedExtractorV11({ profile: "inventory" }),
  );
  const schema = withPageGlobals(
    documentFixture(),
    `https://example.invalid/${PUBLIC_LIBRARY}`,
    () => childLensKeeperRedactedExtractorV11({ profile: "schema" }),
  );
  assert.equal(inventory.errorCode, "E_NO_ROWS");
  assert.equal(schema.errorCode, "E_NO_SCHEMA_HEADERS");
});

test("serialized function executes without module closure state", () => {
  const serializedFunction = childLensKeeperRedactedExtractorV11.toString();
  const reconstructed = Function(`return (${serializedFunction});`)();
  const pageDocument = documentFixture({
    rows: [node({ attributes: { "data-file-name": "synthetic.mp4" } })],
  });
  const receipt = withPageGlobals(
    pageDocument,
    `https://example.invalid/${PUBLIC_LIBRARY}`,
    () => reconstructed({ profile: "inventory" }),
  );

  assert.equal(receipt.status, "ok");
  assert.equal(receipt.inventory.videoCount, 1);
  assertExactTopLevelShape(receipt);
});

test("output-size guard returns a constant failure code", () => {
  const oldStringify = JSON.stringify;
  const pageDocument = documentFixture();
  let receipt;
  try {
    JSON.stringify = () => "x".repeat(OUTPUT_LIMIT + 1);
    receipt = withPageGlobals(
      pageDocument,
      `https://example.invalid/${PUBLIC_LIBRARY}`,
      () => childLensKeeperRedactedExtractorV11({ profile: "release" }),
    );
  } finally {
    JSON.stringify = oldStringify;
  }

  assert.equal(receipt.status, "error");
  assert.equal(receipt.errorCode, "E_OUTPUT_SIZE");
  assert.equal(receipt.controls.outputSizeCheckPassed, false);
  assertExactTopLevelShape(receipt);
});

test("every observed error code is from the fixed error allowlist", () => {
  const allowed = new Set([
    "E_CONFIG",
    "E_DOCUMENT_UNAVAILABLE",
    "E_IDENTITY_MISMATCH",
    "E_NO_ROWS",
    "E_NO_SCHEMA_HEADERS",
    "E_NUMERIC_RANGE",
    "E_OUTPUT_KEYS",
    "E_OUTPUT_VALUE",
    "E_OUTPUT_SIZE",
    "E_STAGE_DOCUMENT",
    "E_STAGE_IDENTITY",
    "E_STAGE_INVENTORY",
    "E_STAGE_SCHEMA",
    "E_STAGE_VALIDATE",
    "E_INTERNAL",
  ]);
  const receipts = [
    childLensKeeperRedactedExtractorV11({ profile: "invalid" }),
    withPageGlobals(undefined, "", () =>
      childLensKeeperRedactedExtractorV11({ profile: "release" }),
    ),
    withPageGlobals(documentFixture(), "https://example.invalid/other", () =>
      childLensKeeperRedactedExtractorV11({ profile: "release" }),
    ),
  ];
  for (const receipt of receipts) {
    assert.equal(allowed.has(receipt.errorCode), true);
    assertNoSentinels(receipt, ["SYNTHETIC_UNRETURNED_EXCEPTION_DETAIL"]);
  }
});
