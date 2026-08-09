/**
 * Fail-closed, page-local Keeper receipt extractor for ChildLens feasibility v1.1.
 *
 * This function is deliberately self-contained. Playwright can serialize it
 * exactly with:
 *
 *   await page.evaluate(childLensKeeperRedactedExtractorV11, {
 *     profile: "inventory",
 *   });
 *
 * It reads page values only inside the browser context. No source text,
 * filename, path, identifier, timestamp, row, exception, or fallback payload
 * has a return path.
 */
export function childLensKeeperRedactedExtractorV11(input) {
  "use strict";

  const VERSION = "childlens-keeper-redacted-extractor-v1.1.7";
  const DOI = "10.17617/4.fe";
  const LIBRARY_UUID = "c8ed0104-b793-4c35-817e-302afd4e036b";
  const SNAPSHOT_COMMIT = "4856662653b2fa183e53268d088cffba02a33443";
  const OUTPUT_SIZE_LIMIT_BYTES = 4096;
  const PROFILES = new Set(["combined", "inventory", "schema", "release"]);
  const ERROR_CODES = new Set([
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

  function blankReceipt(profile, status, errorCode) {
    return {
      status,
      errorCode,
      extractorVersion: VERSION,
      requestedProfile: profile,
      publicReleaseIdentity: {
        doi: DOI,
        libraryUuid: LIBRARY_UUID,
        snapshotCommit: SNAPSHOT_COMMIT,
        doiObserved: false,
        libraryObserved: false,
        snapshotObserved: false,
      },
      inventory: {
        pageRowCount: 0,
        scrollableContainerCount: 0,
        primaryScrollX: 0,
        primaryScrollY: 0,
        primaryScrollRange: 0,
        fileObjectCount: 0,
        folderCount: 0,
        annotationFolderCount: 0,
        imageFolderCount: 0,
        videoFolderCount: 0,
        finalRepresentationFolderOrdinalPlusOne: 0,
        platformRepresentationFolderOrdinalPlusOne: 0,
        videoCount: 0,
        audioCount: 0,
        imageCount: 0,
        annotationCount: 0,
        metadataCount: 0,
        readmeCount: 0,
        certificateCount: 0,
        otherCount: 0,
        byteKnownCount: 0,
        byteMissingCount: 0,
        exactByteAttributeCount: 0,
        displayParsedByteCount: 0,
        totalKnownBytes: 0,
        videoKnownBytes: 0,
        audioKnownBytes: 0,
        imageKnownBytes: 0,
        annotationKnownBytes: 0,
        metadataKnownBytes: 0,
        documentKnownBytes: 0,
        otherKnownBytes: 0,
      },
      schemaCategories: {
        participantGrouping: false,
        sessionOrDateGrouping: false,
        mediaAnnotationLinkage: false,
        activity: false,
        location: false,
        speechPresence: false,
        utteranceOrFrameTiming: false,
        speakerRole: false,
        unknownFieldPresent: false,
      },
      controls: {
        pageLocalAggregation: true,
        fixedKeyAllowlist: true,
        constantErrorCodes: true,
        outputSizeLimitBytes: OUTPUT_SIZE_LIMIT_BYTES,
        outputSizeCheckPassed: false,
        rawTextReturned: false,
        rowDataReturned: false,
        restrictedIdentifierReturned: false,
        exactTimestampReturned: false,
        fallbackPayloadReturned: false,
      },
    };
  }

  function failure(code, profile) {
    const safeCode = ERROR_CODES.has(code) ? code : "E_INTERNAL";
    const safeProfile = PROFILES.has(profile) ? profile : "combined";
    return blankReceipt(safeProfile, "error", safeCode);
  }

  function sameKeys(value, expected) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      return false;
    }
    const actualKeys = Object.keys(value).sort();
    const expectedKeys = [...expected].sort();
    return (
      actualKeys.length === expectedKeys.length &&
      actualKeys.every((key, index) => key === expectedKeys[index])
    );
  }

  function hasExactOutputKeys(value) {
    return (
      sameKeys(value, [
        "status",
        "errorCode",
        "extractorVersion",
        "requestedProfile",
        "publicReleaseIdentity",
        "inventory",
        "schemaCategories",
        "controls",
      ]) &&
      sameKeys(value.publicReleaseIdentity, [
        "doi",
        "libraryUuid",
        "snapshotCommit",
        "doiObserved",
        "libraryObserved",
        "snapshotObserved",
      ]) &&
      sameKeys(value.inventory, [
        "pageRowCount",
        "scrollableContainerCount",
        "primaryScrollX",
        "primaryScrollY",
        "primaryScrollRange",
        "fileObjectCount",
        "folderCount",
        "annotationFolderCount",
        "imageFolderCount",
        "videoFolderCount",
        "finalRepresentationFolderOrdinalPlusOne",
        "platformRepresentationFolderOrdinalPlusOne",
        "videoCount",
        "audioCount",
        "imageCount",
        "annotationCount",
        "metadataCount",
        "readmeCount",
        "certificateCount",
        "otherCount",
        "byteKnownCount",
        "byteMissingCount",
        "exactByteAttributeCount",
        "displayParsedByteCount",
        "totalKnownBytes",
        "videoKnownBytes",
        "audioKnownBytes",
        "imageKnownBytes",
        "annotationKnownBytes",
        "metadataKnownBytes",
        "documentKnownBytes",
        "otherKnownBytes",
      ]) &&
      sameKeys(value.schemaCategories, [
        "participantGrouping",
        "sessionOrDateGrouping",
        "mediaAnnotationLinkage",
        "activity",
        "location",
        "speechPresence",
        "utteranceOrFrameTiming",
        "speakerRole",
        "unknownFieldPresent",
      ]) &&
      sameKeys(value.controls, [
        "pageLocalAggregation",
        "fixedKeyAllowlist",
        "constantErrorCodes",
        "outputSizeLimitBytes",
        "outputSizeCheckPassed",
        "rawTextReturned",
        "rowDataReturned",
        "restrictedIdentifierReturned",
        "exactTimestampReturned",
        "fallbackPayloadReturned",
      ])
    );
  }

  function hasOnlyAllowedValues(value) {
    const allowedStrings = new Set([
      "ok",
      "error",
      VERSION,
      "combined",
      "inventory",
      "schema",
      "release",
      DOI,
      LIBRARY_UUID,
      SNAPSHOT_COMMIT,
      ...ERROR_CODES,
    ]);
    const pending = [value];
    while (pending.length > 0) {
      const current = pending.pop();
      if (current === null) {
        continue;
      }
      if (typeof current === "boolean") {
        continue;
      }
      if (typeof current === "number") {
        if (!Number.isSafeInteger(current) || current < 0) {
          return false;
        }
        continue;
      }
      if (typeof current === "string") {
        if (!allowedStrings.has(current)) {
          return false;
        }
        continue;
      }
      if (typeof current === "object" && !Array.isArray(current)) {
        pending.push(...Object.values(current));
        continue;
      }
      return false;
    }
    return true;
  }

  function validateAndReturn(receipt, profile) {
    if (!hasExactOutputKeys(receipt)) {
      return failure("E_OUTPUT_KEYS", profile);
    }
    if (!hasOnlyAllowedValues(receipt)) {
      return failure("E_OUTPUT_VALUE", profile);
    }
    receipt.controls.outputSizeCheckPassed = true;
    const serialized = JSON.stringify(receipt);
    if (typeof serialized !== "string" || serialized.length > OUTPUT_SIZE_LIMIT_BYTES) {
      return failure("E_OUTPUT_SIZE", profile);
    }
    return receipt;
  }

  function safeAttribute(node, attribute) {
    try {
      const value = node && typeof node.getAttribute === "function"
        ? node.getAttribute(attribute)
        : null;
      return typeof value === "string" ? value : "";
    } catch {
      return "";
    }
  }

  function safeText(node) {
    try {
      return node && typeof node.textContent === "string" ? node.textContent : "";
    } catch {
      return "";
    }
  }

  function safeQueryAll(root, selector) {
    try {
      if (!root || typeof root.querySelectorAll !== "function") {
        return [];
      }
      return Array.from(root.querySelectorAll(selector));
    } catch {
      return [];
    }
  }

  function safeQuery(root, selector) {
    try {
      if (!root || typeof root.querySelector !== "function") {
        return null;
      }
      return root.querySelector(selector);
    } catch {
      return null;
    }
  }

  function normalizeInternal(value) {
    return String(value).normalize("NFKC").trim().toLowerCase();
  }

  function classifyInternal(label, row) {
    const normalized = normalizeInternal(label).split(/[?#]/, 1)[0];
    const explicitType = normalizeInternal(
      safeAttribute(row, "data-type") || safeAttribute(row, "data-kind"),
    );
    if (
      explicitType === "folder" ||
      explicitType === "directory" ||
      safeAttribute(row, "aria-expanded") !== "" ||
      safeQuery(
        row,
        "[data-type='folder'], [data-kind='folder'], .folder-icon, .fa-folder, [class*='folder-icon'], a[href*='/library/']",
      ) !== null
    ) {
      return "folder";
    }
    if (/\.(mp4|mov|mkv|webm|m4v)$/.test(normalized)) {
      return "video";
    }
    if (/\.(wav|flac|mp3|m4a|aac|ogg)$/.test(normalized)) {
      return "audio";
    }
    if (/\.(jpg|jpeg|png|webp|tif|tiff|bmp)$/.test(normalized)) {
      return "image";
    }
    if (/\.(json|eaf|textgrid|vtt|srt)$/.test(normalized)) {
      return "annotation";
    }
    if (/\.(csv|tsv|parquet|xlsx|xls)$/.test(normalized)) {
      return "metadata";
    }
    if (/(^|\/)readme(?:\.[a-z0-9]+)?$/.test(normalized)) {
      return "readme";
    }
    if (/(^|\/)(certificate|certification)(?:\.[a-z0-9]+)?$/.test(normalized)) {
      return "certificate";
    }
    return "other";
  }

  function getInternalLabel(row) {
    const direct = [
      safeAttribute(row, "data-file-name"),
      safeAttribute(row, "data-name"),
    ].find((value) => value !== "");
    if (direct) {
      return direct;
    }
    const candidate = safeQuery(
      row,
      "[data-file-name], [data-name], a[download], .file-name, .name-column, [data-column='name'], .sf-table-cell-name a, .sf-table-cell-name, .dirent-name a, .dirent-name, [class*='name'] a, td a[href]",
    );
    if (candidate) {
      const fromCandidate =
        safeAttribute(candidate, "data-file-name") ||
        safeAttribute(candidate, "data-name") ||
        safeAttribute(candidate, "download") ||
        safeText(candidate);
      if (fromCandidate) {
        return fromCandidate;
      }
    }
    return safeText(row);
  }

  function parseInternalBytes(raw, bytesOnly) {
    const normalized = String(raw).normalize("NFKC").replace(/\u00a0/g, " ").trim();
    if (bytesOnly) {
      if (!/^\d+$/.test(normalized)) {
        return null;
      }
      const parsed = Number(normalized);
      return Number.isSafeInteger(parsed) ? parsed : "range";
    }
    const match = normalized.match(/^(\d+(?:\.\d+)?)\s*(b|kb|mb|gb|tb|kib|mib|gib|tib)$/i);
    if (!match) {
      return null;
    }
    const unit = match[2].toLowerCase();
    const multipliers = {
      b: 1,
      kb: 1000,
      mb: 1000 ** 2,
      gb: 1000 ** 3,
      tb: 1000 ** 4,
      kib: 1024,
      mib: 1024 ** 2,
      gib: 1024 ** 3,
      tib: 1024 ** 4,
    };
    const parsed = Math.round(Number(match[1]) * multipliers[unit]);
    return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : "range";
  }

  function getInternalBytes(row) {
    const directBytes = safeAttribute(row, "data-size-bytes");
    if (directBytes !== "") {
      return parseInternalBytes(directBytes, true);
    }
    const cell = safeQuery(
      row,
      "[data-size-bytes], [data-file-size], .file-size, .size-column, [data-column='size']",
    );
    if (!cell) {
      return null;
    }
    const cellBytes = safeAttribute(cell, "data-size-bytes");
    if (cellBytes !== "") {
      return parseInternalBytes(cellBytes, true);
    }
    return parseInternalBytes(
      safeAttribute(cell, "data-file-size") || safeText(cell),
      false,
    );
  }

  function incrementSafe(inventory, key, amount) {
    const next = inventory[key] + amount;
    if (!Number.isSafeInteger(next) || next < 0) {
      return false;
    }
    inventory[key] = next;
    return true;
  }

  let profile = "combined";
  let stageErrorCode = "E_INTERNAL";
  try {
    if (input !== undefined && input !== null) {
      if (
        typeof input !== "object" ||
        Array.isArray(input) ||
        !sameKeys(input, ["profile"]) ||
        !PROFILES.has(input.profile)
      ) {
        return failure("E_CONFIG", profile);
      }
      profile = input.profile;
    }

    stageErrorCode = "E_STAGE_DOCUMENT";
    const pageDocument = typeof document === "object" ? document : null;
    if (!pageDocument || typeof pageDocument.querySelectorAll !== "function") {
      return failure("E_DOCUMENT_UNAVAILABLE", profile);
    }

    const receipt = blankReceipt(profile, "ok", null);

    stageErrorCode = "E_STAGE_IDENTITY";
    let pageLocation = "";
    try {
      pageLocation = typeof location === "object" && typeof location.href === "string"
        ? location.href
        : "";
    } catch {
      pageLocation = "";
    }
    const identityCandidates = [pageLocation];
    for (const anchor of safeQueryAll(pageDocument, "a[href]")) {
      identityCandidates.push(safeAttribute(anchor, "href"));
    }
    for (const candidate of identityCandidates) {
      const normalized = normalizeInternal(candidate);
      if (normalized.includes(DOI.toLowerCase())) {
        receipt.publicReleaseIdentity.doiObserved = true;
      }
      if (normalized.includes(LIBRARY_UUID)) {
        receipt.publicReleaseIdentity.libraryObserved = true;
      }
      if (normalized.includes(SNAPSHOT_COMMIT)) {
        receipt.publicReleaseIdentity.snapshotObserved = true;
      }
    }

    if (
      !receipt.publicReleaseIdentity.libraryObserved &&
      !receipt.publicReleaseIdentity.doiObserved
    ) {
      return failure("E_IDENTITY_MISMATCH", profile);
    }

    if (profile === "combined" || profile === "inventory") {
      stageErrorCode = "E_STAGE_INVENTORY";
      const rawRows = safeQueryAll(
        pageDocument,
        "[data-childlens-file-row], [data-file-row], table tbody tr, [role='row'][data-file], [role='row'][data-name]",
      );
      const rows = [...new Set(rawRows)];
      for (const row of rows) {
        const label = getInternalLabel(row);
        if (normalizeInternal(label) === "") {
          continue;
        }
        const kind = classifyInternal(label, row);
        if (!incrementSafe(receipt.inventory, "pageRowCount", 1)) {
          return failure("E_NUMERIC_RANGE", profile);
        }
        if (kind === "folder") {
          if (!incrementSafe(receipt.inventory, "folderCount", 1)) {
            return failure("E_NUMERIC_RANGE", profile);
          }
          const normalizedFolderLabel = normalizeInternal(label);
          if (/annotation/.test(normalizedFolderLabel)) {
            if (!incrementSafe(receipt.inventory, "annotationFolderCount", 1)) {
              return failure("E_NUMERIC_RANGE", profile);
            }
          } else if (/image/.test(normalizedFolderLabel)) {
            if (!incrementSafe(receipt.inventory, "imageFolderCount", 1)) {
              return failure("E_NUMERIC_RANGE", profile);
            }
          } else if (/video/.test(normalizedFolderLabel)) {
            if (!incrementSafe(receipt.inventory, "videoFolderCount", 1)) {
              return failure("E_NUMERIC_RANGE", profile);
            }
          }
          if (/(^|\/)final$/.test(normalizedFolderLabel)) {
            receipt.inventory.finalRepresentationFolderOrdinalPlusOne =
              receipt.inventory.pageRowCount;
          } else if (/platform/.test(normalizedFolderLabel)) {
            receipt.inventory.platformRepresentationFolderOrdinalPlusOne =
              receipt.inventory.pageRowCount;
          }
          continue;
        }

        const countKey = {
          video: "videoCount",
          audio: "audioCount",
          image: "imageCount",
          annotation: "annotationCount",
          metadata: "metadataCount",
          readme: "readmeCount",
          certificate: "certificateCount",
          other: "otherCount",
        }[kind];
        if (
          !incrementSafe(receipt.inventory, "fileObjectCount", 1) ||
          !incrementSafe(receipt.inventory, countKey, 1)
        ) {
          return failure("E_NUMERIC_RANGE", profile);
        }

        const byteValue = getInternalBytes(row);
        if (byteValue === "range") {
          return failure("E_NUMERIC_RANGE", profile);
        }
        if (byteValue === null) {
          if (!incrementSafe(receipt.inventory, "byteMissingCount", 1)) {
            return failure("E_NUMERIC_RANGE", profile);
          }
          continue;
        }

        const sizeCellForProvenance = safeQuery(
          row,
          "[data-size-bytes], [data-file-size], .file-size, .size-column, [data-column='size']",
        );
        const exactByteAttribute =
          safeAttribute(row, "data-size-bytes") ||
          safeAttribute(sizeCellForProvenance, "data-size-bytes");
        if (/^\d+$/.test(exactByteAttribute)) {
          if (!incrementSafe(receipt.inventory, "exactByteAttributeCount", 1)) {
            return failure("E_NUMERIC_RANGE", profile);
          }
        } else if (!incrementSafe(receipt.inventory, "displayParsedByteCount", 1)) {
          return failure("E_NUMERIC_RANGE", profile);
        }

        const byteKey = {
          video: "videoKnownBytes",
          audio: "audioKnownBytes",
          image: "imageKnownBytes",
          annotation: "annotationKnownBytes",
          metadata: "metadataKnownBytes",
          readme: "documentKnownBytes",
          certificate: "documentKnownBytes",
          other: "otherKnownBytes",
        }[kind];
        if (
          !incrementSafe(receipt.inventory, "byteKnownCount", 1) ||
          !incrementSafe(receipt.inventory, "totalKnownBytes", byteValue) ||
          !incrementSafe(receipt.inventory, byteKey, byteValue)
        ) {
          return failure("E_NUMERIC_RANGE", profile);
        }
      }
      if (receipt.inventory.pageRowCount === 0) {
        return failure("E_NO_ROWS", profile);
      }

      const scrollCandidates = safeQueryAll(
        pageDocument,
        "main, section, [role='grid'], [role='table'], div",
      );
      let bestScrollRange = 0;
      for (const candidate of scrollCandidates) {
        let scrollHeight = 0;
        let clientHeight = 0;
        let rect = null;
        try {
          scrollHeight = Number(candidate.scrollHeight);
          clientHeight = Number(candidate.clientHeight);
          rect = typeof candidate.getBoundingClientRect === "function"
            ? candidate.getBoundingClientRect()
            : null;
        } catch {
          continue;
        }
        const range = Math.round(scrollHeight - clientHeight);
        if (!Number.isSafeInteger(range) || range <= 10) {
          continue;
        }
        if (!incrementSafe(receipt.inventory, "scrollableContainerCount", 1)) {
          return failure("E_NUMERIC_RANGE", profile);
        }
        if (range > bestScrollRange && rect) {
          const x = Math.max(0, Math.round(Number(rect.left) + Number(rect.width) / 2));
          const y = Math.max(0, Math.round(Number(rect.top) + Number(rect.height) / 2));
          if (
            Number.isSafeInteger(x) &&
            Number.isSafeInteger(y) &&
            Number.isSafeInteger(range)
          ) {
            bestScrollRange = range;
            receipt.inventory.primaryScrollX = x;
            receipt.inventory.primaryScrollY = y;
            receipt.inventory.primaryScrollRange = range;
          }
        }
      }
    }

    if (profile === "combined" || profile === "schema") {
      stageErrorCode = "E_STAGE_SCHEMA";
      const headers = safeQueryAll(
        pageDocument,
        "table thead th, [role='columnheader'], [data-schema-field]",
      );
      if (headers.length === 0) {
        return failure("E_NO_SCHEMA_HEADERS", profile);
      }
      for (const header of headers) {
        const internalHeader = normalizeInternal(
          safeAttribute(header, "data-schema-field") || safeText(header),
        );
        let recognized = false;
        if (/participant|child(?:[_ -]?id)?|subject/.test(internalHeader)) {
          receipt.schemaCategories.participantGrouping = true;
          recognized = true;
        }
        if (/session|recording[_ -]?date|visit|episode/.test(internalHeader)) {
          receipt.schemaCategories.sessionOrDateGrouping = true;
          recognized = true;
        }
        if (/media|video|file|annotation[_ -]?link/.test(internalHeader)) {
          receipt.schemaCategories.mediaAnnotationLinkage = true;
          recognized = true;
        }
        if (/activity/.test(internalHeader)) {
          receipt.schemaCategories.activity = true;
          recognized = true;
        }
        if (/location|place|setting/.test(internalHeader)) {
          receipt.schemaCategories.location = true;
          recognized = true;
        }
        if (/speech|voice[_ -]?presence|vocal/.test(internalHeader)) {
          receipt.schemaCategories.speechPresence = true;
          recognized = true;
        }
        if (/start|end|duration|frame|time|onset|offset/.test(internalHeader)) {
          receipt.schemaCategories.utteranceOrFrameTiming = true;
          recognized = true;
        }
        if (/speaker|voice[_ -]?type|role/.test(internalHeader)) {
          receipt.schemaCategories.speakerRole = true;
          recognized = true;
        }
        if (internalHeader !== "" && !recognized) {
          receipt.schemaCategories.unknownFieldPresent = true;
        }
      }
    }

    stageErrorCode = "E_STAGE_VALIDATE";
    return validateAndReturn(receipt, profile);
  } catch {
    return failure(stageErrorCode, profile);
  }
}

export default childLensKeeperRedactedExtractorV11;
