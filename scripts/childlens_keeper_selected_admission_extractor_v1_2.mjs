/**
 * Fail-closed page-local selected-size admission extractor for ChildLens v1.2.
 *
 * Restricted source locators enter the page context only as matching inputs.
 * They, row labels, filenames, paths, timestamps, and per-item values have no
 * return path. The output is a fixed aggregate receipt.
 */
export function childLensKeeperSelectedAdmissionExtractorV12(input) {
  "use strict";

  const VERSION = "childlens-keeper-selected-admission-extractor-v1.2.0";
  const LIBRARY_UUID = "c8ed0104-b793-4c35-817e-302afd4e036b";
  const OUTPUT_LIMIT = 2048;
  const RAW_CAP = 20 * 1024 ** 3;
  const ADMISSION_CEILING = 18 * 1024 ** 3;
  const PER_OBJECT_OVERHEAD = 16 * 1024 ** 2;
  const ERROR_CODES = new Set([
    "E_CONFIG",
    "E_DOCUMENT_UNAVAILABLE",
    "E_IDENTITY_MISMATCH",
    "E_NO_ROWS",
    "E_SELECTION_INCOMPLETE",
    "E_SELECTION_DUPLICATE",
    "E_SIZE_MISSING",
    "E_NUMERIC_RANGE",
    "E_OUTPUT_KEYS",
    "E_OUTPUT_VALUE",
    "E_OUTPUT_SIZE",
    "E_INTERNAL",
  ]);

  function blank(status, errorCode) {
    return {
      status,
      errorCode,
      extractorVersion: VERSION,
      requestedCount: 0,
      matchedCount: 0,
      missingCount: 0,
      duplicateMatchCount: 0,
      exactByteCount: 0,
      displayParsedCount: 0,
      displayedParsedBytes: 0,
      roundingUpperBoundBytes: 0,
      transferOverheadBytes: 0,
      conservativeAdmissionBytes: 0,
      rawCapBytes: RAW_CAP,
      admissionCeilingBytes: ADMISSION_CEILING,
      admissionCeilingPass: false,
      controls: {
        pageLocalAggregation: true,
        fixedKeyAllowlist: true,
        constantErrorCodes: true,
        fullSelectionRequired: true,
        rawTextReturned: false,
        rowDataReturned: false,
        sourceLocatorReturned: false,
        restrictedIdentifierReturned: false,
        exactTimestampReturned: false,
        perItemSizeReturned: false,
        outputSizeLimitBytes: OUTPUT_LIMIT,
        outputSizeCheckPassed: false,
      },
    };
  }

  function failure(code) {
    return blank("error", ERROR_CODES.has(code) ? code : "E_INTERNAL");
  }

  function sameKeys(value, expected) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      return false;
    }
    const actual = Object.keys(value).sort();
    const wanted = [...expected].sort();
    return actual.length === wanted.length && actual.every((key, index) => key === wanted[index]);
  }

  function safeAttribute(node, name) {
    try {
      const value = node && typeof node.getAttribute === "function" ? node.getAttribute(name) : null;
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

  function safeQuery(root, selector) {
    try {
      return root && typeof root.querySelector === "function" ? root.querySelector(selector) : null;
    } catch {
      return null;
    }
  }

  function safeQueryAll(root, selector) {
    try {
      return root && typeof root.querySelectorAll === "function"
        ? Array.from(root.querySelectorAll(selector))
        : [];
    } catch {
      return [];
    }
  }

  function normalizeInternal(value) {
    return String(value).normalize("NFKC").trim().toLowerCase().split(/[?#]/, 1)[0];
  }

  function basenameInternal(value) {
    const normalized = normalizeInternal(value).replace(/\\/g, "/");
    const pieces = normalized.split("/").filter((piece) => piece !== "");
    return pieces.length ? pieces[pieces.length - 1] : "";
  }

  function getInternalLabel(row) {
    const direct = safeAttribute(row, "data-file-name") || safeAttribute(row, "data-name");
    if (direct) {
      return direct;
    }
    const candidate = safeQuery(
      row,
      "[data-file-name], [data-name], a[download], .file-name, .name-column, [data-column='name'], .sf-table-cell-name a, .sf-table-cell-name, .dirent-name a, .dirent-name, [class*='name'] a, td a[href]",
    );
    return candidate
      ? safeAttribute(candidate, "data-file-name") ||
          safeAttribute(candidate, "data-name") ||
          safeAttribute(candidate, "download") ||
          safeText(candidate)
      : safeText(row);
  }

  function getSizeSource(row) {
    const direct = safeAttribute(row, "data-size-bytes");
    if (direct) {
      return { raw: direct, exact: true };
    }
    const cell = safeQuery(
      row,
      "[data-size-bytes], [data-file-size], .file-size, .size-column, [data-column='size']",
    );
    if (!cell) {
      return null;
    }
    const exact = safeAttribute(cell, "data-size-bytes");
    return exact
      ? { raw: exact, exact: true }
      : { raw: safeAttribute(cell, "data-file-size") || safeText(cell), exact: false };
  }

  function parseSize(source) {
    const raw = String(source.raw).normalize("NFKC").replace(/\u00a0/g, " ").trim();
    if (source.exact) {
      if (!/^\d+$/.test(raw)) {
        return null;
      }
      const bytes = Number(raw);
      return Number.isSafeInteger(bytes) && bytes >= 0
        ? { displayed: bytes, upper: bytes, exact: true }
        : null;
    }
    const match = raw.match(/^(\d+(?:\.(\d+))?)\s*(b|kb|mb|gb|tb|kib|mib|gib|tib)$/i);
    if (!match) {
      return null;
    }
    const multiplier = {
      b: 1,
      kb: 1000,
      mb: 1000 ** 2,
      gb: 1000 ** 3,
      tb: 1000 ** 4,
      kib: 1024,
      mib: 1024 ** 2,
      gib: 1024 ** 3,
      tib: 1024 ** 4,
    }[match[3].toLowerCase()];
    const decimals = match[2] ? match[2].length : 0;
    const displayed = Math.round(Number(match[1]) * multiplier);
    const fullLeastShownUnit = Math.ceil(multiplier / 10 ** decimals);
    const upper = displayed + fullLeastShownUnit;
    return Number.isSafeInteger(displayed) && Number.isSafeInteger(upper) && displayed >= 0
      ? { displayed, upper, exact: false }
      : null;
  }

  function addSafe(receipt, key, amount) {
    const next = receipt[key] + amount;
    if (!Number.isSafeInteger(next) || next < 0) {
      return false;
    }
    receipt[key] = next;
    return true;
  }

  function exactOutputKeys(receipt) {
    return (
      sameKeys(receipt, [
        "status", "errorCode", "extractorVersion", "requestedCount", "matchedCount",
        "missingCount", "duplicateMatchCount", "exactByteCount", "displayParsedCount",
        "displayedParsedBytes", "roundingUpperBoundBytes", "transferOverheadBytes",
        "conservativeAdmissionBytes", "rawCapBytes", "admissionCeilingBytes",
        "admissionCeilingPass", "controls",
      ]) &&
      sameKeys(receipt.controls, [
        "pageLocalAggregation", "fixedKeyAllowlist", "constantErrorCodes",
        "fullSelectionRequired", "rawTextReturned", "rowDataReturned",
        "sourceLocatorReturned", "restrictedIdentifierReturned", "exactTimestampReturned",
        "perItemSizeReturned", "outputSizeLimitBytes", "outputSizeCheckPassed",
      ])
    );
  }

  function validateReturn(receipt) {
    if (!exactOutputKeys(receipt)) {
      return failure("E_OUTPUT_KEYS");
    }
    const allowedStrings = new Set(["ok", "error", VERSION, ...ERROR_CODES]);
    const pending = [receipt];
    while (pending.length) {
      const value = pending.pop();
      if (value === null || typeof value === "boolean") {
        continue;
      }
      if (typeof value === "number") {
        if (!Number.isSafeInteger(value) || value < 0) {
          return failure("E_OUTPUT_VALUE");
        }
        continue;
      }
      if (typeof value === "string") {
        if (!allowedStrings.has(value)) {
          return failure("E_OUTPUT_VALUE");
        }
        continue;
      }
      if (typeof value === "object" && !Array.isArray(value)) {
        pending.push(...Object.values(value));
        continue;
      }
      return failure("E_OUTPUT_VALUE");
    }
    receipt.controls.outputSizeCheckPassed = true;
    const serialized = JSON.stringify(receipt);
    return typeof serialized === "string" && serialized.length <= OUTPUT_LIMIT
      ? receipt
      : failure("E_OUTPUT_SIZE");
  }

  try {
    if (
      !sameKeys(input, ["sourceLocators"]) ||
      !Array.isArray(input.sourceLocators) ||
      input.sourceLocators.length < 1 ||
      input.sourceLocators.length > 18 ||
      input.sourceLocators.some(
        (value) =>
          typeof value !== "string" ||
          value.length < 1 ||
          value.length > 255 ||
          /^https?:\/\//i.test(value) ||
          basenameInternal(value) === "",
      )
    ) {
      return failure("E_CONFIG");
    }
    const selected = input.sourceLocators.map(basenameInternal);
    if (new Set(selected).size !== selected.length) {
      return failure("E_CONFIG");
    }
    const selectedSet = new Set(selected);
    const pageDocument = typeof document === "object" ? document : null;
    if (!pageDocument || typeof pageDocument.querySelectorAll !== "function") {
      return failure("E_DOCUMENT_UNAVAILABLE");
    }
    let href = "";
    try {
      href = typeof location === "object" && typeof location.href === "string" ? location.href : "";
    } catch {
      href = "";
    }
    const identityCandidates = [href, ...safeQueryAll(pageDocument, "a[href]").map((node) => safeAttribute(node, "href"))];
    if (!identityCandidates.some((value) => normalizeInternal(value).includes(LIBRARY_UUID))) {
      return failure("E_IDENTITY_MISMATCH");
    }

    const receipt = blank("ok", null);
    receipt.requestedCount = selected.length;
    const seen = new Map();
    const rows = [...new Set(safeQueryAll(
      pageDocument,
      "[data-childlens-file-row], [data-file-row], table tbody tr, [role='row'][data-file], [role='row'][data-name]",
    ))];
    if (!rows.length) {
      return failure("E_NO_ROWS");
    }
    for (const row of rows) {
      const internalBase = basenameInternal(getInternalLabel(row));
      if (!selectedSet.has(internalBase)) {
        continue;
      }
      const count = (seen.get(internalBase) || 0) + 1;
      seen.set(internalBase, count);
      if (count > 1) {
        receipt.duplicateMatchCount += 1;
        return failure("E_SELECTION_DUPLICATE");
      }
      const source = getSizeSource(row);
      const parsed = source ? parseSize(source) : null;
      if (!parsed) {
        return failure("E_SIZE_MISSING");
      }
      receipt.matchedCount += 1;
      receipt.exactByteCount += parsed.exact ? 1 : 0;
      receipt.displayParsedCount += parsed.exact ? 0 : 1;
      if (
        !addSafe(receipt, "displayedParsedBytes", parsed.displayed) ||
        !addSafe(receipt, "roundingUpperBoundBytes", parsed.upper)
      ) {
        return failure("E_NUMERIC_RANGE");
      }
    }
    receipt.missingCount = receipt.requestedCount - receipt.matchedCount;
    if (receipt.missingCount !== 0) {
      return failure("E_SELECTION_INCOMPLETE");
    }
    const percentOverhead = Math.ceil(receipt.roundingUpperBoundBytes / 100);
    const fixedOverhead = PER_OBJECT_OVERHEAD * receipt.requestedCount;
    receipt.transferOverheadBytes = percentOverhead + fixedOverhead;
    receipt.conservativeAdmissionBytes = receipt.roundingUpperBoundBytes + receipt.transferOverheadBytes;
    if (!Number.isSafeInteger(receipt.conservativeAdmissionBytes)) {
      return failure("E_NUMERIC_RANGE");
    }
    receipt.admissionCeilingPass = receipt.conservativeAdmissionBytes <= ADMISSION_CEILING;
    return validateReturn(receipt);
  } catch {
    return failure("E_INTERNAL");
  }
}
