/** Build ordered numeric conservative bounds inside the restricted v1.2 flow.
 *
 * This function is not reportable output. It returns only fixed status fields
 * and numeric size pairs aligned to the caller's already-frozen locator order.
 * It never returns locators, labels, names, rows, identifiers, or timestamps.
 */
export function childLensKeeperRestrictedBoundsBuilderV12(input) {
  "use strict";
  const VERSION = "childlens-keeper-restricted-bounds-builder-v1.2.0";
  const LIBRARY_UUID = "c8ed0104-b793-4c35-817e-302afd4e036b";
  const ERRORS = new Set([
    "E_CONFIG", "E_DOCUMENT", "E_IDENTITY", "E_ROWS", "E_MISSING",
    "E_DUPLICATE", "E_SIZE", "E_RANGE", "E_OUTPUT", "E_INTERNAL",
  ]);
  function result(status, errorCode, count, bounds) {
    return {
      status,
      errorCode: ERRORS.has(errorCode) ? errorCode : errorCode === null ? null : "E_INTERNAL",
      builderVersion: VERSION,
      requestedCount: count,
      matchedCount: status === "ok" ? bounds.length : 0,
      bounds: status === "ok" ? bounds : [],
      controls: {
        locatorOrderOnly: true,
        rawTextReturned: false,
        rowReturned: false,
        locatorReturned: false,
        nameReturned: false,
        identifierReturned: false,
        timestampReturned: false,
      },
    };
  }
  function fail(code, count = 0) {
    return result("error", code, count, []);
  }
  function attr(node, key) {
    try {
      const value = node && typeof node.getAttribute === "function" ? node.getAttribute(key) : null;
      return typeof value === "string" ? value : "";
    } catch { return ""; }
  }
  function text(node) {
    try { return node && typeof node.textContent === "string" ? node.textContent : ""; }
    catch { return ""; }
  }
  function one(root, selector) {
    try { return root && typeof root.querySelector === "function" ? root.querySelector(selector) : null; }
    catch { return null; }
  }
  function many(root, selector) {
    try { return root && typeof root.querySelectorAll === "function" ? Array.from(root.querySelectorAll(selector)) : []; }
    catch { return []; }
  }
  function normal(value) {
    return String(value).normalize("NFKC").trim().toLowerCase().split(/[?#]/, 1)[0];
  }
  function base(value) {
    const pieces = normal(value).replace(/\\/g, "/").split("/").filter(Boolean);
    return pieces.length ? pieces[pieces.length - 1] : "";
  }
  function label(row) {
    const direct = attr(row, "data-file-name") || attr(row, "data-name");
    if (direct) return direct;
    const candidate = one(row, "[data-file-name], [data-name], a[download], .file-name, .name-column, [data-column='name'], .sf-table-cell-name a, .sf-table-cell-name, .dirent-name a, .dirent-name, [class*='name'] a, td a[href]");
    return candidate
      ? attr(candidate, "data-file-name") || attr(candidate, "data-name") || attr(candidate, "download") || text(candidate)
      : text(row);
  }
  function sizeSource(row) {
    const exact = attr(row, "data-size-bytes");
    if (exact) return {raw: exact, exact: true};
    const cell = one(row, "[data-size-bytes], [data-file-size], .file-size, .size-column, [data-column='size']");
    if (!cell) return null;
    const cellExact = attr(cell, "data-size-bytes");
    return cellExact
      ? {raw: cellExact, exact: true}
      : {raw: attr(cell, "data-file-size") || text(cell), exact: false};
  }
  function parse(source) {
    const raw = String(source.raw).normalize("NFKC").replace(/\u00a0/g, " ").trim();
    if (source.exact) {
      if (!/^\d+$/.test(raw)) return null;
      const bytes = Number(raw);
      return Number.isSafeInteger(bytes) && bytes > 0 ? {displayBytes: bytes, quantumBytes: 0} : null;
    }
    const match = raw.match(/^(\d+(?:\.(\d+))?)\s*(b|kb|mb|gb|tb|kib|mib|gib|tib)$/i);
    if (!match) return null;
    const multiplier = {b:1,kb:1000,mb:1000**2,gb:1000**3,tb:1000**4,kib:1024,mib:1024**2,gib:1024**3,tib:1024**4}[match[3].toLowerCase()];
    const decimals = match[2] ? match[2].length : 0;
    const displayBytes = Math.round(Number(match[1]) * multiplier);
    const quantumBytes = Math.ceil(multiplier / 10 ** decimals);
    return Number.isSafeInteger(displayBytes) && Number.isSafeInteger(quantumBytes) && displayBytes > 0
      ? {displayBytes, quantumBytes}
      : null;
  }
  try {
    if (!input || typeof input !== "object" || Array.isArray(input) || Object.keys(input).length !== 1 || !Array.isArray(input.sourceLocators) || input.sourceLocators.length < 1 || input.sourceLocators.length > 18) return fail("E_CONFIG");
    if (input.sourceLocators.some(value => typeof value !== "string" || value.length < 1 || value.length > 255 || /^https?:\/\//i.test(value) || !base(value))) return fail("E_CONFIG");
    const ordered = input.sourceLocators.map(base);
    if (new Set(ordered).size !== ordered.length) return fail("E_CONFIG");
    const doc = typeof document === "object" ? document : null;
    if (!doc || typeof doc.querySelectorAll !== "function") return fail("E_DOCUMENT", ordered.length);
    let href = "";
    try { href = typeof location === "object" && typeof location.href === "string" ? location.href : ""; } catch {}
    const identities = [href, ...many(doc, "a[href]").map(node => attr(node, "href"))];
    if (!identities.some(value => normal(value).includes(LIBRARY_UUID))) return fail("E_IDENTITY", ordered.length);
    const rows = [...new Set(many(doc, "[data-childlens-file-row], [data-file-row], table tbody tr, [role='row'][data-file], [role='row'][data-name]"))];
    if (!rows.length) return fail("E_ROWS", ordered.length);
    const byBase = new Map();
    for (const row of rows) {
      const key = base(label(row));
      if (!ordered.includes(key)) continue;
      if (byBase.has(key)) return fail("E_DUPLICATE", ordered.length);
      const parsed = parse(sizeSource(row) || {raw:"",exact:false});
      if (!parsed) return fail("E_SIZE", ordered.length);
      byBase.set(key, parsed);
    }
    if (ordered.some(key => !byBase.has(key))) return fail("E_MISSING", ordered.length);
    const bounds = ordered.map(key => byBase.get(key));
    const serialized = JSON.stringify(result("ok", null, ordered.length, bounds));
    if (typeof serialized !== "string" || serialized.length > 4096) return fail("E_OUTPUT", ordered.length);
    return result("ok", null, ordered.length, bounds);
  } catch {
    return fail("E_INTERNAL");
  }
}
