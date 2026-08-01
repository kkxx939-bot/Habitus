import { createHash } from "node:crypto";

const END_MARKER = "</m2bos-memory-context>";
const NONCE = /^[0-9a-f]{32}$/;
const DIGEST = /^[0-9a-f]{64}$/;

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function normalizeReceipts(values) {
  if (!Array.isArray(values)) return [];
  return values.filter((value) => (
    value
    && (value.nonce === null || NONCE.test(value.nonce))
    && DIGEST.test(value.digest)
  ));
}

function stripText(value, receipts) {
  let result = value;
  let removed = false;
  for (const receipt of receipts) {
    if (receipt.nonce === null) {
      if (digest(result) === receipt.digest) {
        result = "";
        removed = true;
      }
      continue;
    }
    const startMarker = `<m2bos-memory-context receipt="${receipt.nonce}">`;
    let offset = 0;
    while (offset < result.length) {
      const start = result.indexOf(startMarker, offset);
      if (start < 0) break;
      const endStart = result.indexOf(END_MARKER, start + startMarker.length);
      if (endStart < 0) break;
      const end = endStart + END_MARKER.length;
      const candidate = result.slice(start, end);
      if (digest(candidate) === receipt.digest) {
        result = result.slice(0, start) + result.slice(end);
        removed = true;
        offset = start;
      } else {
        offset = start + startMarker.length;
      }
    }
  }
  return { value: result, removed };
}

export function stripInjectionReceipts(value, receiptValues) {
  const receipts = normalizeReceipts(receiptValues);
  if (receipts.length === 0) return { value, removed: false };
  if (typeof value === "string") return stripText(value, receipts);
  if (Array.isArray(value)) {
    let removed = false;
    const result = value.map((item) => {
      const stripped = stripInjectionReceipts(item, receipts);
      removed ||= stripped.removed;
      return stripped.value;
    });
    return { value: result, removed };
  }
  if (value && typeof value === "object") {
    let removed = false;
    const result = {};
    for (const [key, item] of Object.entries(value)) {
      const stripped = stripInjectionReceipts(item, receipts);
      removed ||= stripped.removed;
      result[key] = stripped.value;
    }
    return { value: result, removed };
  }
  return { value, removed: false };
}

export function hasMeaningfulMessageContent(value) {
  if (typeof value === "string") return Boolean(value.trim());
  if (!Array.isArray(value)) return value != null;
  return value.some((item) => {
    if (typeof item === "string") return Boolean(item.trim());
    if (!item || typeof item !== "object") return item != null;
    if (typeof item.text === "string") return Boolean(item.text.trim());
    return !["input_text", "output_text", "text"].includes(item.type);
  });
}

export function requireInjectionReceipt(value) {
  if (!value || !NONCE.test(value.nonce || "") || !DIGEST.test(value.digest || "")) {
    throw new Error("injection receipt is invalid");
  }
  return Object.freeze({ nonce: value.nonce, digest: value.digest });
}
