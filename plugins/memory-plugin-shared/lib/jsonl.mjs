import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { open } from "node:fs/promises";

const CURSOR_SCHEMA = 2;
const ANCHOR_BYTES = 128;

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function identity(metadata) {
  return `${metadata.dev}:${metadata.ino}`;
}

function validateCursor(cursor) {
  if (cursor == null) return null;
  if (
    ![1, CURSOR_SCHEMA].includes(cursor.schemaVersion)
    || !Number.isSafeInteger(cursor.generation)
    || cursor.generation < 0
    || typeof cursor.fileIdentity !== "string"
    || !Number.isSafeInteger(cursor.byteOffset)
    || cursor.byteOffset < 0
    || !Number.isSafeInteger(cursor.lineCount)
    || cursor.lineCount < 0
    || (cursor.schemaVersion === 1 && typeof cursor.anchorDigest !== "string")
    || (cursor.schemaVersion === CURSOR_SCHEMA && typeof cursor.prefixDigest !== "string")
  ) throw new Error("transcript cursor is invalid");
  return cursor;
}

function anchor(raw, byteOffset) {
  return digest(raw.subarray(Math.max(0, byteOffset - ANCHOR_BYTES), byteOffset));
}

function prefixDigest(raw, byteOffset) {
  return digest(raw.subarray(0, byteOffset));
}

function parseCompleteLines(raw, byteOffset, lineCount) {
  const entries = [];
  let start = byteOffset;
  let lines = lineCount;
  for (let index = byteOffset; index < raw.length; index += 1) {
    if (raw[index] !== 0x0a) continue;
    const end = index + 1;
    let contentEnd = index;
    if (contentEnd > start && raw[contentEnd - 1] === 0x0d) contentEnd -= 1;
    const text = raw.subarray(start, contentEnd).toString("utf8");
    lines += 1;
    let record = null;
    if (text.trim()) {
      try { record = JSON.parse(text); }
      catch (error) { throw new Error(`transcript line ${lines} is invalid JSON`, { cause: error }); }
    }
    entries.push(Object.freeze({ record, endOffset: end, lineCount: lines }));
    start = end;
  }
  return entries;
}

/**
 * 读取一个已完成轮次的 JSONL 增量。
 *
 * 游标绑定文件身份、字节位置和前缀锚点；文件轮转、截断或原地重写会开启新 generation，
 * 不会把旧文件的行号误用于新 transcript。末尾未换行的记录永远不会被确认。
 */
export async function readJsonlDelta(path, cursorValue, { maxBytes, selectCompleted }) {
  if (typeof path !== "string" || !path) return null;
  if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error("maxBytes must be positive");
  if (typeof selectCompleted !== "function") throw new Error("selectCompleted must be a function");
  const cursor = validateCursor(cursorValue);
  const noFollow = constants.O_NOFOLLOW || 0;
  let handle;
  try { handle = await open(path, constants.O_RDONLY | noFollow); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
  try {
    const metadata = await handle.stat();
    if (!metadata.isFile() || metadata.size > maxBytes) throw new Error("transcript is not a bounded regular file");
    const raw = await handle.readFile();
    const fileIdentity = identity(metadata);
    let generation = cursor?.generation ?? 0;
    let byteOffset = cursor?.byteOffset ?? 0;
    let lineCount = cursor?.lineCount ?? 0;
    const sameFile = cursor != null && cursor.fileIdentity === fileIdentity;
    const prefixMatches = sameFile
      && byteOffset <= raw.length
      && (
        cursor.schemaVersion === 1
          ? cursor.anchorDigest === anchor(raw, byteOffset)
          : cursor.prefixDigest === prefixDigest(raw, byteOffset)
      );
    if (cursor != null && !prefixMatches) {
      generation += 1;
      byteOffset = 0;
      lineCount = 0;
    }
    const entries = parseCompleteLines(raw, byteOffset, lineCount);
    if (entries.length === 0) return null;
    const selected = selectCompleted(entries.map((entry) => entry.record));
    const consumedCount = selected?.consumedCount;
    if (!Number.isSafeInteger(consumedCount) || consumedCount <= 0 || consumedCount > entries.length) return null;
    const terminal = entries[consumedCount - 1];
    const nextCursor = Object.freeze({
      schemaVersion: CURSOR_SCHEMA,
      generation,
      fileIdentity,
      byteOffset: terminal.endOffset,
      lineCount: terminal.lineCount,
      prefixDigest: prefixDigest(raw, terminal.endOffset),
    });
    return Object.freeze({
      records: Object.freeze([...(selected.records || [])]),
      startCursor: cursor,
      nextCursor,
      afterTurn: selected.afterTurn === true,
    });
  } finally {
    await handle.close();
  }
}
