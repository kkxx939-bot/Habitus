// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { open } from "node:fs/promises";
import { stripMedia } from "./media.mjs";

const CURSOR_SCHEMA = 3;
// v3 游标的前缀校验：文件身份不变、文件不短于游标位置、开头一段与游标前一段的摘要都对得上。宿主改写转写的方式是换文件
// （身份变）、截短（变短）或整份重写（会话头、行内容变，长度随之变，游标前那一段跟着错位）；三样都不用把整份读进来
const ANCHOR_BYTES = 64 * 1024;
const LEGACY_ANCHOR_BYTES = 128;
const READ_CHUNK_BYTES = 4 * 1024 * 1024;

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function identity(metadata) {
  return `${metadata.dev}:${metadata.ino}`;
}

function validateCursor(cursor) {
  if (cursor == null) return null;
  if (
    ![1, 2, CURSOR_SCHEMA].includes(cursor.schemaVersion)
    || !Number.isSafeInteger(cursor.generation)
    || cursor.generation < 0
    || typeof cursor.fileIdentity !== "string"
    || !Number.isSafeInteger(cursor.byteOffset)
    || cursor.byteOffset < 0
    || !Number.isSafeInteger(cursor.lineCount)
    || cursor.lineCount < 0
    || (cursor.schemaVersion === 1 && typeof cursor.anchorDigest !== "string")
    || (cursor.schemaVersion === 2 && typeof cursor.prefixDigest !== "string")
    || (cursor.schemaVersion === CURSOR_SCHEMA && (typeof cursor.headDigest !== "string" || typeof cursor.tailDigest !== "string"))
  ) throw new Error("transcript cursor is invalid");
  return cursor;
}

async function readRange(handle, start, end) {
  const buffer = Buffer.alloc(Math.max(0, end - start));
  let filled = 0;
  while (filled < buffer.length) {
    const { bytesRead } = await handle.read(buffer, filled, buffer.length - filled, start + filled);
    if (bytesRead === 0) break;
    filled += bytesRead;
  }
  return buffer.subarray(0, filled);
}

// 老的 v2 游标记的是整个前缀的摘要：分块流过去算，内存不随文件大小涨（只在升级后的第一次读时走一遍）
async function streamedPrefixDigest(handle, byteOffset) {
  const hash = createHash("sha256");
  for (let start = 0; start < byteOffset; start += READ_CHUNK_BYTES) {
    hash.update(await readRange(handle, start, Math.min(byteOffset, start + READ_CHUNK_BYTES)));
  }
  return hash.digest("hex");
}

async function prefixUnchanged(handle, cursor, head, size) {
  const offset = cursor.byteOffset;
  if (offset > size) return false;
  if (cursor.schemaVersion === 1) {
    return cursor.anchorDigest === digest(await readRange(handle, Math.max(0, offset - LEGACY_ANCHOR_BYTES), offset));
  }
  if (cursor.schemaVersion === 2) return cursor.prefixDigest === await streamedPrefixDigest(handle, offset);
  return cursor.headDigest === digest(head.subarray(0, Math.min(head.length, offset)))
    && cursor.tailDigest === digest(await readRange(handle, Math.max(0, offset - ANCHOR_BYTES), offset));
}

// 从 byteOffset 起分块往后读、逐行解析（末尾没换行的那行不算）。relevant 给了的话，它说不相干的记录不留（记成 null，位置照记），
// 也不占这一次的读取额度。留下的记录先剥掉媒体载荷（media.mjs）再计额度：额度只管剥完留下来的那些，留满就停；截图、附件再大，
// 剥完只剩占位与文字。内存只随留下的记录、一块读缓冲与一行未读完的开头涨，不随文件大小涨。
// 一行原样（不含换行）就超过额度的不解析、不留（剥媒体要先把那一行完整读进来）：记成 null，放进 skipped（第几条、第几行、多少字节），
// 不看它在文件里的什么位置、与对话相不相干；跨块的这种行读过去时只数字节、不攒起来，读到它的换行再接着往后读
async function readCompleteLines(handle, size, byteOffset, lineCount, maxBytes, relevant) {
  const entries = [];
  const skipped = [];
  let lines = lineCount;
  let kept = 0;
  let position = byteOffset;
  let carry = Buffer.alloc(0);
  // 正在越过的超额度行已经读过的字节数（还没读到它的换行）；不在越过时为 null
  let skipping = null;
  const skip = (bytes, endOffset) => {
    lines += 1;
    skipped.push(Object.freeze({ index: entries.length, lineCount: lines, bytes }));
    entries.push(Object.freeze({ record: null, endOffset, lineCount: lines }));
  };
  while (position < size && kept < maxBytes) {
    const chunk = await readRange(handle, position, Math.min(size, position + READ_CHUNK_BYTES));
    if (chunk.length === 0) break;
    let buffer = carry.length > 0 ? Buffer.concat([carry, chunk]) : chunk;
    let bufferStart = position - carry.length;
    position += chunk.length;
    if (skipping != null) {
      const newline = buffer.indexOf(0x0a);
      if (newline < 0) {
        skipping += buffer.length;
        continue;
      }
      const contentEnd = newline > 0 && buffer[newline - 1] === 0x0d ? newline - 1 : newline;
      skip(skipping + contentEnd, bufferStart + newline + 1);
      skipping = null;
      buffer = buffer.subarray(newline + 1);
      bufferStart += newline + 1;
    }
    let start = 0;
    for (let index = buffer.indexOf(0x0a); index >= 0 && kept < maxBytes; index = buffer.indexOf(0x0a, start)) {
      let contentEnd = index;
      if (contentEnd > start && buffer[contentEnd - 1] === 0x0d) contentEnd -= 1;
      if (contentEnd - start > maxBytes) {
        skip(contentEnd - start, bufferStart + index + 1);
        start = index + 1;
        continue;
      }
      const text = buffer.subarray(start, contentEnd).toString("utf8");
      lines += 1;
      let record = null;
      if (text.trim()) {
        try { record = JSON.parse(text); }
        catch (error) { throw new Error(`transcript line ${lines} is invalid JSON`, { cause: error }); }
      }
      if (record != null && (relevant == null || relevant(record))) {
        const stripped = stripMedia(record);
        kept += stripped === record ? index + 1 - start : Buffer.byteLength(JSON.stringify(stripped)) + 1;
        record = stripped;
      } else record = null;
      entries.push(Object.freeze({ record, endOffset: bufferStart + index + 1, lineCount: lines }));
      start = index + 1;
    }
    if (kept >= maxBytes) break;
    // 这一块里剩下的是一行的开头（还没读到它的换行）：留着接下一块；已经比额度还长的不再攒，转成越过它
    carry = Buffer.from(buffer.subarray(start));
    if (carry.length > maxBytes) {
      skipping = carry.length;
      carry = Buffer.alloc(0);
    }
  }
  const end = entries.at(-1)?.endOffset ?? byteOffset;
  return { entries, skipped, more: kept >= maxBytes && end < size };
}

/**
 * 读取已完成轮次的 JSONL 增量：从游标处分块往后读，一次最多留 maxBytes 的记录（只管这一次的增量，不管整个文件多大；
 * 按剥掉媒体载荷之后的大小算，给出的记录也是剥过的；relevant 说不相干的记录不留、不占这个额度）。
 *
 * 游标绑定文件身份、字节位置和前缀锚点；文件轮转、截断或原地重写会开启新 generation，不会把旧文件的行号误用于新 transcript。
 * 末尾未换行的记录永远不会被确认。selectCompleted(records, {more, skipped}) 给出这一段里每一轮结束的位置（boundaries），返回值按轮带上
 * 各自的结束游标，调用方据此在轮的边界拆批；more 为真表示这一次留满了额度、后面还有。skipped 是这一段里原样超过额度、没读的那几行
 * 在 records 里的下标（那几条是 null）：适配器据此决定这几条怎么不影响切轮（比如它可能是某个工具调用的结果）。返回值的 skippedLines
 * 只列游标这一次越过了的那几行（第几行、多少字节），没越过的下一次重读时再报，同一行不会报两次。
 */
export async function readJsonlDelta(path, cursorValue, { maxBytes, selectCompleted, relevant }) {
  if (typeof path !== "string" || !path) return null;
  if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error("maxBytes must be positive");
  if (typeof selectCompleted !== "function") throw new Error("selectCompleted must be a function");
  if (relevant != null && typeof relevant !== "function") throw new Error("relevant must be a function");
  const cursor = validateCursor(cursorValue);
  const noFollow = constants.O_NOFOLLOW || 0;
  let handle;
  try { handle = await open(path, constants.O_RDONLY | noFollow); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
  try {
    const metadata = await handle.stat();
    if (!metadata.isFile()) throw new Error("transcript is not a regular file");
    const size = metadata.size;
    const fileIdentity = identity(metadata);
    const head = await readRange(handle, 0, Math.min(size, ANCHOR_BYTES));
    let generation = cursor?.generation ?? 0;
    let byteOffset = cursor?.byteOffset ?? 0;
    let lineCount = cursor?.lineCount ?? 0;
    const continues = cursor != null && cursor.fileIdentity === fileIdentity && await prefixUnchanged(handle, cursor, head, size);
    if (cursor != null && !continues) {
      generation += 1;
      byteOffset = 0;
      lineCount = 0;
    }
    const { entries, skipped, more } = await readCompleteLines(handle, size, byteOffset, lineCount, maxBytes, relevant);
    if (entries.length === 0) return null;
    // more 告诉适配器这一段停在额度上、不是读到了文件末尾：段尾那条后面还有没读到的记录
    const selected = selectCompleted(entries.map((entry) => entry.record), {
      more, skipped: new Set(skipped.map((line) => line.index)),
    });
    const consumedCount = selected?.consumedCount;
    if (!Number.isSafeInteger(consumedCount) || consumedCount <= 0 || consumedCount > entries.length) {
      if (more) throw new Error(`a transcript turn exceeds the ${maxBytes}-byte read limit`);
      return null;
    }
    const records = [...(selected.records || [])];
    const cursorAt = async (consumed) => {
      const terminal = entries[consumed - 1];
      const end = terminal.endOffset;
      return Object.freeze({
        schemaVersion: CURSOR_SCHEMA,
        generation,
        fileIdentity,
        byteOffset: end,
        lineCount: terminal.lineCount,
        headDigest: digest(head.subarray(0, Math.min(head.length, end))),
        tailDigest: digest(await readRange(handle, Math.max(0, end - ANCHOR_BYTES), end)),
      });
    };
    const boundaries = Array.isArray(selected.boundaries) && selected.boundaries.length > 0
      ? selected.boundaries
      : [{ consumedCount, recordCount: records.length }];
    const turns = [];
    let recordStart = 0;
    for (const boundary of boundaries) {
      turns.push(Object.freeze({
        records: Object.freeze(records.slice(recordStart, boundary.recordCount)),
        nextCursor: await cursorAt(boundary.consumedCount),
      }));
      recordStart = boundary.recordCount;
    }
    const passed = boundaries.at(-1).consumedCount;
    return Object.freeze({
      records: Object.freeze(records),
      turns: Object.freeze(turns),
      startCursor: cursor,
      nextCursor: turns.at(-1).nextCursor,
      afterTurn: selected.afterTurn === true,
      more,
      skippedLines: Object.freeze(skipped
        .filter((line) => line.index < passed)
        .map((line) => Object.freeze({ lineCount: line.lineCount, bytes: line.bytes }))),
    });
  } finally {
    await handle.close();
  }
}

// 转写的第一条记录（Codex 的 session_meta 等）：只读到第一个换行，最多 maxBytes；读不到完整的一行给 null
export async function readFirstRecord(path, maxBytes = 1024 * 1024) {
  if (typeof path !== "string" || !path) return null;
  let handle;
  try { handle = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW || 0)); }
  catch (error) { if (["ENOENT", "ELOOP"].includes(error?.code)) return null; throw error; }
  try {
    const buffer = Buffer.alloc(64 * 1024);
    const chunks = [];
    let total = 0;
    while (total < maxBytes) {
      const { bytesRead } = await handle.read(buffer, 0, Math.min(buffer.length, maxBytes - total), total);
      if (bytesRead === 0) return null;
      const newline = buffer.subarray(0, bytesRead).indexOf(0x0a);
      chunks.push(Buffer.from(buffer.subarray(0, newline < 0 ? bytesRead : newline)));
      total += bytesRead;
      if (newline >= 0) {
        try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch { return null; }
      }
    }
    return null;
  } finally {
    await handle.close();
  }
}
