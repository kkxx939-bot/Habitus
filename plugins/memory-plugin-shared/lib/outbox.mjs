import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, rmdir } from "node:fs/promises";
import { join } from "node:path";
import { createJsonDurably, replaceJsonDurably, unlinkDurably } from "./atomic-file.mjs";
import { errorText } from "./service-client.mjs";
import { sessionKey } from "./state-store.mjs";

const DIR_MODE = 0o700;
const ITEM_SCHEMA = 2;

function stable(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}`;
}

function directory(root, host, nativeSessionId) {
  return join(root, "outbox", host, sessionKey(host, nativeSessionId));
}

function semanticDraft(draft) {
  return {
    host: draft.host,
    nativeSessionId: draft.nativeSessionId,
    conversationId: draft.conversationId,
    startedOn: draft.startedOn,
    protocol: draft.protocol,
    payload: draft.payload,
    occurredAt: draft.occurredAt,
    afterTurn: draft.afterTurn,
    transcriptStartCursor: draft.transcriptStartCursor,
    transcriptEndCursor: draft.transcriptEndCursor,
  };
}

function cursorOrder(cursor) {
  const generation = Number(cursor?.generation) || 0;
  const offset = Number(cursor?.byteOffset) || 0;
  return `${String(generation).padStart(8, "0")}-${String(offset).padStart(20, "0")}`;
}

function itemPath(root, item) {
  return join(directory(root, item.host, item.nativeSessionId), item.filename);
}

function withoutFilename(item) {
  const value = { ...item };
  delete value.filename;
  return value;
}

export async function enqueueDelta(root, draft) {
  const dir = directory(root, draft.host, draft.nativeSessionId);
  await mkdir(dir, { recursive: true, mode: DIR_MODE });
  const deliveryId = createHash("sha256").update(stable(semanticDraft(draft))).digest("hex");
  const filename = `${cursorOrder(draft.transcriptStartCursor)}-${cursorOrder(draft.transcriptEndCursor)}-${deliveryId}.json`;
  const path = join(dir, filename);
  const item = {
    schemaVersion: ITEM_SCHEMA,
    id: deliveryId,
    deliveryId,
    status: "queued",
    startSequence: null,
    claimToken: null,
    claimExpiresAt: null,
    blockedReason: null,
    ...draft,
  };
  const created = await createJsonDurably(path, item);
  return created ? { ...item, filename } : { ...JSON.parse(await readFile(path, "utf8")), filename };
}

export async function listOutbox(root, host, nativeSessionId) {
  const dir = directory(root, host, nativeSessionId);
  let files;
  try { files = (await readdir(dir)).filter((name) => name.endsWith(".json")).sort(); }
  catch (error) { if (error?.code === "ENOENT") return []; throw error; }
  const items = [];
  for (const filename of files) {
    const item = JSON.parse(await readFile(join(dir, filename), "utf8"));
    if (item?.schemaVersion !== ITEM_SCHEMA || !/^[0-9a-f]{64}$/.test(item?.deliveryId || "")) {
      throw new Error("invalid plugin outbox item");
    }
    items.push({ ...item, filename });
  }
  return items;
}

// 不加锁读：ack 删完最后一个文件时会在会话锁里把目录删掉（26 号清单 7.3），列的这一刻目录或文件可能已经没了，
// 没了的就跳过（下面三个读法都只容 ENOENT）
async function entriesOrNull(path) {
  try { return await readdir(path, { withFileTypes: true }); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
}

async function itemFilesOrNull(directoryPath) {
  try { return (await readdir(directoryPath)).filter((name) => name.endsWith(".json")).sort(); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
}

async function itemOrNull(path) {
  let text;
  try { text = await readFile(path, "utf8"); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
  return JSON.parse(text);
}

export async function listPendingSessions(root, host) {
  const hostRoot = join(root, "outbox", host);
  const sessions = await entriesOrNull(hostRoot);
  if (sessions == null) return [];
  const identities = new Set();
  for (const entry of sessions.filter((value) => value.isDirectory())) {
    const files = await itemFilesOrNull(join(hostRoot, entry.name));
    if (!files?.length) continue;
    const item = await itemOrNull(join(hostRoot, entry.name, files[0]));
    if (item == null) continue;
    if (item?.schemaVersion !== ITEM_SCHEMA || typeof item.nativeSessionId !== "string") {
      throw new Error("invalid plugin outbox session");
    }
    identities.add(item.nativeSessionId);
  }
  return [...identities].sort();
}

// 整个 outbox 的盘点（doctor 用）：按宿主、会话列出每一条还在的条目；条目本身不合规就报错
export async function inventoryOutbox(root) {
  const outbox = join(root, "outbox");
  const sessions = [];
  for (const host of ((await entriesOrNull(outbox)) || []).filter((entry) => entry.isDirectory())) {
    for (const session of ((await entriesOrNull(join(outbox, host.name))) || []).filter((entry) => entry.isDirectory())) {
      const directoryPath = join(outbox, host.name, session.name);
      const items = [];
      for (const filename of (await itemFilesOrNull(directoryPath)) || []) {
        const item = await itemOrNull(join(directoryPath, filename));
        if (item == null) continue;
        if (item?.schemaVersion !== ITEM_SCHEMA || !/^[0-9a-f]{64}$/.test(item?.deliveryId || "")) {
          throw new Error("outbox item schema is invalid");
        }
        if (sessionKey(host.name, item.nativeSessionId) !== session.name) throw new Error("outbox session key is invalid");
        items.push(item);
      }
      if (items.length > 0) sessions.push({ host: host.name, nativeSessionId: items[0].nativeSessionId, items });
    }
  }
  return sessions;
}

export async function bindOutboxItem(root, item, startSequence) {
  if (!Number.isSafeInteger(startSequence) || startSequence < 0) throw new Error("start sequence is invalid");
  const updated = {
    ...withoutFilename(item), status: "bound", startSequence, claimToken: null, claimExpiresAt: null, blockedReason: null,
  };
  await replaceJsonDurably(itemPath(root, item), updated);
  return { ...updated, filename: item.filename };
}

export async function claimOutboxItem(root, item, claimToken, leaseMs) {
  if (typeof claimToken !== "string" || !claimToken) throw new Error("claim token is invalid");
  if (!Number.isSafeInteger(leaseMs) || leaseMs <= 0) throw new Error("claim lease is invalid");
  if (!["bound", "inflight"].includes(item.status)) throw new Error("only a bound item can be claimed");
  const updated = {
    ...withoutFilename(item),
    status: "inflight",
    claimToken,
    claimExpiresAt: new Date(Date.now() + leaseMs).toISOString(),
    blockedReason: null,
  };
  await replaceJsonDurably(itemPath(root, item), updated);
  return { ...updated, filename: item.filename };
}

export async function releaseOutboxItem(root, item, claimToken) {
  if (item.status !== "inflight" || item.claimToken !== claimToken) throw new Error("outbox release lost its claim");
  const updated = {
    ...withoutFilename(item), status: "bound", claimToken: null, claimExpiresAt: null, blockedReason: null,
  };
  await replaceJsonDurably(itemPath(root, item), updated);
  return { ...updated, filename: item.filename };
}

export async function blockOutboxItem(root, item, reason, claimToken = null) {
  if (claimToken != null && item.claimToken !== claimToken) throw new Error("outbox claim was superseded");
  const updated = {
    ...withoutFilename(item),
    status: "blocked",
    claimToken: null,
    claimExpiresAt: null,
    blockedReason: errorText(reason) || "non-retryable service error",
  };
  await replaceJsonDurably(itemPath(root, item), updated);
  return { ...updated, filename: item.filename };
}

export async function acknowledgeOutboxItem(root, item, claimToken) {
  if (item.status !== "inflight" || item.claimToken !== claimToken) throw new Error("outbox acknowledgement lost its claim");
  await unlinkDurably(itemPath(root, item));
  // 删完最后一个文件就把这个会话的空目录删掉（26 号清单 7.3）。调用方持着会话锁，入队也在同一把锁里，不会删到一半有人往里写
  try { await rmdir(directory(root, item.host, item.nativeSessionId)); }
  catch (error) { if (!["ENOTEMPTY", "EEXIST", "ENOENT"].includes(error?.code)) throw error; }
}

export function claimIsActive(item, now = Date.now()) {
  const expiresAt = Date.parse(item.claimExpiresAt);
  return item.status === "inflight" && Number.isFinite(expiresAt) && expiresAt > now;
}
