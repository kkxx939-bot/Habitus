// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
import { createHash } from "node:crypto";
import { mkdir, readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { createJsonDurably, replaceJsonDurably, unlinkDurably } from "./atomic-file.mjs";
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

export async function listPendingSessions(root, host) {
  const hostRoot = join(root, "outbox", host);
  let sessions;
  try { sessions = await readdir(hostRoot, { withFileTypes: true }); }
  catch (error) { if (error?.code === "ENOENT") return []; throw error; }
  const identities = new Set();
  for (const entry of sessions.filter((value) => value.isDirectory())) {
    const files = (await readdir(join(hostRoot, entry.name))).filter((name) => name.endsWith(".json")).sort();
    if (files.length === 0) continue;
    const item = JSON.parse(await readFile(join(hostRoot, entry.name, files[0]), "utf8"));
    if (item?.schemaVersion !== ITEM_SCHEMA || typeof item.nativeSessionId !== "string") {
      throw new Error("invalid plugin outbox session");
    }
    identities.add(item.nativeSessionId);
  }
  return [...identities].sort();
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
    blockedReason: String(reason || "non-retryable service error"),
  };
  await replaceJsonDurably(itemPath(root, item), updated);
  return { ...updated, filename: item.filename };
}

export async function acknowledgeOutboxItem(root, item, claimToken) {
  if (item.status !== "inflight" || item.claimToken !== claimToken) throw new Error("outbox acknowledgement lost its claim");
  await unlinkDurably(itemPath(root, item));
}

export function claimIsActive(item, now = Date.now()) {
  const expiresAt = Date.parse(item.claimExpiresAt);
  return item.status === "inflight" && Number.isFinite(expiresAt) && expiresAt > now;
}
