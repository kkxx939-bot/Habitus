import { createHash, randomUUID } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdir, readFile, readdir, rename, rm, stat, utimes } from "node:fs/promises";
import { join } from "node:path";
import { promisify } from "node:util";
import { replaceJsonDurably } from "./atomic-file.mjs";

const DIR_MODE = 0o700;
const LOCK_STALE_MS = 60_000;
const LOCK_HEARTBEAT_MS = 10_000;
const LOCK_UNVERIFIED_LIVE_MAX_STALE_MS = 300_000;
const execFileAsync = promisify(execFile);

export function sessionKey(host, nativeSessionId) {
  return createHash("sha256").update(`${host}\n${nativeSessionId}`).digest("hex");
}

function paths(root, host, nativeSessionId) {
  const key = sessionKey(host, nativeSessionId);
  const directory = join(root, "sessions", host);
  const lock = join(directory, `${key}.lock`);
  return { key, directory, state: join(directory, `${key}.json`), lock, owner: join(lock, "owner.json") };
}

function validateState(value) {
  if (![2, 3].includes(value?.schemaVersion)) throw new Error("unsupported plugin state schema");
  if (typeof value.nativeSessionId !== "string" || !value.nativeSessionId) throw new Error("plugin state session is invalid");
  if (!Number.isSafeInteger(value.nextSequence) || value.nextSequence < 0) throw new Error("plugin state sequence is invalid");
  if (value.acknowledgedTranscriptCursor != null && ![1, 2].includes(value.acknowledgedTranscriptCursor.schemaVersion)) {
    throw new Error("plugin transcript cursor schema is invalid");
  }
  if (value.schemaVersion === 2) {
    if (!Array.isArray(value.injectionDigests) || value.injectionDigests.some((item) => !/^[0-9a-f]{64}$/.test(item))) {
      throw new Error("plugin injection receipts are invalid");
    }
    const { injectionDigests: _legacy, ...rest } = value;
    return {
      ...rest,
      schemaVersion: 3,
      injectionReceipts: value.injectionDigests.map((digest) => ({ nonce: null, digest })),
    };
  }
  if (!Array.isArray(value.injectionReceipts) || value.injectionReceipts.some((item) => (
    !item
    || (item.nonce !== null && !/^[0-9a-f]{32}$/.test(item.nonce || ""))
    || !/^[0-9a-f]{64}$/.test(item.digest || "")
  ))) {
    throw new Error("plugin injection receipts are invalid");
  }
  // 取过的提醒（方案 32 第 5.1 条，只为去重与运维日志）：可缺省；有就得是 {request_id, claimed_at} 的清单
  if (value.remindersClaimed != null && (!Array.isArray(value.remindersClaimed) || value.remindersClaimed.some((item) => (
    !item || typeof item.request_id !== "string" || typeof item.claimed_at !== "string"
  )))) {
    throw new Error("plugin claimed reminders are invalid");
  }
  return value;
}

export async function readState(root, host, nativeSessionId) {
  const target = paths(root, host, nativeSessionId).state;
  try { return validateState(JSON.parse(await readFile(target, "utf8"))); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
}

export async function stateModifiedAt(root, host, nativeSessionId) {
  try { return (await stat(paths(root, host, nativeSessionId).state)).mtimeMs; }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
}

export async function removeState(root, host, nativeSessionId) {
  await rm(paths(root, host, nativeSessionId).state, { force: true });
}

export async function listStateSessions(root, host) {
  const directory = join(root, "sessions", host);
  let names;
  try { names = await readdir(directory); }
  catch (error) { if (error?.code === "ENOENT") return []; throw error; }
  const sessions = [];
  for (const name of names.filter((value) => /^[0-9a-f]{64}\.json$/.test(value))) {
    try {
      const value = JSON.parse(await readFile(join(directory, name), "utf8"));
      if (typeof value?.nativeSessionId === "string" && value.nativeSessionId) sessions.push(value.nativeSessionId);
    } catch (error) { if (error?.code !== "ENOENT" && !(error instanceof SyntaxError)) throw error; }
  }
  return sessions.sort();
}

export async function writeState(root, host, nativeSessionId, value) {
  const target = paths(root, host, nativeSessionId);
  await mkdir(target.directory, { recursive: true, mode: DIR_MODE });
  await replaceJsonDurably(target.state, { ...value, schemaVersion: 3 });
}

export async function withSessionLock(root, host, nativeSessionId, callback) {
  const target = paths(root, host, nativeSessionId);
  await mkdir(target.directory, { recursive: true, mode: DIR_MODE });
  const deadline = Date.now() + 5000;
  const ownerToken = randomUUID();
  const processIdentity = await readProcessIdentity(process.pid);
  while (true) {
    try {
      await mkdir(target.lock, { mode: DIR_MODE });
      await replaceJsonDurably(target.owner, {
        schemaVersion: 2, ownerToken, pid: process.pid, processIdentity, acquiredAt: new Date().toISOString(),
      });
      break;
    } catch (error) {
      if (error?.code !== "EEXIST") { await discardUninitializedLock(target, ownerToken); throw error; }
      if (await reclaimAbandonedLock(target, ownerToken)) continue;
      if (Date.now() >= deadline) throw new Error("timed out waiting for the plugin session lock");
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
  let heartbeatFailure;
  const heartbeat = setInterval(() => {
    const now = new Date();
    utimes(target.owner, now, now).catch((error) => { heartbeatFailure = error; });
  }, LOCK_HEARTBEAT_MS);
  heartbeat.unref?.();
  try {
    const result = await callback();
    if (heartbeatFailure) throw new Error("plugin session lock heartbeat failed", { cause: heartbeatFailure });
    return result;
  } finally {
    clearInterval(heartbeat);
    await releaseOwnedLock(target, ownerToken);
  }
}

async function discardUninitializedLock(target, ownerToken) {
  const quarantine = `${target.lock}.failed-${ownerToken}`;
  try { await rename(target.lock, quarantine); }
  catch (error) { if (["ENOENT", "EEXIST", "ENOTEMPTY"].includes(error?.code)) return; return; }
  await rm(quarantine, { recursive: true, force: true });
}

async function reclaimAbandonedLock(target, contenderToken) {
  let metadata;
  try { metadata = await stat(target.owner); }
  catch (error) {
    if (error?.code !== "ENOENT") return false;
    try { metadata = await stat(target.lock); } catch { return false; }
  }
  let owner;
  try { owner = JSON.parse(await readFile(target.owner, "utf8")); } catch {}
  const staleAge = Date.now() - metadata.mtimeMs;
  if (owner?.schemaVersion === 2 && processIsAlive(owner.pid)) {
    const currentIdentity = await readProcessIdentity(owner.pid);
    if (currentIdentity !== null && typeof owner.processIdentity === "string" && currentIdentity === owner.processIdentity) return false;
    if (currentIdentity === null && staleAge <= LOCK_UNVERIFIED_LIVE_MAX_STALE_MS) return false;
  } else if (owner?.schemaVersion === 1 && processIsAlive(owner.pid)) {
    if (staleAge <= LOCK_UNVERIFIED_LIVE_MAX_STALE_MS) return false;
  }
  if (staleAge <= LOCK_STALE_MS) return false;
  const quarantine = `${target.lock}.stale-${contenderToken}`;
  try { await rename(target.lock, quarantine); }
  catch (error) { if (["ENOENT", "EEXIST", "ENOTEMPTY"].includes(error?.code)) return false; throw error; }
  await rm(quarantine, { recursive: true, force: true });
  return true;
}

function processIsAlive(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; }
  catch (error) { return error?.code === "EPERM"; }
}

async function readProcessIdentity(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  if (process.platform === "linux") {
    try {
      const value = await readFile(`/proc/${pid}/stat`, "utf8");
      const fields = value.slice(value.lastIndexOf(") ") + 2).trim().split(/\s+/);
      return fields.length > 19 ? `linux:${fields[19]}` : null;
    } catch { return null; }
  }
  if (process.platform === "darwin") {
    try {
      const { stdout } = await execFileAsync("ps", ["-o", "lstart=", "-p", String(pid)], { timeout: 1000, maxBuffer: 4096 });
      const started = stdout.trim();
      return started ? `darwin:${started}` : null;
    } catch { return null; }
  }
  return null;
}

async function releaseOwnedLock(target, ownerToken) {
  let owner;
  try { owner = JSON.parse(await readFile(target.owner, "utf8")); } catch { return; }
  if (![1, 2].includes(owner?.schemaVersion) || owner.ownerToken !== ownerToken) return;
  const quarantine = `${target.lock}.released-${ownerToken}`;
  try { await rename(target.lock, quarantine); }
  catch (error) { if (error?.code === "ENOENT") return; throw error; }
  await rm(quarantine, { recursive: true, force: true });
}
