// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
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
// Windows 上锁目录里有文件正被打开（别的进程在读 owner.json、杀毒软件在扫）时，改名这个目录会被拒（EPERM / EACCES / EBUSY），
// 稍等就好；POSIX 上这几个码是真的没权限，不重试
const WINDOWS_TRANSIENT_RENAME_CODES = ["EPERM", "EACCES", "EBUSY"];
const RELEASE_RENAME_RETRY_DELAYS_MS = [10, 20, 40, 80, 160];
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
  if (value.acknowledgedTranscriptCursor != null && ![1, 2, 3].includes(value.acknowledgedTranscriptCursor.schemaVersion)) {
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

async function eachState(root, visit) {
  const sessionsRoot = join(root, "sessions");
  let hosts;
  try { hosts = await readdir(sessionsRoot, { withFileTypes: true }); }
  catch (error) { if (error?.code === "ENOENT") return; throw error; }
  for (const host of hosts.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort()) {
    for (const nativeSessionId of await listStateSessions(root, host)) {
      let state;
      try { state = await readState(root, host, nativeSessionId); } catch { continue; }
      if (state != null) visit(host, nativeSessionId, state);
    }
  }
}

// 最近一次读转写失败、之后还没读成功过的会话（plugin-core 的 enqueueCapture 记下 captureFailure）：给 doctor 列
export async function listCaptureFailures(root) {
  const failures = [];
  await eachState(root, (host, nativeSessionId, state) => {
    const failure = state.captureFailure;
    if (failure != null) {
      failures.push({ host, nativeSessionId, reason: String(failure.reason), since: String(failure.since ?? failure.at), at: String(failure.at) });
    }
  });
  return failures;
}

// 原样超过读取额度、跳过没读的转写行（plugin-core 记在 skippedTranscriptLines，每个会话留最近几条）：给 doctor 列
export async function listSkippedTranscriptLines(root) {
  const skipped = [];
  await eachState(root, (host, nativeSessionId, state) => {
    for (const line of Array.isArray(state.skippedTranscriptLines) ? state.skippedTranscriptLines : []) {
      skipped.push({ host, nativeSessionId, lineCount: Number(line?.lineCount), bytes: Number(line?.bytes), at: String(line?.at) });
    }
  });
  return skipped;
}

export async function writeState(root, host, nativeSessionId, value) {
  const target = paths(root, host, nativeSessionId);
  await mkdir(target.directory, { recursive: true, mode: DIR_MODE });
  await replaceJsonDurably(target.state, { ...value, schemaVersion: 3 });
}

// io 只给测试换掉锁目录的改名与平台判断；平时不传
export async function withSessionLock(root, host, nativeSessionId, callback, io = {}) {
  const lockIo = { rename: io.rename ?? rename, platform: io.platform ?? process.platform };
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
        // Windows 没有 /proc 与 ps：记下本进程启动时刻（Node 启动时取的墙钟，之后时钟怎么拨都不变），
        // 争锁方拿内核记录的同一 pid 启动时刻来比；不必每次拿锁都起 PowerShell。
        ...(process.platform === "win32" ? { processStartedAtMs: Math.round(performance.timeOrigin) } : {}),
      });
      break;
    } catch (error) {
      if (error?.code !== "EEXIST") { await discardUninitializedLock(target, ownerToken); throw error; }
      if (await reclaimAbandonedLock(target, ownerToken, lockIo)) continue;
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
    await releaseOwnedLock(target, ownerToken, lockIo);
  }
}

function isWindowsTransientRename(error, platform) {
  return platform === "win32" && WINDOWS_TRANSIENT_RENAME_CODES.includes(error?.code);
}

async function discardUninitializedLock(target, ownerToken) {
  const quarantine = `${target.lock}.failed-${ownerToken}`;
  try { await rename(target.lock, quarantine); }
  catch (error) { if (["ENOENT", "EEXIST", "ENOTEMPTY"].includes(error?.code)) return; return; }
  await rm(quarantine, { recursive: true, force: true });
}

async function reclaimAbandonedLock(target, contenderToken, lockIo) {
  let metadata;
  try { metadata = await stat(target.owner); }
  catch (error) {
    if (error?.code !== "ENOENT") return false;
    try { metadata = await stat(target.lock); } catch { return false; }
  }
  let owner;
  try { owner = JSON.parse(await readFile(target.owner, "utf8")); } catch {}
  const staleAge = Date.now() - metadata.mtimeMs;
  // 心跳仍新鲜的锁不论持有者是谁都不回收；先判它，正常争锁就不必去查进程身份（Windows 上那要起一次 PowerShell）。
  if (staleAge <= LOCK_STALE_MS) return false;
  if (owner?.schemaVersion === 2 && processIsAlive(owner.pid)) {
    const currentIdentity = await readProcessIdentity(owner.pid);
    if (currentIdentity !== null && typeof owner.processIdentity === "string" && currentIdentity === owner.processIdentity) return false;
    // Windows：这个 pid 现在的进程启动时刻与持锁者记下的相同 → 还是它；不同 → pid 被复用、原主已死。
    const verdict = currentIdentity === null ? windowsOwnerVerdict(await readWindowsProcessStart(owner.pid), owner.processStartedAtMs) : null;
    if (verdict === "holder") return false;
    if (currentIdentity === null && verdict === null && staleAge <= LOCK_UNVERIFIED_LIVE_MAX_STALE_MS) return false;
  } else if (owner?.schemaVersion === 1 && processIsAlive(owner.pid)) {
    if (staleAge <= LOCK_UNVERIFIED_LIVE_MAX_STALE_MS) return false;
  }
  const quarantine = `${target.lock}.stale-${contenderToken}`;
  try { await lockIo.rename(target.lock, quarantine); }
  catch (error) {
    // Windows 上一时改不了名同"抢输了"：外层的等锁循环本来就会再试，到截止为止
    if (["ENOENT", "EEXIST", "ENOTEMPTY"].includes(error?.code) || isWindowsTransientRename(error, lockIo.platform)) return false;
    throw error;
  }
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

// 两个时刻都是各自那一刻取的墙钟、之后固定不变，所以时钟回拨不影响比较；容差吸收 Node 启动耗时（内核建进程到
// Node 取 timeOrigin 之间）。没有记录（旧锁、非 Windows 写的锁）或查不到时返回 null，交回 5 分钟的保守上限。
const WINDOWS_START_TOLERANCE_MS = 5000;

export function windowsOwnerVerdict(startedAtMs, recordedStartMs) {
  if (!Number.isFinite(startedAtMs) || !Number.isFinite(recordedStartMs)) return null;
  return Math.abs(startedAtMs - recordedStartMs) <= WINDOWS_START_TOLERANCE_MS ? "holder" : "reused";
}

async function readWindowsProcessStart(pid) {
  if (process.platform !== "win32" || !Number.isSafeInteger(pid) || pid <= 0) return null;
  const script = `[Diagnostics.Process]::GetProcessById(${pid}).StartTime.ToUniversalTime().ToString('o')`;
  try {
    // 按系统目录里的绝对路径起：宿主给 hook 的 PATH 不一定带 System32，也不让 PATH 上同名的别的程序冒充
    const powershell = join(process.env.SystemRoot || "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
    const { stdout } = await execFileAsync(
      powershell,
      ["-NoProfile", "-NonInteractive", "-Command", script],
      { timeout: 10_000, maxBuffer: 4096, windowsHide: true },
    );
    const started = Date.parse(stdout.trim());
    return Number.isFinite(started) ? started : null;
  } catch { return null; }
}

async function releaseOwnedLock(target, ownerToken, lockIo) {
  let owner;
  try { owner = JSON.parse(await readFile(target.owner, "utf8")); } catch { return; }
  if (![1, 2].includes(owner?.schemaVersion) || owner.ownerToken !== ownerToken) return;
  const quarantine = `${target.lock}.released-${ownerToken}`;
  // 放锁只能自己做：Windows 上一时改不了名就退避重试，用完仍不行照样报错（锁留着，等陈旧后由别人回收）
  for (let attempt = 0; ; attempt += 1) {
    try { await lockIo.rename(target.lock, quarantine); break; }
    catch (error) {
      if (error?.code === "ENOENT") return;
      if (!isWindowsTransientRename(error, lockIo.platform) || attempt >= RELEASE_RENAME_RETRY_DELAYS_MS.length) throw error;
      await new Promise((resolve) => setTimeout(resolve, RELEASE_RENAME_RETRY_DELAYS_MS[attempt]));
    }
  }
  await rm(quarantine, { recursive: true, force: true });
}
