import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, stat, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { readState, sessionKey, withSessionLock, writeState } from "../lib/state-store.mjs";

test(
  "a live session lock cannot be stolen solely because its mtime is old",
  async (t) => {
    const root = await mkdtemp(join(tmpdir(), "habitus-plugin-lock-test-"));
    t.after(() => rm(root, { recursive: true, force: true }));
    let active = 0;
    let maximumActive = 0;
    let notifyFirstEntered;
    const firstEntered = new Promise((resolve) => { notifyFirstEntered = resolve; });
    let releaseFirst;
    const firstGate = new Promise((resolve) => { releaseFirst = resolve; });

    const first = withSessionLock(root, "test-host", "session", async () => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      notifyFirstEntered();
      await firstGate;
      active -= 1;
    });
    await firstEntered;
    const lock = join(
      root,
      "sessions",
      "test-host",
      `${sessionKey("test-host", "session")}.lock`,
    );
    const stale = new Date(Date.now() - 61_000);
    await utimes(lock, stale, stale);

    let notifySecondEntered;
    const secondEntered = new Promise((resolve) => { notifySecondEntered = resolve; });
    const second = withSessionLock(root, "test-host", "session", async () => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      notifySecondEntered();
      active -= 1;
    });
    const enteredBeforeRelease = await Promise.race([
      secondEntered.then(() => true),
      new Promise((resolve) => setTimeout(() => resolve(false), 100)),
    ]);
    releaseFirst();
    await Promise.all([first, second]);

    assert.equal(enteredBeforeRelease, false);
    assert.equal(maximumActive, 1);
  },
);

test("a stale lock is recoverable when its pid was reused by another process", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-plugin-lock-reuse-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const directory = join(root, "sessions", "test-host");
  const lock = join(directory, `${sessionKey("test-host", "session")}.lock`);
  await mkdir(lock, { recursive: true });
  const owner = join(lock, "owner.json");
  await writeFile(owner, JSON.stringify({
    schemaVersion: 2,
    ownerToken: "crashed-owner",
    pid: process.pid,
    processIdentity: "different-process-start",
    acquiredAt: new Date(0).toISOString(),
  }));
  const stale = new Date(Date.now() - 301_000);
  await utimes(owner, stale, stale);

  let entered = false;
  await withSessionLock(root, "test-host", "session", async () => { entered = true; });

  assert.equal(entered, true);
});

// 一个肯定不在跑的 pid（kill(pid, 0) 报 ESRCH）
const DEAD_PID = 2 ** 22 + 12345;

// 杀变异 SS02（LOCK_STALE_MS = 0）：主人进程已经不在的锁，也要等它旧到一定程度才回收——
// 刚建的锁可能正在写 owner.json，或主人刚崩溃，这时立刻抢会让两个进程同时进锁
test("a fresh lock is not reclaimed even when its owner process is gone", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-plugin-lock-fresh-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const lock = join(root, "sessions", "test-host", `${sessionKey("test-host", "session")}.lock`);
  await mkdir(lock, { recursive: true });
  await writeFile(join(lock, "owner.json"), JSON.stringify({
    schemaVersion: 2, ownerToken: "crashed-owner", pid: DEAD_PID, processIdentity: null, acquiredAt: new Date().toISOString(),
  }));
  let entered = false;
  const waiting = withSessionLock(root, "test-host", "session", async () => { entered = true; });
  await new Promise((resolve) => setTimeout(resolve, 300));
  assert.equal(entered, false);
  await rm(lock, { recursive: true, force: true });
  await waiting;
  assert.equal(entered, true);
});

// 杀变异 SS04（放锁不核 ownerToken）：锁在回调期间被别人回收、重新拿走之后，原来那一路放锁时不能把别人的锁删了
test("releasing a lock leaves it alone when someone else owns it by then", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-plugin-lock-owner-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const lock = join(root, "sessions", "test-host", `${sessionKey("test-host", "session")}.lock`);
  await withSessionLock(root, "test-host", "session", async () => {
    await writeFile(join(lock, "owner.json"), JSON.stringify({
      schemaVersion: 2, ownerToken: "next-owner", pid: process.pid, processIdentity: null, acquiredAt: new Date().toISOString(),
    }));
  });
  assert.equal((await stat(join(lock, "owner.json"))).isFile(), true);
});

// 让 withSessionLock 第一次取的时刻（算等锁截止）照常，之后的时刻都往后跳 10 秒：等锁的截止一下就到，不用真等 5 秒
function jumpClockAfterFirstRead(t) {
  const realNow = Date.now;
  let reads = 0;
  Date.now = () => realNow() + (reads++ === 0 ? 0 : 10_000);
  t.after(() => { Date.now = realNow; });
  return () => { Date.now = realNow; };
}

// 杀变异 SS05（等锁截止 5 秒改成 10 分钟）：主人活着、锁一直被占时，等锁的一方到点报错退出，不陪着挂住宿主的 hook
test("waiting for a session lock that a live owner holds gives up at the deadline", { timeout: 4000 }, async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-plugin-lock-deadline-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  let entered;
  const holding = new Promise((resolve) => { entered = resolve; });
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const holder = withSessionLock(root, "test-host", "session", async () => { entered(); await gate; });
  t.after(async () => { release(); await holder; });
  await holding;
  const restore = jumpClockAfterFirstRead(t);
  const contender = withSessionLock(root, "test-host", "session", async () => "entered");
  await assert.rejects(contender, /timed out waiting for the plugin session lock/);
  restore();
});

// 杀变异 SS07（主人身份核不了时的宽限 300 秒改成 0）：老格式（schemaVersion 1，不记进程身份）的锁，主人进程还活着、
// 锁旧了两分钟（过了 60 秒的一般陈旧线、没过 300 秒）时不回收
test("a lock whose live owner cannot be verified is kept for five minutes, not one", { timeout: 4000 }, async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-plugin-lock-unverified-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const lock = join(root, "sessions", "test-host", `${sessionKey("test-host", "session")}.lock`);
  await mkdir(lock, { recursive: true });
  const owner = join(lock, "owner.json");
  await writeFile(owner, JSON.stringify({ schemaVersion: 1, ownerToken: "old-format-owner", pid: process.pid }));
  const old = new Date(Date.now() - 120_000);
  await utimes(owner, old, old);
  jumpClockAfterFirstRead(t);
  let entered = false;
  await assert.rejects(
    withSessionLock(root, "test-host", "session", async () => { entered = true; }),
    /timed out waiting for the plugin session lock/,
  );
  assert.equal(entered, false);
  assert.equal((await stat(owner)).isFile(), true);
});

// 杀变异 SS06（心跳失败不报）：持锁期间刷不了 owner.json 的时刻（锁可能已被当成陈旧收走），回调的结果不算数，报错
test("a callback whose lock heartbeat failed is reported as failed", { timeout: 4000 }, async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-plugin-lock-heartbeat-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const realSetInterval = globalThis.setInterval;
  globalThis.setInterval = (callback, _ms, ...rest) => realSetInterval(callback, 5, ...rest);
  t.after(() => { globalThis.setInterval = realSetInterval; });
  const owner = join(root, "sessions", "test-host", `${sessionKey("test-host", "session")}.lock`, "owner.json");
  await assert.rejects(
    withSessionLock(root, "test-host", "session", async () => {
      await rm(owner);
      await new Promise((resolve) => setTimeout(resolve, 100));
      return "done";
    }),
    /plugin session lock heartbeat failed/,
  );
});

// 杀变异 SS09 / SS10（会话状态不核取过的提醒、不核会话 id）：格式不对的状态文件读出来就报错，不拿它接着跑
test("a session state with a malformed claimed-reminder list or an empty session id is refused", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-plugin-state-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const valid = { nativeSessionId: "session", nextSequence: 0, acknowledgedTranscriptCursor: null, injectionReceipts: [] };
  await writeState(root, "test-host", "session", { ...valid, remindersClaimed: [{ request_id: "r1", claimed_at: "2026-10-26T00:00:00Z" }] });
  assert.equal((await readState(root, "test-host", "session")).remindersClaimed.length, 1);
  const cases = [
    [{ ...valid, remindersClaimed: "r1" }, /plugin claimed reminders are invalid/],
    [{ ...valid, remindersClaimed: [{ request_id: "r1" }] }, /plugin claimed reminders are invalid/],
    [{ ...valid, nativeSessionId: "" }, /plugin state session is invalid/],
  ];
  for (const [value, error] of cases) {
    await writeState(root, "test-host", "session", value);
    await assert.rejects(readState(root, "test-host", "session"), error);
  }
});

test("windows owner verdict compares the recorded and the live process start times, not the wall clock", async () => {
  const { windowsOwnerVerdict } = await import("../lib/state-store.mjs");
  const recorded = Date.parse("2026-11-07T08:00:00.000Z");
  assert.equal(windowsOwnerVerdict(recorded - 300, recorded), "holder");
  assert.equal(windowsOwnerVerdict(recorded + 4_000, recorded), "holder");
  assert.equal(windowsOwnerVerdict(recorded + 60_000, recorded), "reused");
  assert.equal(windowsOwnerVerdict(recorded - 60_000, recorded), "reused");
  assert.equal(windowsOwnerVerdict(null, recorded), null);
  assert.equal(windowsOwnerVerdict(recorded, undefined), null);
});

async function plantOwner(root, processStartedAtMs) {
  const lock = join(root, "sessions", "test-host", `${sessionKey("test-host", "session")}.lock`);
  await mkdir(lock, { recursive: true });
  const owner = join(lock, "owner.json");
  await writeFile(owner, JSON.stringify({ schemaVersion: 2, ownerToken: "planted", pid: process.pid, processIdentity: null, acquiredAt: new Date().toISOString(), processStartedAtMs }));
  const old = new Date(Date.now() - 600_000);
  await utimes(owner, old, old);
  await utimes(lock, old, old);
}

test(
  "windows: a planted lock whose recorded start differs from the live pid is reclaimed, a matching one is not",
  { skip: process.platform !== "win32" && "Windows 才走进程启动时刻判定" },
  async (t) => {
    const root = await mkdtemp(join(tmpdir(), "habitus-plugin-lock-win-"));
    t.after(() => rm(root, { recursive: true, force: true }));
    await plantOwner(root, Date.parse("2000-01-01T00:00:00.000Z"));
    let entered = false;
    await withSessionLock(root, "test-host", "session", async () => { entered = true; });
    assert.equal(entered, true);

    await plantOwner(root, Math.round(performance.timeOrigin));
    await assert.rejects(
      withSessionLock(root, "test-host", "session", async () => {}),
      /timed out waiting for the plugin session lock/,
    );
  },
);
