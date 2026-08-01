import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { sessionKey, withSessionLock } from "../lib/state-store.mjs";

test(
  "a live session lock cannot be stolen solely because its mtime is old",
  async (t) => {
    const root = await mkdtemp(join(tmpdir(), "m2bos-plugin-lock-test-"));
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
  const root = await mkdtemp(join(tmpdir(), "m2bos-plugin-lock-reuse-test-"));
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
