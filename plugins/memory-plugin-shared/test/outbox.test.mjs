import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { outboxInventory } from "../doctor.mjs";

import {
  acknowledgeOutboxItem,
  bindOutboxItem,
  blockOutboxItem,
  claimIsActive,
  claimOutboxItem,
  enqueueDelta,
  inventoryOutbox,
  listOutbox,
  listPendingSessions,
  releaseOutboxItem,
} from "../lib/outbox.mjs";
import { sessionKey } from "../lib/state-store.mjs";

function draft(start, end) {
  return {
    host: "test-host",
    nativeSessionId: "session-1",
    conversationId: "conv-1",
    startedOn: "2026-10-19",
    protocol: "normalized",
    payload: { messages: [] },
    occurredAt: "2026-10-19T00:00:00.000Z",
    afterTurn: true,
    transcriptStartCursor: { generation: 1, byteOffset: start },
    transcriptEndCursor: { generation: 1, byteOffset: end },
  };
}

async function acknowledge(root, item) {
  const bound = await bindOutboxItem(root, item, 0);
  const claimed = await claimOutboxItem(root, bound, "claim", 60_000);
  await acknowledgeOutboxItem(root, claimed, "claim");
}

test("acknowledging the last item of a session removes its empty outbox directory", async () => {
  const root = await mkdtemp(join(tmpdir(), "habitus-outbox-"));
  const first = await enqueueDelta(root, draft(0, 10));
  const second = await enqueueDelta(root, draft(10, 20));
  assert.deepEqual(await listPendingSessions(root, "test-host"), ["session-1"]);
  await acknowledge(root, first);
  assert.equal((await readdir(join(root, "outbox", "test-host"))).length, 1); // 还剩一条：目录留着
  await acknowledge(root, second);
  assert.deepEqual(await readdir(join(root, "outbox", "test-host")), []); // 最后一条 ack 了：空目录一起删
  assert.deepEqual(await listPendingSessions(root, "test-host"), []);
  assert.deepEqual(await listOutbox(root, "test-host", "session-1"), []);
  const again = await enqueueDelta(root, draft(20, 30)); // 之后再入队照常建目录
  assert.equal((await listOutbox(root, "test-host", "session-1"))[0].id, again.id);
});

test("the doctor's outbox inventory skips a session directory or an item that an acknowledgement removed mid-listing", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-outbox-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const hostRoot = join(root, "outbox", "test-host");
  await enqueueDelta(root, { ...draft(0, 10), createdAt: new Date().toISOString() });
  // 列到文件之后、读之前被 ack 删掉的条目：指向已不存在文件的 .json
  const vanishedItem = join(hostRoot, sessionKey("test-host", "session-2"));
  await mkdir(vanishedItem, { recursive: true });
  await symlink(join(root, "acknowledged.json"), join(vanishedItem, "0-0.json"));
  // 列出会话之后、读它之前被 ack 连目录一起删掉的会话
  const vanishedSession = join(hostRoot, sessionKey("test-host", "session-3"));
  await mkdir(vanishedSession, { recursive: true });
  const fsPromises = createRequire(import.meta.url)("node:fs/promises");
  const readdirOriginal = fsPromises.readdir;
  fsPromises.readdir = async (path, ...rest) => {
    const entries = await readdirOriginal(path, ...rest);
    if (path === hostRoot) await rm(vanishedSession, { recursive: true, force: true });
    return entries;
  };
  syncBuiltinESMExports();
  try {
    const inventory = await outboxInventory(root);
    assert.equal(inventory.queued, 1);
    assert.equal(inventory.missingState, 1);
  } finally {
    fsPromises.readdir = readdirOriginal;
    syncBuiltinESMExports();
  }
});

function cursorDraft(startGeneration, start, endGeneration, end) {
  return {
    ...draft(start, end),
    transcriptStartCursor: { generation: startGeneration, byteOffset: start },
    transcriptEndCursor: { generation: endGeneration, byteOffset: end },
  };
}

// 杀变异 OB01（文件名里的代数、偏移不补零）与 OB02（listOutbox 倒序）：送的顺序就是转写里的先后，
// 偏移位数不同（9 与 10）、换了一代（压缩或改写后从 0 开始）都不能排乱
test("the outbox lists deliveries in transcript order whatever the digit count or generation", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-outbox-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const later = await enqueueDelta(root, cursorDraft(2, 0, 2, 5));
  const tenth = await enqueueDelta(root, cursorDraft(1, 10, 1, 100));
  const first = await enqueueDelta(root, cursorDraft(1, 0, 1, 9));
  const ninth = await enqueueDelta(root, cursorDraft(1, 9, 1, 10));
  assert.deepEqual((await listOutbox(root, "test-host", "session-1")).map((item) => item.id), [first.id, ninth.id, tenth.id, later.id]);
});

// 杀变异 OB05（claimIsActive 判反）：租约没到期的 inflight 才算有人在送
test("a claim is active only while its lease has not expired", () => {
  const now = Date.parse("2026-10-23T00:00:00Z");
  const inflight = { status: "inflight", claimToken: "a", claimExpiresAt: "2026-10-23T00:00:30Z" };
  assert.equal(claimIsActive(inflight, now), true);
  assert.equal(claimIsActive({ ...inflight, claimExpiresAt: "2026-10-22T23:59:30Z" }, now), false);
  assert.equal(claimIsActive({ ...inflight, status: "bound" }, now), false);
  assert.equal(claimIsActive({ ...inflight, claimExpiresAt: null }, now), false);
});

// 杀变异 OB06 / OB07 / OB10（release / block / ack 不核 claimToken）：租约过期后被别人重新认领的条目，原先那一路回来时
// 不能放回、封住或删掉它
test("release, block and acknowledge refuse a claim token that no longer holds the item", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-outbox-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const bound = await bindOutboxItem(root, await enqueueDelta(root, draft(0, 10)), 0);
  const stale = await claimOutboxItem(root, bound, "first", 60_000);
  const current = await claimOutboxItem(root, stale, "second", 60_000);
  const path = join(root, "outbox", "test-host", sessionKey("test-host", "session-1"), current.filename);
  const before = await readFile(path, "utf8");
  await assert.rejects(releaseOutboxItem(root, current, "first"), /outbox release lost its claim/);
  await assert.rejects(blockOutboxItem(root, current, "rejected", "first"), /outbox claim was superseded/);
  await assert.rejects(acknowledgeOutboxItem(root, current, "first"), /outbox acknowledgement lost its claim/);
  assert.equal(await readFile(path, "utf8"), before);
  assert.equal(JSON.parse(before).claimToken, "second");
});

// 杀变异 OB01 / OB02（盘点时不核条目的格式与会话键）：放错会话目录的条目、格式不对的条目都要报出来，doctor 的 outbox 那一项据此失败
test("the outbox inventory refuses an item filed under another session or with an invalid schema", async (t) => {
  const cases = [
    [(item) => ({ ...item, nativeSessionId: "session-2" }), /outbox session key is invalid/],
    [(item) => ({ ...item, schemaVersion: 99 }), /outbox item schema is invalid/],
    [(item) => ({ ...item, deliveryId: "not-hex" }), /outbox item schema is invalid/],
  ];
  for (const [change, error] of cases) {
    const root = await mkdtemp(join(tmpdir(), "habitus-outbox-"));
    t.after(() => rm(root, { recursive: true, force: true }));
    const item = await enqueueDelta(root, draft(0, 10));
    const path = join(root, "outbox", "test-host", sessionKey("test-host", "session-1"), item.filename);
    const { filename: _filename, ...stored } = item;
    await writeFile(path, JSON.stringify(change(stored)));
    await assert.rejects(inventoryOutbox(root), error);
  }
});

// 杀变异 OB03 / OB12（列条目、列待送会话时不核格式）：格式不对的条目不拿去送，读出来就报错
test("listing a session's deliveries or the sessions with pending deliveries refuses an item with an invalid schema", async (t) => {
  const cases = [
    [(item) => ({ ...item, schemaVersion: 1 }), /invalid plugin outbox item/, /invalid plugin outbox session/],
    [(item) => ({ ...item, deliveryId: "not-hex" }), /invalid plugin outbox item/, null],
    [(item) => ({ ...item, nativeSessionId: 7 }), null, /invalid plugin outbox session/],
  ];
  for (const [change, itemError, sessionError] of cases) {
    const root = await mkdtemp(join(tmpdir(), "habitus-outbox-"));
    t.after(() => rm(root, { recursive: true, force: true }));
    const item = await enqueueDelta(root, draft(0, 10));
    const path = join(root, "outbox", "test-host", sessionKey("test-host", "session-1"), item.filename);
    const { filename: _filename, ...stored } = item;
    await writeFile(path, JSON.stringify(change(stored)));
    if (itemError) await assert.rejects(listOutbox(root, "test-host", "session-1"), itemError);
    if (sessionError) await assert.rejects(listPendingSessions(root, "test-host"), sessionError);
  }
});

// 杀变异 OB11（什么状态都能认领）：只有绑定了序号的（bound，或租约过期的 inflight）能认领；还没绑定序号的、封住的不能
test("only a bound or inflight delivery can be claimed", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "habitus-outbox-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const queued = await enqueueDelta(root, draft(0, 10));
  await assert.rejects(claimOutboxItem(root, queued, "claim", 60_000), /only a bound item can be claimed/);
  const blocked = await blockOutboxItem(root, await bindOutboxItem(root, queued, 0), "rejected");
  await assert.rejects(claimOutboxItem(root, blocked, "claim", 60_000), /only a bound item can be claimed/);
  assert.equal((await listOutbox(root, "test-host", "session-1"))[0].status, "blocked");
  const reclaimed = await claimOutboxItem(root, await bindOutboxItem(root, blocked, 0), "claim", 60_000);
  assert.equal((await claimOutboxItem(root, reclaimed, "again", 60_000)).claimToken, "again");
});
