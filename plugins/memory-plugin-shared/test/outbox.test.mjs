import assert from "node:assert/strict";
import { mkdtemp, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  acknowledgeOutboxItem,
  bindOutboxItem,
  claimOutboxItem,
  enqueueDelta,
  listOutbox,
  listPendingSessions,
} from "../lib/outbox.mjs";

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
