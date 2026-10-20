import assert from "node:assert/strict";
import { mkdtemp, rm, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { hostAdapter } from "../../habitus-memory-claude-code/scripts/host-adapter.mjs";
import { loadPluginConfig } from "../lib/config.mjs";
import { enqueueDelta } from "../lib/outbox.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";
import { readState, sessionKey, writeState } from "../lib/state-store.mjs";

const HOUR = 3_600_000;

function core(stateRoot) {
  return new PluginCore(loadPluginConfig({ HABITUS_PLUGIN_STATE_DIR: stateRoot }), { service: {} });
}

async function seed(stateRoot, nativeSessionId, transcriptPath, hoursAgo) {
  const state = {
    nativeSessionId,
    conversationId: `claude-${nativeSessionId}`,
    startedOn: "2026-10-01",
    protocol: hostAdapter.protocol,
    nextSequence: 0,
    acknowledgedTranscriptCursor: null,
    injectionReceipts: [],
    ...(transcriptPath ? { transcriptPath } : {}),
  };
  await writeState(stateRoot, hostAdapter.host, nativeSessionId, state);
  const when = new Date(Date.now() - hoursAgo * HOUR);
  const path = join(stateRoot, "sessions", hostAdapter.host, `${sessionKey(hostAdapter.host, nativeSessionId)}.json`);
  await utimes(path, when, when);
}

test("the transcript path given by the host adapter is kept in the session state", async () => {
  const stateRoot = await mkdtemp(join(tmpdir(), "habitus-retire-path-"));
  const transcript = join(stateRoot, "transcript.jsonl");
  const input = { session_id: "s1", agent_transcript_path: transcript };
  await core(stateRoot).recordInjection(hostAdapter, input, { nonce: "a".repeat(32), digest: "b".repeat(64) });
  assert.equal((await readState(stateRoot, hostAdapter.host, "s1")).transcriptPath, transcript);
});

test("a session state retires only when its transcript is gone, its outbox is empty and the grace period has passed", async () => {
  const stateRoot = await mkdtemp(join(tmpdir(), "habitus-retire-"));
  const present = join(stateRoot, "present.jsonl");
  await writeFile(present, "");
  const gone = join(stateRoot, "gone.jsonl");
  await seed(stateRoot, "retires", gone, 48);
  await seed(stateRoot, "still-writing", present, 48); // transcript 还在
  await seed(stateRoot, "just-now", gone, 1); // transcript 暂时不在、宽限期内：不删
  await seed(stateRoot, "legacy", null, 48); // 老的状态没存路径：不删
  await seed(stateRoot, "queued", gone, 48);
  await enqueueDelta(stateRoot, {
    host: hostAdapter.host,
    nativeSessionId: "queued",
    conversationId: "claude-queued",
    startedOn: "2026-10-01",
    protocol: hostAdapter.protocol,
    payload: { messages: [] },
    occurredAt: "2026-10-01T00:00:00.000Z",
    afterTurn: true,
    transcriptStartCursor: { generation: 1, byteOffset: 0 },
    transcriptEndCursor: { generation: 1, byteOffset: 10 },
  });
  const { retired } = await core(stateRoot).retireIdleSessions(hostAdapter);
  assert.deepEqual(retired, ["retires"]);
  assert.equal(await readState(stateRoot, hostAdapter.host, "retires"), null);
  for (const kept of ["still-writing", "just-now", "legacy", "queued"]) {
    assert.ok(await readState(stateRoot, hostAdapter.host, kept), `${kept} kept`);
  }
  await rm(present);
  assert.deepEqual((await core(stateRoot).retireIdleSessions(hostAdapter)).retired, ["still-writing"]);
});
