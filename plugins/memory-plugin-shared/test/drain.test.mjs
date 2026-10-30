// outbox 送出（drain）对服务回话的核对、认领与租约、flush 的前置条件、召回前看 ready。
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { claimOutboxItem, listOutbox } from "../lib/outbox.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";
import { readState, writeState } from "../lib/state-store.mjs";

const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];
const HOST = "test-host";

function cursorAt(byteOffset) {
  return { schemaVersion: 2, generation: 0, fileIdentity: "test:0", byteOffset, lineCount: byteOffset, prefixDigest: "0".repeat(64) };
}

// 每次读转写给出下一轮（偏移 +1）
function adapter() {
  return {
    host: HOST,
    protocol: "codex_rollout",
    nativeSessionId: (input) => input.session_id,
    conversationId: (input) => `test-${input.session_id}`,
    prompt: (input) => input.prompt || "",
    successOutput: () => ({}),
    contextOutput: (context) => ({ context }),
    readTranscriptDelta: async (_input, cursor) => {
      const offset = cursor?.byteOffset || 0;
      return { records: [{ offset }], startCursor: cursor, nextCursor: cursorAt(offset + 1), afterTurn: true };
    },
  };
}

// 服务的游标从 0 起，收下一批就往前推一格
function service(overrides = {}) {
  const calls = { cursor: 0, remember: [], flush: 0, recall: 0 };
  const remote = {
    calls,
    sequence: 0,
    capabilities: async () => ({ ok: true, result: { api_version: "1.0", protocols: ["codex_rollout"], features: FEATURES } }),
    cursor: async () => { calls.cursor += 1; return { ok: true, result: { next_sequence: remote.sequence } }; },
    remember: async (item) => remote.accept(item),
    accept: (item) => {
      calls.remember.push(item.deliveryId);
      remote.sequence = item.startSequence + 1;
      return { ok: true, result: { next_sequence: remote.sequence } };
    },
    flush: async () => { calls.flush += 1; return { ok: true, result: {} }; },
    ready: async () => ({ ok: true, result: { ready: true, recall_ready: true } }),
    recall: async () => { calls.recall += 1; return { ok: true, result: { context: "remembered" } }; },
    ...overrides,
  };
  return remote;
}

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-drain-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

function gate() {
  let open;
  const opened = new Promise((resolve) => { open = resolve; });
  return { opened, open };
}

// 等被测代码去开的门时，同时盯着那次调用本身：它没开门就先返回了（比如服务被判不兼容、根本没送），立刻失败，不在门前一直挂着
function openedBefore(door, operation, what) {
  return Promise.race([
    door.opened,
    operation.then((result) => { throw new Error(`${what} finished without opening the gate: ${JSON.stringify(result)}`); }),
  ]);
}

// 杀变异 PC12（活着的 inflight 被当可回收）、PC13（租约 1 ms）、OB05（claimIsActive 判反）：
// Stop 的 worker 与 SessionStart 的 worker 会同时送同一个会话；一条正在送的，另一路只能等，不能再送一遍
test("two concurrent drains of one session deliver each batch exactly once", { timeout: 20_000 }, async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "concurrent" };
  const started = gate();
  const release = gate();
  const sending = [];
  const remote = service({
    remember: async (item) => {
      sending.push(item.deliveryId);
      started.open();
      await release.opened;
      return remote.accept(item);
    },
  });
  const stopWorker = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: remote });
  const startWorker = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: remote });
  await stopWorker.enqueueCapture(adapter(), input);
  await stopWorker.enqueueCapture(adapter(), input);

  const first = stopWorker.drain(adapter(), input);
  try {
    await openedBefore(started, first, "the first drain");
    const inflight = (await listOutbox(stateRoot, HOST, input.session_id))[0];
    assert.equal(inflight.status, "inflight");
    assert.ok(Date.parse(inflight.claimExpiresAt) - Date.now() >= 25_000, `lease ends at ${inflight.claimExpiresAt}`);
    // 第二路在第一路送的时候进来：应当看到它在送、马上回"待会儿再来"，而不是再送一遍（那样会卡在同一个 remember 上）
    const second = await Promise.race([
      startWorker.recoverPending(adapter(), input),
      new Promise((resolve) => setTimeout(() => resolve("second drain is sending too"), 2_000)),
    ]);
    assert.deepEqual(sending, [inflight.id]);
    assert.equal(second.current.pending, inflight.id);
    assert.equal(second.current.retryable, true);
  } finally {
    release.open();
  }
  assert.equal((await first).drained, 2);
  assert.equal(new Set(sending).size, 2);
  assert.deepEqual(remote.calls.remember, sending);
  assert.deepEqual(await listOutbox(stateRoot, HOST, input.session_id), []);
  assert.equal((await readState(stateRoot, HOST, input.session_id)).nextSequence, 2);
});

// 杀变异 PC16（是否被顶替不比 claimToken）：送到一半租约过期、被另一路重新认领的条目，原先那一路的回话不能拿去 ack
test("a remember answer for a claim that was taken over in the meantime is not acknowledged", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "superseded" };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, {
    service: service({
      remember: async (item) => {
        // 另一路在这期间重新认领了它（换了 claimToken）
        await claimOutboxItem(stateRoot, (await listOutbox(stateRoot, HOST, input.session_id))[0], "other-worker", 60_000);
        return { ok: true, result: { next_sequence: item.startSequence + 1 } };
      },
    }),
  });
  await core.enqueueCapture(adapter(), input);
  const result = await core.drain(adapter(), input);
  assert.equal(result.retryable, true);
  const [item] = await listOutbox(stateRoot, HOST, input.session_id);
  assert.equal(item.status, "inflight");
  assert.equal(item.claimToken, "other-worker");
  assert.equal((await readState(stateRoot, HOST, input.session_id)).nextSequence, 0);
});

// 杀变异 PC14（服务游标后退不拦）：服务说的游标比插件已确认的还小（服务端数据被回滚、换了库），不能拿它去编号重送
test("a service cursor behind the acknowledged plugin state blocks the batch instead of renumbering it", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "cursor-behind" };
  const remote = service({ cursor: async () => ({ ok: true, result: { next_sequence: 3 } }) });
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: remote });
  await core.enqueueCapture(adapter(), input);
  await writeState(stateRoot, HOST, input.session_id, { ...await readState(stateRoot, HOST, input.session_id), nextSequence: 5 });
  const result = await core.drain(adapter(), input);
  const [item] = await listOutbox(stateRoot, HOST, input.session_id);
  assert.equal(result.blocked, item.id);
  assert.equal(item.status, "blocked");
  assert.equal(item.blockedReason, "service cursor moved behind acknowledged plugin state");
  assert.equal(remote.calls.remember.length, 0);
});

// 杀变异 PC15（绑定时不把服务游标写回状态）：绑定那一刻插件就认下服务的游标，送失败了下一次也从这里比
test("binding a batch records the service cursor in the session state", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "bind-cursor" };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, {
    service: service({
      cursor: async () => ({ ok: true, result: { next_sequence: 7 } }),
      remember: async () => ({ ok: false, retryable: true, error: "connection lost" }),
    }),
  });
  await core.enqueueCapture(adapter(), input);
  await core.drain(adapter(), input);
  const [item] = await listOutbox(stateRoot, HOST, input.session_id);
  assert.equal(item.status, "bound");
  assert.equal(item.startSequence, 7);
  assert.equal((await readState(stateRoot, HOST, input.session_id)).nextSequence, 7);
});

// 杀变异 PC18（没往前走的 ack 当成功）：服务回的 next_sequence 不大于这一批的起点，说明它没收下，封住这一批、不前移游标
test("an acknowledgement that does not move the sequence forward blocks the batch", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "flat-ack" };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, {
    service: service({ remember: async (item) => ({ ok: true, result: { next_sequence: item.startSequence } }) }),
  });
  await core.enqueueCapture(adapter(), input);
  const result = await core.drain(adapter(), input);
  const [item] = await listOutbox(stateRoot, HOST, input.session_id);
  assert.equal(result.blocked, item.id);
  assert.equal(item.blockedReason, "service returned an invalid remember acknowledgement");
  const state = await readState(stateRoot, HOST, input.session_id);
  assert.equal(state.nextSequence, 0);
  assert.equal(state.acknowledgedTranscriptCursor, null);
});

// 杀变异 PC08（flush 不看 outbox 是否送空）：还有没送出去的批次时不能让服务封段，否则这些轮次落在段外
test("flush does not ask the service to seal the segment while batches are still queued", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "flush-pending" };
  const remote = service({ remember: async () => ({ ok: false, retryable: true, error: "connection lost" }) });
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: remote });
  await core.enqueueCapture(adapter(), input);
  const flushed = await core.flush(adapter(), input);
  assert.equal(flushed.ok, false);
  assert.equal(flushed.reason, "outbox_not_empty");
  assert.equal(remote.calls.flush, 0);
});

// 杀变异 PC24（召回不看 ready）：服务没就绪（比如向量库没起来）时不召回，这一轮不注入
test("recall stays silent while the service reports it is not ready", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const remote = service({ ready: async () => ({ ok: true, result: { ready: false, recall_ready: false, status: "unhealthy" } }) });
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: remote });
  assert.equal(await core.recall(adapter(), { session_id: "not-ready", prompt: "你好" }), "");
  assert.equal(remote.calls.recall, 0);
});
