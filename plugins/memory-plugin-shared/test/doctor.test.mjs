// doctor 各项检查的判定：哪些情况算失败（✗、退出码 1）、哪些只提示（!）。
import assert from "node:assert/strict";
import { appendFile, cp, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { HARNESS_REGISTRY, createHarnessRegistry } from "../../harnesses.mjs";
import { runPluginDoctor } from "../doctor.mjs";
import { enqueueDelta, listOutbox } from "../lib/outbox.mjs";
import { sessionKey, writeState } from "../lib/state-store.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";
import { hostAdapter as claude } from "../../habitus-memory-claude-code/scripts/host-adapter.mjs";

const PLUGINS_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];
const HOST = "test-host";
const DAY = 24 * 60 * 60 * 1000;

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-doctor-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

function service(features = [...FEATURES, "reminders_v1"]) {
  return {
    health: async () => ({ ok: true, result: { status: "healthy" } }),
    ready: async () => ({ ok: true, result: { ready: true, recall_ready: true, status: "healthy" } }),
    capabilities: async () => ({
      ok: true, result: { api_version: "1.0", service_version: "test", protocols: ["codex_rollout"], features },
    }),
  };
}

// 宿主那几项都换成"通过"，只看要测的那一项
function doctor(stateRoot, overrides = {}) {
  return runPluginDoctor({ stateRoot }, {
    service: service(),
    inspectHarnesses: async () => [],
    inspectFeatures: async () => [],
    inspectHookReviews: async () => [],
    inspectMarketplace: async () => ({ name: "marketplace", status: "pass", detail: "" }),
    ...overrides,
  });
}

function check(report, name) {
  return report.checks.find((item) => item.name === name);
}

// 一个会话、一条 outbox 条目；change 改条目本身，state 为 null 时不写会话状态
async function seed(stateRoot, { change = (item) => item, state = {} } = {}) {
  const item = await enqueueDelta(stateRoot, {
    host: HOST, nativeSessionId: "s1", conversationId: "c1", startedOn: "2026-10-22", protocol: "codex_rollout",
    payload: { records: [{}] }, occurredAt: "2026-10-22T00:00:00.000Z", afterTurn: true,
    transcriptStartCursor: null, transcriptEndCursor: { generation: 0, byteOffset: 1 }, createdAt: new Date().toISOString(),
  });
  const path = join(stateRoot, "outbox", HOST, sessionKey(HOST, "s1"), item.filename);
  const { filename: _filename, ...stored } = item;
  await writeFile(path, JSON.stringify(change(stored)));
  if (state != null) {
    await writeState(stateRoot, HOST, "s1", {
      nativeSessionId: "s1", conversationId: "c1", startedOn: "2026-10-22", protocol: "codex_rollout",
      nextSequence: 0, acknowledgedTranscriptCursor: null, injectionReceipts: [], ...state,
    });
  }
}

// 杀变异 DR01（封住的条目不算失败）、DR07（租约过期的 inflight 不算失败）、DR16（会话状态读不了不算失败）、
// DR02（积压超过一天不提示）：outbox 里送不出去的东西要让 doctor 报出来
test("the doctor fails a blocked item, an expired claim or an unreadable session state, and warns about old backlog", async (t) => {
  const cases = [
    ["blocked", { change: (item) => ({ ...item, status: "blocked", blockedReason: "INVALID_ARGUMENT: nope" }) }, "fail", /blocked_items=1/],
    ["expired claim", {
      change: (item) => ({ ...item, status: "inflight", startSequence: 0, claimToken: "gone", claimExpiresAt: new Date(Date.now() - 60_000).toISOString() }),
    }, "fail", /expiredInflight_items=1/],
    ["unreadable state", { state: { nextSequence: -1 } }, "fail", /invalidState_items=1/],
    ["old backlog", { change: (item) => ({ ...item, createdAt: new Date(Date.now() - 2 * DAY).toISOString() }) }, "warn", /aged_items=1/],
    ["healthy queue", {}, "pass", /queued_items=1/],
  ];
  for (const [label, setup, status, detail] of cases) {
    const stateRoot = await temporaryRoot(t);
    await seed(stateRoot, setup);
    const report = await doctor(stateRoot);
    const outbox = check(report, "outbox");
    assert.equal(outbox.status, status, `${label}: ${outbox.detail}`);
    assert.match(outbox.detail, detail, label);
    assert.equal(report.ok, status !== "fail", label);
  }
});

// 杀变异 DR04（服务没有 reminders_v1 也报通过）：提醒出口不是必需特性，没有就提示一句"只做 memory"
test("the doctor warns when the service has no reminders and passes when it has", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const without = await doctor(stateRoot, { service: service(FEATURES) });
  assert.equal(check(without, "reminders").status, "warn");
  assert.match(check(without, "reminders").detail, /memory only/);
  // 服务只在预测层开着时公布 reminders_v1：没有就说提醒没开、怎么开（裁定 80 第 1 条）
  assert.match(check(without, "reminders").detail, /reminders are off: the service's prediction layer is not enabled/);
  assert.match(check(without, "reminders").detail, /foresight\.enabled: true/);
  assert.equal(without.ok, true);
  assert.equal(check(await doctor(stateRoot), "reminders").status, "pass");
});

// 裁定 80 问题 2：记忆写入在等模型恢复（额度 / 余额、鉴权、断网）不是失败，只提示该看哪里、召回照常；
// Job 真失败待处置、召回本身不可用仍是失败
test("the doctor warns while memory writes wait for the model and fails a failed job or a recall outage", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const readiness = (result) => ({ ...service(), ready: async () => ({ ok: true, status: 503, result }) });
  const queue = (detail) => ({ name: "memory_queue", status: "degraded", detail, critical: true });
  const waiting = await doctor(stateRoot, { service: readiness({
    status: "degraded", ready: false, recall_ready: true,
    checks: [queue("waiting_for_model:3;waits=2;next_attempt_at=2026-10-28T01:00:00+00:00")],
  }) });
  assert.equal(check(waiting, "ready").status, "warn");
  assert.match(check(waiting, "ready").detail, /waiting for the chat model to recover/);
  assert.equal(waiting.ok, true);

  const failed = await doctor(stateRoot, { service: readiness({
    status: "degraded", ready: false, recall_ready: true, checks: [queue("failed:3")],
  }) });
  assert.equal(check(failed, "ready").status, "fail");
  assert.match(check(failed, "ready").detail, /recall still works; memory writes are stopped/);

  const outage = await doctor(stateRoot, { service: readiness({
    status: "unhealthy", ready: false, recall_ready: false,
    checks: [{ name: "memory_vector", status: "unhealthy", detail: "ConnectionError", critical: true }],
  }) });
  assert.equal(check(outage, "ready").status, "fail");
  assert.doesNotMatch(check(outage, "ready").detail, /recall still works/);
});

// 杀变异 DR06（不再比对插件里的共享库副本）：插件里 scripts/shared 的副本与 memory-plugin-shared/lib 不一致，宿主跑的是旧代码
test("the doctor fails when a plugin's copy of the shared library is stale", async (t) => {
  const root = await temporaryRoot(t);
  const copy = join(root, "habitus-memory");
  await cp(join(PLUGINS_ROOT, "habitus-memory"), copy, { recursive: true });
  const codex = HARNESS_REGISTRY.resolve("codex");
  const registry = createHarnessRegistry([{ ...codex, pluginDirectory: relative(PLUGINS_ROOT, copy) }]);
  const fresh = await doctor(root, { registry });
  assert.equal(check(fresh, "shared_bundle").status, "pass", check(fresh, "shared_bundle").detail);
  const [shared] = (await readdir(join(copy, "scripts", "shared"))).filter((name) => name.endsWith(".mjs")).sort();
  await appendFile(join(copy, "scripts", "shared", shared), "// edited by hand\n");
  const stale = await doctor(root, { registry });
  assert.equal(check(stale, "shared_bundle").status, "fail");
  assert.match(check(stale, "shared_bundle").detail, new RegExp(`${shared} is stale`));
  assert.equal(stale.ok, false);
  assert.ok((await readFile(join(copy, "scripts", "shared", shared), "utf8")).endsWith("// edited by hand\n"));
});

// 杀变异 DR15（不查 node 版本）：插件要 Node 18 起（node:test、fetch、AbortController）
test("the doctor fails on a Node runtime older than 18", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const original = Object.getOwnPropertyDescriptor(process.versions, "node");
  Object.defineProperty(process.versions, "node", { ...original, value: "16.20.2" });
  let report;
  try {
    report = await doctor(stateRoot);
  } finally {
    Object.defineProperty(process.versions, "node", original);
  }
  assert.equal(check(report, "node_runtime").status, "fail");
  assert.equal(check(report, "node_runtime").detail, "version=16.20.2; minimum=18");
  assert.equal(report.ok, false);
  assert.equal(check(await doctor(stateRoot), "node_runtime").status, "pass");
});

// R2：转写读不出来（剥完媒体只剩文字仍超读取额度）的会话没进 outbox，每次 Stop 都照样失败：doctor 要报出是哪个会话、为什么；
// 调大额度读成功一次之后不再报
test("the doctor fails a session whose transcript could not be read until a later read succeeds", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = join(stateRoot, "transcript.jsonl");
  await writeFile(transcript, [
    { type: "user", message: { role: "user", content: "x".repeat(1500) } },
    { type: "assistant", message: { role: "assistant", stop_reason: "tool_use", content: [{ type: "tool_use", id: "t1", name: "Bash", input: {} }] } },
    { type: "user", message: { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "y".repeat(1500) }] } },
    { type: "assistant", message: { role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "好" }] } },
  ].map((value) => `${JSON.stringify(value)}\n`).join(""));
  const input = { session_id: "too-long", transcript_path: transcript };
  assert.equal(check(await doctor(stateRoot), "capture").status, "pass");
  await assert.rejects(
    new PluginCore({ stateRoot, maxTranscriptBytes: 2048 }, { service: {} }).enqueueCapture(claude, input),
    /a transcript turn exceeds the 2048-byte read limit/,
  );
  const failed = check(await doctor(stateRoot), "capture");
  assert.equal(failed.status, "fail");
  assert.match(failed.detail, /claude-code session too-long: transcript not read since .*a transcript turn exceeds the 2048-byte read limit/);
  await new PluginCore({ stateRoot, maxTranscriptBytes: 1024 * 1024 }, { service: {} }).enqueueCapture(claude, input);
  assert.equal(check(await doctor(stateRoot), "capture").status, "pass");
});

// T3（裁定 75 第 1 条）：一行原样超过读取额度的跳过：采集照常（这一轮其余的记录进 outbox、游标越过它），操作日志记一条，
// doctor 的 capture 报 warn、写出是哪个会话的第几行；之后再采集同一行不重复记
test("a transcript line over the read limit is skipped, logged once and named by the doctor while capture goes on", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = join(stateRoot, "transcript.jsonl");
  const records = [
    { type: "user", message: { role: "user", content: "截个图" } },
    { type: "assistant", message: { role: "assistant", stop_reason: "tool_use", content: [{ type: "tool_use", id: "t1", name: "Screenshot", input: {} }] } },
    { type: "user", message: { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "y".repeat(3000) }] } },
    { type: "assistant", message: { role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "好" }] } },
  ];
  await writeFile(transcript, records.map((value) => `${JSON.stringify(value)}\n`).join(""));
  const input = { session_id: "big-line", transcript_path: transcript };
  const core = new PluginCore({ stateRoot, maxTranscriptBytes: 2048, operationLogMaxBytes: 1024 * 1024, operationLogKeep: 3 }, { service: {} });
  const captured = await core.enqueueCapture(claude, input);
  assert.equal(captured.enqueued, true);
  assert.equal(captured.state.captureFailure, undefined);
  const [queued] = await listOutbox(stateRoot, "claude-code", "big-line");
  assert.deepEqual(queued.payload.records.map((record) => record.type), ["user", "assistant", "assistant"]);
  const logged = () => readFile(join(stateRoot, "logs", "operations.jsonl"), "utf8")
    .then((text) => text.trim().split("\n").map((line) => JSON.parse(line)));
  const [entry] = await logged();
  assert.equal(entry.stage, "read_transcript");
  assert.equal(entry.status, "skipped");
  assert.equal(entry.retryable, false);
  assert.equal(entry.reason, "line_exceeds_read_limit");
  assert.match(entry.detail, /transcript line 3 \(\d+ bytes\) exceeds the 2048-byte read limit and was skipped/);
  const warned = check(await doctor(stateRoot), "capture");
  assert.equal(warned.status, "warn");
  assert.match(warned.detail, /claude-code session big-line: transcript line 3 \(\d+ bytes\) skipped at .*over the read limit/);
  await appendFile(transcript, `${JSON.stringify({ type: "user", message: { role: "user", content: "再来" } })}\n${JSON.stringify(records[3])}\n`);
  await core.enqueueCapture(claude, input);
  assert.equal((await logged()).length, 1);
});

// T9：同一会话接连读不出来，doctor 写的 "since" 是这一串失败的第一次（从那以后一直没读成），最近一次另写
test("the doctor dates a stuck session from its first failed read, not its latest", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = join(stateRoot, "transcript.jsonl");
  await writeFile(transcript, [
    { type: "user", message: { role: "user", content: "x".repeat(700) } },
    { type: "user", message: { role: "user", content: "x".repeat(700) } },
    { type: "assistant", message: { role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "好" }] } },
  ].map((value) => `${JSON.stringify(value)}\n`).join(""));
  const core = new PluginCore({ stateRoot, maxTranscriptBytes: 1024 }, { service: {} });
  const input = { session_id: "stuck", transcript_path: transcript };
  const attempt = async () => {
    const before = new Date().toISOString();
    await assert.rejects(core.enqueueCapture(claude, input), /read limit/);
    return before;
  };
  const first = await attempt();
  await new Promise((resolve) => setTimeout(resolve, 20));
  const second = await attempt();
  const [, since, last] = /not read since (\S+) \(last tried (\S+)\)/.exec(check(await doctor(stateRoot), "capture").detail);
  assert.ok(since >= first && since < second, `since ${since} is the first failure (${first} .. ${second})`);
  assert.ok(last >= second, `last tried ${last} is the latest failure (after ${second})`);
});

// 杀变异 J10（只有这次真的入队了才清 captureFailure）：调大额度后读成功了、只是还没有一轮做完（没东西入队），会话也不再算卡住
test("a successful read clears a stuck session even when no turn is complete yet", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = join(stateRoot, "transcript.jsonl");
  await writeFile(transcript, [
    { type: "user", message: { role: "user", content: "x".repeat(700) } },
    { type: "user", message: { role: "user", content: "x".repeat(700) } },
    { type: "assistant", message: { role: "assistant", stop_reason: "tool_use", content: [{ type: "tool_use", id: "t1", name: "Bash", input: {} }] } },
  ].map((value) => `${JSON.stringify(value)}\n`).join(""));
  const input = { session_id: "pending-turn", transcript_path: transcript };
  await assert.rejects(new PluginCore({ stateRoot, maxTranscriptBytes: 1024 }, { service: {} }).enqueueCapture(claude, input), /read limit/);
  assert.equal(check(await doctor(stateRoot), "capture").status, "fail");
  const result = await new PluginCore({ stateRoot, maxTranscriptBytes: 1024 * 1024 }, { service: {} }).enqueueCapture(claude, input);
  assert.equal(result.enqueued, false);
  assert.equal(check(await doctor(stateRoot), "capture").status, "pass");
});

// 杀变异 J11（读失败的原因不截断）：记进会话状态的原因最多 512 字，状态文件不随错误消息变大
test("the reason kept for a failed transcript read is bounded to 512 characters", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const adapter = { ...claude, readTranscriptDelta: async () => { throw new Error(`boom ${"x".repeat(4000)}`); } };
  await assert.rejects(new PluginCore({ stateRoot }, { service: {} }).enqueueCapture(adapter, { session_id: "long-reason" }), /boom/);
  const state = JSON.parse(await readFile(join(stateRoot, "sessions", "claude-code", `${sessionKey("claude-code", "long-reason")}.json`), "utf8"));
  assert.equal(state.captureFailure.reason.length, 512);
});

// 杀变异 J12（有一份会话状态读不了，整个 capture 检查就失败）：读不了的那份跳过，别的卡住的会话照样列出来
test("an unreadable session state does not hide the other stuck sessions from the doctor", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const adapter = { ...claude, readTranscriptDelta: async () => { throw new Error("cannot read"); } };
  await assert.rejects(new PluginCore({ stateRoot }, { service: {} }).enqueueCapture(adapter, { session_id: "stuck-1" }), /cannot read/);
  await writeFile(
    join(stateRoot, "sessions", "claude-code", `${sessionKey("claude-code", "broken")}.json`),
    JSON.stringify({ schemaVersion: 99, nativeSessionId: "broken" }),
  );
  const capture = check(await doctor(stateRoot), "capture");
  assert.equal(capture.status, "fail");
  assert.match(capture.detail, /^claude-code session stuck-1: transcript not read since .*: cannot read$/);
});
