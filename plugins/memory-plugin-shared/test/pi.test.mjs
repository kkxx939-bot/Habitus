import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

import HabitusMemoryExtension from "../../habitus-memory-pi/index.mjs";
import { hostAdapter } from "../../habitus-memory-pi/scripts/host-adapter.mjs";
import { QUESTION_TOOL, questionTool } from "../../habitus-memory-pi/scripts/question-tool.mjs";
import { HARNESS_REGISTRY } from "../../harnesses.mjs";
import { executePluginLifecycle } from "../../install-memory-plugin.mjs";
import { fileEntryPath } from "../../registrations.mjs";
import { createContextInjection } from "../lib/hook-runner.mjs";
import { requireHostAdapter } from "../lib/host-adapter.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";

const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-pi-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

function fakeService({ reminders = [], context = "", remembered = [] } = {}) {
  return {
    capabilities: async () => ({ ok: true, result: { api_version: "1.0", protocols: ["pi"], features: [...FEATURES, "reminders_v1"] } }),
    ready: async () => ({ ok: true, result: { ready: true, recall_ready: true } }),
    recall: async () => ({ ok: true, result: { context } }),
    claimReminders: async () => ({ ok: true, result: { reminders } }),
    cursor: async () => ({ ok: true, result: { next_sequence: 0 } }),
    remember: async (item) => {
      remembered.push(item);
      return { ok: true, result: { next_sequence: item.startSequence + item.payload.records.length * 3 } };
    },
    flush: async () => ({ ok: true, result: { jobs: [], consistency: [] } }),
  };
}

// pi 的 ExtensionAPI 只用到 registerTool 与 on
function fakePi() {
  const handlers = new Map();
  const tools = [];
  return {
    tools,
    registerTool: (tool) => tools.push(tool),
    on: (event, handler) => handlers.set(event, handler),
    emit: (event, payload, ctx) => handlers.get(event)?.(payload, ctx),
  };
}

function session(file, id = "019a-session") {
  return { sessionManager: { getSessionFile: () => file, getSessionId: () => id } };
}

const line = (value) => `${JSON.stringify(value)}\n`;

test("the pi host adapter satisfies the shared contract and names its own question tool", () => {
  assert.equal(requireHostAdapter(hostAdapter), hostAdapter);
  assert.equal(hostAdapter.conversationId({ session_id: "s1" }), "pi-s1");
  assert.equal(hostAdapter.questionTool, QUESTION_TOOL);
});

test("before_agent_start injects memory and the reminder block as one hidden custom message", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20, reminderClaimLimit: 3 };
  const reminder = { request_id: "e1", kind: "写周报", text: "周四了，周报还没写。", options: ["现在就做", "这次不用", "知道了", "已经做过了"], claimed_at: "2026-10-22T15:52:10+08:00" };
  const pi = fakePi();
  HabitusMemoryExtension(pi, { config, core: new PluginCore(config, { service: fakeService({ context: "用户偏好简洁回答。", reminders: [reminder] }) }) });
  assert.deepEqual(pi.tools.map((tool) => tool.name), [QUESTION_TOOL]);
  const file = join(stateRoot, "session.jsonl");
  const result = await pi.emit("before_agent_start", { messageType: "user", prompt: "帮我看看这个报错" }, session(file));
  assert.equal(result.message.customType, "habitus-memory");
  assert.equal(result.message.display, false);
  assert.match(result.message.content, /<habitus-memory-context receipt="[0-9a-f]{32}">/);
  assert.match(result.message.content, /<habitus-reminder receipt="[0-9a-f]{32}">/);
  assert.match(result.message.content, /ask_user/);
  // 不落盘的进程（-p --no-session 的子代理）、不是人说话起的这一轮：什么都不做
  assert.equal(await pi.emit("before_agent_start", { messageType: "user", prompt: "x" }, session(undefined)), undefined);
  assert.equal(await pi.emit("before_agent_start", { messageType: "custom", message: {} }, session(file)), undefined);
});

test("agent_settled sends pi's own session file up to the last finished assistant message", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  const remembered = [];
  const pi = fakePi();
  HabitusMemoryExtension(pi, { config, core: new PluginCore(config, { service: fakeService({ remembered }) }) });
  const file = join(stateRoot, "session.jsonl");
  await writeFile(file, [
    line({ type: "session", version: 3, id: "019a-session", timestamp: "2026-10-22T07:00:00Z", cwd: "/w" }),
    line({ type: "message", id: "b", timestamp: "2026-10-22T07:00:01Z", message: { role: "user", content: "看报错", timestamp: 1 } }),
    line({ type: "message", id: "c", timestamp: "2026-10-22T07:00:02Z", message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "好" }], timestamp: 2 } }),
    line({ type: "message", id: "d", timestamp: "2026-10-22T07:00:03Z", message: { role: "user", content: "再看看", timestamp: 3 } }),
    line({ type: "message", id: "e", timestamp: "2026-10-22T07:00:04Z", message: { role: "assistant", stopReason: "toolUse", content: [{ type: "toolCall", id: "t1", name: "read", arguments: {} }], timestamp: 4 } }),
  ].join(""));
  await pi.emit("agent_settled", {}, session(file));
  assert.equal(remembered.length, 1);
  assert.equal(remembered[0].protocol, "pi");
  assert.equal(remembered[0].conversationId, "pi-019a-session");
  // 读到最后一条说完了的助手消息为止：还在跑工具（toolUse）的那一段留到下一轮
  assert.deepEqual(remembered[0].payload.records.filter((item) => item.type === "message").map((item) => item.id), ["b", "c"]);
  await pi.emit("session_shutdown", {}, session(file));
});

// Q3：pi 的子代理是另起的不落盘进程（-p --no-session）：一轮跑完、压缩前、结束时都什么都不送
test("a pi process without a session file (a subagent) sends nothing when it settles, compacts or shuts down", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  const calls = [];
  const service = new Proxy(fakeService(), { get: (target, name) => (...args) => { calls.push(String(name)); return target[name](...args); } });
  const pi = fakePi();
  HabitusMemoryExtension(pi, { config, core: new PluginCore(config, { service }) });
  for (const event of ["agent_settled", "session_before_compact", "session_shutdown"]) await pi.emit(event, {}, session(undefined));
  assert.equal(await pi.emit("before_agent_start", { messageType: "user", prompt: "去查" }, session(undefined)), undefined);
  assert.deepEqual(calls, []);
});

test("an error answer that pi retries is not where a turn ends: nothing goes out before the overflow compaction", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { maxTranscriptBytes: 1 << 20 };
  const file = join(stateRoot, "session.jsonl");
  const message = (id, body) => line({ type: "message", id, message: body });
  // 他说 u1 → 工具 → 再调模型时上下文溢出：pi 写 context_edit 拿掉 a2、触发 session_before_compact，压缩后 agent.continue()
  await writeFile(file, [
    line({ type: "session", version: 3, id: "s" }),
    message("u1", { role: "user", content: [{ type: "text", text: "把测试全跑一遍" }] }),
    message("a1", { role: "assistant", stopReason: "toolUse", content: [{ type: "toolCall", id: "c1", name: "bash" }] }),
    message("r1", { role: "toolResult", toolCallId: "c1", content: [{ type: "text", text: "..." }] }),
    message("a2", { role: "assistant", stopReason: "error", errorMessage: "prompt is too long", content: [] }),
    line({ type: "context_edit", id: "e1", targetId: "a2", replacement: null }),
  ].join(""));
  assert.equal(await hostAdapter.readTranscriptDelta({ transcript_path: file }, null, config, []), null);
  await writeFile(file, [
    line({ type: "compaction", id: "k1", summary: "..." }),
    message("a3", { role: "assistant", stopReason: "toolUse", content: [{ type: "toolCall", id: "c2", name: "bash" }] }),
    message("r2", { role: "toolResult", toolCallId: "c2", content: [{ type: "text", text: "ok" }] }),
    message("a4", { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "测试全过了" }] }),
  ].join(""), { flag: "a" });
  const delta = await hostAdapter.readTranscriptDelta({ transcript_path: file }, null, config, []);
  assert.deepEqual(delta.records.filter((item) => item.type === "message").map((item) => item.id), ["u1", "a1", "r1", "a2", "a3", "r2", "a4"]);
});

test("an error answer pi does not retry ends the turn", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const file = join(stateRoot, "session.jsonl");
  await writeFile(file, [
    line({ type: "session", version: 3, id: "s" }),
    line({ type: "message", id: "u1", message: { role: "user", content: [{ type: "text", text: "看报错" }] } }),
    line({ type: "message", id: "a1", message: { role: "assistant", stopReason: "error", errorMessage: "401 unauthorized", content: [] } }),
  ].join(""));
  const delta = await hostAdapter.readTranscriptDelta({ transcript_path: file }, null, { maxTranscriptBytes: 1 << 20 }, []);
  assert.deepEqual(delta.records.filter((item) => item.type === "message").map((item) => item.id), ["u1", "a1"]);
});

test("ask_user returns one line per question, lets the user type, and fails when dismissed or headless", async () => {
  const tool = questionTool();
  const answers = ["现在就做", "自己填…"];
  const ui = { select: async () => answers.shift(), input: async () => "下周再弄" };
  const result = await tool.execute("call", {
    questions: [
      { question: "周报还没写。", options: [{ label: "现在就做" }, { label: "这次不用" }] },
      { question: "备份呢？", options: [{ label: "现在就做" }] },
    ],
  }, undefined, undefined, { hasUI: true, ui });
  assert.equal(result.content[0].text, '"周报还没写。"="现在就做"\n"备份呢？"="下周再弄"');
  assert.deepEqual(result.details.answers, ["现在就做", "下周再弄"]);
  const params = { questions: [{ question: "q", options: [{ label: "a" }] }] };
  await assert.rejects(tool.execute("c", params, undefined, undefined, { hasUI: true, ui: { select: async () => undefined } }), /dismissed/);
  await assert.rejects(tool.execute("c", params, undefined, undefined, { hasUI: false }), /No interactive UI/);
});

test("installing for pi writes the managed entry under ~/.pi/agent/extensions", async (t) => {
  const root = await temporaryRoot(t);
  const env = { HOME: join(root, "home") };
  await mkdir(join(env.HOME, ".pi"), { recursive: true });
  const noCli = { available: () => false, json: () => { throw new Error("no CLI"); }, run: () => { throw new Error("no CLI"); } };
  const marketplace = join(root, "marketplace");
  const installed = await executePluginLifecycle({ action: "install", harnesses: ["pi"], root: marketplace }, { runner: noCli, env });
  assert.equal(installed.harnesses[0].installed, true);
  const entryPath = fileEntryPath(HARNESS_REGISTRY.resolve("pi"), env);
  assert.equal(entryPath, join(env.HOME, ".pi", "agent", "extensions", "habitus-memory.js"));
  const entry = await readFile(entryPath, "utf8");
  const target = pathToFileURL(join(marketplace, "plugins", "habitus-memory-pi", "index.mjs")).href;
  assert.ok(entry.includes(JSON.stringify(target)));
  assert.equal(typeof (await import(target)).default, "function");
});

// 杀变异 PI02（pending 的答复也算说完）：还在流式写的助手消息不是一轮的结束，不能在它后面切（裁定 69）
test("a pending assistant message is not the end of a turn", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const file = join(stateRoot, "session.jsonl");
  const config = { maxTranscriptBytes: 1 << 20 };
  await writeFile(file, [
    line({ type: "message", id: "u1", message: { role: "user", content: "看报错" } }),
    line({ type: "message", id: "a1", message: { role: "assistant", stopReason: "pending", content: [{ type: "text", text: "我先看" }] } }),
  ].join(""));
  assert.equal(await hostAdapter.readTranscriptDelta({ transcript_path: file }, null, config, []), null);
  await writeFile(file, line({ type: "message", id: "a2", message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "好了" }] } }), { flag: "a" });
  const delta = await hostAdapter.readTranscriptDelta({ transcript_path: file }, null, config, []);
  assert.deepEqual(delta.records.map((record) => record.id), ["u1", "a1", "a2"]);
});

// 杀变异 PI04（pi 不剥注入回执）：注入的块被 pi 落进会话 JSONL（role custom 的那条），送回服务前要按回执剥掉
test("blocks the plugin injected are stripped from pi's session file before it is sent", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const file = join(stateRoot, "session.jsonl");
  const prepared = createContextInjection("用户偏好简洁回答。", "5".repeat(32));
  await writeFile(file, [
    line({ type: "message", id: "u1", message: { role: "user", content: "看报错" } }),
    line({ type: "message", id: "c1", message: { role: "custom", customType: "habitus-memory", content: `前 ${prepared.injection} 后`, display: false } }),
    line({ type: "message", id: "a1", message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "好" }] } }),
  ].join(""));
  const delta = await hostAdapter.readTranscriptDelta({ transcript_path: file }, null, { maxTranscriptBytes: 1 << 20 }, [prepared.receipt]);
  assert.equal(delta.records.find((record) => record.id === "c1").message.content, "前  后");
  assert.equal(JSON.stringify(delta.records).includes("habitus-memory-context"), false);
});

// 杀变异 PI03（只留最后一个轮边界）：一段里有几轮就给几个轮边界
test("a pi delta holding two finished turns reports both turn ends", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const file = join(stateRoot, "session.jsonl");
  const message = (id, body) => line({ type: "message", id, message: body });
  await writeFile(file, [
    message("u1", { role: "user", content: [{ type: "text", text: "一" }] }),
    message("a1", { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "好" }] }),
    message("u2", { role: "user", content: [{ type: "text", text: "二" }] }),
    message("a2", { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "行" }] }),
  ].join(""));
  const delta = await hostAdapter.readTranscriptDelta({ transcript_path: file }, null, { maxTranscriptBytes: 1 << 20 }, []);
  assert.deepEqual(delta.turns.map((turn) => turn.records.map((record) => record.id)), [["u1", "a1"], ["u2", "a2"]]);
});

// 杀变异 PI02（截断的答复不算重试）：stopReason 为 length 的答复被 context_edit 拿掉后 pi 接着做同一轮，那里不是一轮结束
test("a truncated answer pi retries is not where a turn ends", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const file = join(stateRoot, "session.jsonl");
  const message = (id, body) => line({ type: "message", id, message: body });
  await writeFile(file, [
    message("u1", { role: "user", content: [{ type: "text", text: "写长一点" }] }),
    message("a1", { role: "assistant", stopReason: "length", content: [{ type: "text", text: "写到一半" }] }),
    line({ type: "context_edit", id: "e1", targetId: "a1", replacement: null }),
    message("a2", { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "写完了" }] }),
  ].join(""));
  const delta = await hostAdapter.readTranscriptDelta({ transcript_path: file }, null, { maxTranscriptBytes: 1 << 20 }, []);
  assert.deepEqual(delta.turns.map((turn) => turn.records.map((record) => record.id)), [["u1", "a1", "e1", "a2"]]);
});

async function capturedStderr(work) {
  const written = [];
  const write = process.stderr.write;
  process.stderr.write = (chunk) => { written.push(String(chunk)); return true; };
  try { await work(); } finally { process.stderr.write = write; }
  return written.join("");
}

// T6：HABITUS_PLUGIN_DEBUG（config.debug）在 pi 进程内也生效：出错照常只记日志、放行，开着时另把堆栈打到 stderr
test("with debug on, an error inside the pi extension prints its stack to stderr; with it off nothing is printed", async (t) => {
  for (const debug of [true, false]) {
    const stateRoot = await temporaryRoot(t);
    const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20, debug };
    const core = new PluginCore(config, { service: fakeService() });
    core.enqueueCapture = async () => { throw new Error("pi capture exploded"); };
    const pi = fakePi();
    HabitusMemoryExtension(pi, { config, core });
    const stderr = await capturedStderr(() => pi.emit("agent_settled", {}, session(join(stateRoot, "session.jsonl"))));
    if (debug) assert.match(stderr, /^\[habitus-memory\] Error: pi capture exploded\n\s+at /);
    else assert.equal(stderr, "");
    assert.match(await readFile(join(stateRoot, "logs", "operations.jsonl"), "utf8"), /pi capture exploded/);
  }
});
