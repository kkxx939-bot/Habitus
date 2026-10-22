import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

import HabitusMemoryPlugin, { userText } from "../../habitus-memory-opencode/index.mjs";
import { hostAdapter } from "../../habitus-memory-opencode/scripts/host-adapter.mjs";
import { TranscriptMirror as SharedMirror, transcriptPath as sharedPath } from "../lib/parts-transcript.mjs";

class TranscriptMirror extends SharedMirror { constructor(root) { super(root, "opencode"); } }
const transcriptPath = (root, id) => sharedPath(root, id, "opencode");
import { HARNESS_REGISTRY } from "../../harnesses.mjs";
import { executePluginLifecycle } from "../../install-memory-plugin.mjs";
import { fileEntryPath } from "../../registrations.mjs";
import { requireHostAdapter } from "../lib/host-adapter.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";

const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-opencode-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

function message(id, role, parts, { completed = true } = {}) {
  const time = role === "assistant" ? (completed ? { created: 1, completed: 2 } : { created: 1 }) : { created: 1 };
  return { info: { id, role, sessionID: "ses_1", time }, parts };
}

function fakeService({ reminders = [], context = "", remembered = [] } = {}) {
  return {
    capabilities: async () => ({ ok: true, result: { api_version: "1.0", protocols: ["opencode"], features: [...FEATURES, "reminders_v1"] } }),
    ready: async () => ({ ok: true, result: { ready: true } }),
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

test("the OpenCode host adapter satisfies the shared contract", () => {
  assert.equal(requireHostAdapter(hostAdapter), hostAdapter);
  assert.equal(hostAdapter.conversationId({ session_id: "ses_1" }), "opencode-ses_1");
  assert.equal(hostAdapter.nativeSessionId({ session_id: "ses_2", parent_id: "ses_1" }), "ses_1.subagent.ses_2");
  assert.equal(userText([{ type: "text", text: "注入", synthetic: true }, { type: "text", text: " 帮我看看 " }]), "帮我看看");
});

test("chat.message injects recalled memory and the reminder block as one synthetic part", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20, reminderClaimLimit: 3 };
  const reminder = { request_id: "e1", kind: "写周报", text: "周四了，周报还没写。", options: ["现在就做", "这次不用", "知道了"], claimed_at: "2026-10-22T15:52:10+08:00" };
  const core = new PluginCore(config, { service: fakeService({ context: "用户偏好简洁回答。", reminders: [reminder] }) });
  const hooks = await HabitusMemoryPlugin({ client: {} }, { config, core });
  const output = { message: { id: "msg_1", sessionID: "ses_1" }, parts: [{ type: "text", text: "帮我看看这个报错" }] };
  await hooks["chat.message"]({ sessionID: "ses_1", messageID: "msg_1" }, output);
  assert.equal(output.parts.length, 2);
  const injected = output.parts[0];
  assert.equal(injected.synthetic, true);
  assert.equal(injected.messageID, "msg_1");
  assert.match(injected.text, /<habitus-memory-context receipt="[0-9a-f]{32}">/);
  assert.match(injected.text, /<habitus-reminder receipt="[0-9a-f]{32}">/);
  assert.match(injected.text, /question/);
  // 同一次取走再来（租约内重取、claimed_at 不变）：不再注入提醒
  const again = { message: { id: "msg_2", sessionID: "ses_1" }, parts: [{ type: "text", text: "再看看" }] };
  await hooks["chat.message"]({ sessionID: "ses_1", messageID: "msg_2" }, again);
  assert.doesNotMatch(again.parts[0].text, /habitus-reminder/);
  // 服务出错：静默放行，不注入
  const broken = new PluginCore(config, { service: { capabilities: async () => ({ ok: false, error: "down" }) } });
  const quiet = await HabitusMemoryPlugin({ client: {} }, { config, core: broken });
  const untouched = { message: { id: "msg_3", sessionID: "ses_1" }, parts: [{ type: "text", text: "x" }] };
  await quiet["chat.message"]({ sessionID: "ses_1", messageID: "msg_3" }, untouched);
  assert.equal(untouched.parts.length, 1);
});

test("session.idle mirrors completed messages once and sends them to the service", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  const remembered = [];
  const core = new PluginCore(config, { service: fakeService({ remembered }) });
  const question = { questions: [{ question: "周四了，周报还没写。", header: "提醒", options: [{ label: "现在就做", description: "" }] }] };
  const messages = [
    message("msg_1", "user", [{ type: "text", text: "帮我看看" }]),
    message("msg_2", "assistant", [
      { type: "tool", callID: "call_q", tool: "question", state: { status: "completed", input: question, output: "answered", metadata: { answers: [["这次不用"]] }, time: { start: 1, end: 2 } } },
      { type: "text", text: "看完了。" },
    ]),
    message("msg_3", "user", [{ type: "text", text: "下一句" }]),
    message("msg_4", "assistant", [{ type: "text", text: "还在写……" }], { completed: false }),
  ];
  const client = { session: { messages: async () => ({ data: messages }) } };
  const hooks = await HabitusMemoryPlugin({ client }, { config, core });
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "ses_1" } } });

  const lines = (await readFile(transcriptPath(stateRoot, "ses_1"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
  assert.deepEqual(lines.map((item) => item.info.id), ["msg_1", "msg_2", "msg_3"]); // 没完成的助手消息先不镜像，后面的也不越过它
  assert.deepEqual(lines[1].parts[0].state.metadata.answers, [["这次不用"]]); // 作答原样带着，服务端据此认
  assert.equal(remembered.length, 1);
  assert.equal(remembered[0].protocol, "opencode");
  assert.equal(remembered[0].conversationId, "opencode-ses_1");
  assert.deepEqual(remembered[0].payload.records.map((item) => item.info.id), ["msg_1", "msg_2"]); // 到最后一条完成的助手消息为止
  // 再空闲一次：已镜像的不重复写
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "ses_1" } } });
  const again = (await readFile(transcriptPath(stateRoot, "ses_1"), "utf8")).trim().split("\n");
  assert.equal(again.length, 3);
});

test("the mirror survives a restart without duplicating lines", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const first = new TranscriptMirror(stateRoot);
  await first.mirror("ses_9", [message("m1", "user", [{ type: "text", text: "a" }])]);
  const second = new TranscriptMirror(stateRoot);
  const result = await second.mirror("ses_9", [message("m1", "user", [{ type: "text", text: "a" }]), message("m2", "assistant", [{ type: "text", text: "b" }])]);
  assert.equal(result.appended, 1);
});

test("installing for OpenCode writes a managed entry file and removing it deletes only ours", async (t) => {
  const root = await temporaryRoot(t);
  const env = { XDG_CONFIG_HOME: join(root, "config") };
  const definition = HARNESS_REGISTRY.resolve("opencode");
  const runner = { available: () => true, json: () => ({}), run: () => { throw new Error("no CLI commands for file registration"); } };
  const marketplace = join(root, "marketplace");
  const installed = await executePluginLifecycle({ action: "install", harnesses: ["opencode"], root: marketplace }, { runner, env });
  assert.equal(installed.harnesses[0].harness, "opencode");
  const entry = await readFile(fileEntryPath(definition, env), "utf8");
  const target = pathToFileURL(join(marketplace, "plugins", "habitus-memory-opencode", "index.mjs")).href;
  assert.match(entry, /^\/\/ habitus-memory: managed/);
  assert.ok(entry.includes(JSON.stringify(target)));
  const module = await import(target);
  assert.equal(typeof module.default, "function");
  const status = await executePluginLifecycle({ action: "status", harnesses: ["opencode"], root: marketplace }, { runner, env });
  assert.equal(status.harnesses[0].installed, true);
  assert.equal(status.harnesses[0].marketplaceSource, marketplace);
  await executePluginLifecycle({ action: "remove", harnesses: ["opencode"], root: marketplace }, { runner, env });
  await assert.rejects(readFile(fileEntryPath(definition, env), "utf8"), /ENOENT/);
  // 用户自己的同名文件不覆盖、不删
  await mkdir(join(env.XDG_CONFIG_HOME, "opencode", "plugins"), { recursive: true });
  await writeFile(fileEntryPath(definition, env), "export default async () => ({})\n");
  await assert.rejects(
    executePluginLifecycle({ action: "install", harnesses: ["opencode"], root: marketplace }, { runner, env }),
    /not managed by Habitus/,
  );
  assert.equal(await readFile(fileEntryPath(definition, env), "utf8"), "export default async () => ({})\n");
});

// ── review 确认的缺陷（2026-10-21）────────────────────────────────────────────────────────────────

test("concurrent mirrors of one session write each message once", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const messages = [message("m1", "user", [{ type: "text", text: "a" }]), message("m2", "assistant", [{ type: "text", text: "b" }])];
  const mirror = new TranscriptMirror(stateRoot);
  await Promise.all([mirror.mirror("ses_1", messages), mirror.mirror("ses_1", messages)]);
  const ids = (await readFile(transcriptPath(stateRoot, "ses_1"), "utf8")).trim().split("\n").map((line) => JSON.parse(line).info.id);
  assert.deepEqual(ids, ["m1", "m2"]);
});

test("a torn last line from an interrupted write is dropped before appending", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const path = transcriptPath(stateRoot, "ses_1");
  await mkdir(join(stateRoot, "transcripts", "opencode"), { recursive: true });
  await writeFile(path, `${JSON.stringify(message("m1", "user", []))}\n{"info":{"id":"m2","ro`);
  const result = await new TranscriptMirror(stateRoot).mirror("ses_1", [
    message("m1", "user", []), message("m2", "assistant", [{ type: "text", text: "b" }]),
  ]);
  assert.equal(result.appended, 1);
  const ids = (await readFile(path, "utf8")).trim().split("\n").map((line) => JSON.parse(line).info.id);
  assert.deepEqual(ids, ["m1", "m2"]);
});

test("a failed reminder claim still injects the recalled memory", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20, reminderClaimLimit: 3 };
  const core = new PluginCore(config, { service: fakeService({ context: "用户偏好简洁回答。" }) });
  core.claimReminders = async () => { throw new Error("state lock timed out"); };
  const hooks = await HabitusMemoryPlugin({ client: {} }, { config, core });
  const output = { message: { id: "msg_1", sessionID: "ses_1" }, parts: [{ type: "text", text: "帮我看看" }] };
  await hooks["chat.message"]({ sessionID: "ses_1", messageID: "msg_1" }, output);
  assert.equal(output.parts.length, 2);
  assert.match(output.parts[0].text, /habitus-memory-context/);
});

test("an unreadable plugin config disables the plugin instead of failing OpenCode's plugin load", async () => {
  const previous = process.env.HABITUS_URL;
  process.env.HABITUS_URL = "http://192.168.1.5:8787";
  const errors = [];
  const write = process.stderr.write;
  process.stderr.write = (chunk) => { errors.push(String(chunk)); return true; };
  try {
    assert.deepEqual(await HabitusMemoryPlugin({ client: {} }), {});
  } finally {
    process.stderr.write = write;
    if (previous === undefined) delete process.env.HABITUS_URL; else process.env.HABITUS_URL = previous;
  }
  assert.match(errors.join(""), /habitus-memory: plugin disabled/);
});

test("the first capture also delivers other sessions' stuck outbox items, and a deleted session is cleaned up", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  const transcript = (id) => [message(`${id}_1`, "user", [{ type: "text", text: "a" }]), message(`${id}_2`, "assistant", [{ type: "text", text: "b" }])];
  const client = { session: { messages: async ({ path }) => ({ data: transcript(path.id) }) } };
  // 服务没起时 ses_A 空闲：排进 outbox，没送出去
  const down = new PluginCore(config, { service: { capabilities: async () => ({ ok: false, error: "down" }) } });
  await (await HabitusMemoryPlugin({ client }, { config, core: down })).event({ event: { type: "session.idle", properties: { sessionID: "ses_A" } } });
  // OpenCode 重启、服务起来了，另一个会话 ses_B 空闲：ses_A 一并补送
  const remembered = [];
  const up = new PluginCore(config, { service: fakeService({ remembered }) });
  const hooks = await HabitusMemoryPlugin({ client }, { config, core: up });
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "ses_B" } } });
  assert.deepEqual(remembered.map((item) => item.conversationId).sort(), ["opencode-ses_A", "opencode-ses_B"]);
  // 删会话：送完、收尾后镜像文件删掉
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "ses_B" } } } });
  await assert.rejects(readFile(transcriptPath(stateRoot, "ses_B"), "utf8"), /ENOENT/);
  assert.ok(await readFile(transcriptPath(stateRoot, "ses_A"), "utf8"));
});

test("a subagent session's task prompt neither recalls memory nor claims reminders, even after a plugin restart", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20, reminderClaimLimit: 3 };
  let claims = 0;
  const core = new PluginCore(config, { service: { ...fakeService({ context: "记忆" }), claimReminders: async () => { claims += 1; return { ok: true, result: { reminders: [] } }; } } });
  // 插件进程重启过：没收到 session.created，父会话问 OpenCode
  const client = { session: { get: async ({ path }) => ({ data: { id: path.id, parentID: path.id === "ses_child" ? "ses_parent" : undefined } }) } };
  const hooks = await HabitusMemoryPlugin({ client }, { config, core });
  const task = { message: { id: "msg_1", sessionID: "ses_child" }, parts: [{ type: "text", text: "去查一下这个报错的根因" }] };
  await hooks["chat.message"]({ sessionID: "ses_child", messageID: "msg_1" }, task);
  assert.equal(task.parts.length, 1);
  assert.equal(claims, 0);
  const said = { message: { id: "msg_2", sessionID: "ses_parent" }, parts: [{ type: "text", text: "帮我看看" }] };
  await hooks["chat.message"]({ sessionID: "ses_parent", messageID: "msg_2" }, said);
  assert.equal(said.parts.length, 2);
  assert.equal(claims, 1);
});

test("an assistant message left incomplete by a crash does not block later messages", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const result = await new TranscriptMirror(stateRoot).mirror("ses_1", [
    message("m1", "user", [{ type: "text", text: "a" }]),
    message("m2", "assistant", [{ type: "text", text: "写到一半" }], { completed: false }),
    message("m3", "user", [{ type: "text", text: "接着说" }]),
    message("m4", "assistant", [{ type: "text", text: "好" }]),
    message("m5", "assistant", [{ type: "text", text: "还在写" }], { completed: false }),
  ]);
  assert.equal(result.appended, 4);
  const ids = (await readFile(transcriptPath(stateRoot, "ses_1"), "utf8")).trim().split("\n").map((line) => JSON.parse(line).info.id);
  assert.deepEqual(ids, ["m1", "m2", "m3", "m4"]);
});

test("compaction summaries and the replayed prompt are not mirrored; inline attachments become placeholders", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const image = `data:image/png;base64,${"A".repeat(4096)}`;
  const summary = message("s", "assistant", [{ type: "text", text: "## Summary" }]);
  summary.info.summary = true;
  await new TranscriptMirror(stateRoot).mirror("ses_1", [
    message("u1", "user", [{ type: "text", text: "改登录页" }, { type: "file", mime: "image/png", filename: "a.png", url: image }]),
    message("a1", "assistant", [{ type: "tool", callID: "c", tool: "read", state: { status: "completed", input: {}, output: "Image read", attachments: [{ type: "file", mime: "image/png", url: image }], time: { start: 1, end: 2 } } }]),
    message("c", "user", [{ type: "compaction", auto: true, overflow: true }]),
    summary,
    message("u2", "user", [{ type: "text", text: "改登录页" }]),
    message("a2", "assistant", [{ type: "text", text: "继续改" }]),
  ]);
  const text = await readFile(transcriptPath(stateRoot, "ses_1"), "utf8");
  const ids = text.trim().split("\n").map((line) => JSON.parse(line).info.id);
  assert.deepEqual(ids, ["u1", "a1", "c", "a2"]);
  assert.ok(!text.includes("base64"));
  assert.match(text, /\[omitted image\/png \d+B\]/);
});

test("session.deleted sends what was already mirrored without fetching the deleted messages", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  const remembered = [];
  const core = new PluginCore(config, { service: fakeService({ remembered }) });
  let fetches = 0;
  const data = [message("m1", "user", [{ type: "text", text: "a" }]), message("m2", "assistant", [{ type: "text", text: "b" }])];
  const client = { session: { messages: async () => { fetches += 1; return { data }; }, get: async () => ({ data: {} }) } };
  const hooks = await HabitusMemoryPlugin({ client }, { config, core });
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "ses_1" } } });
  await hooks.event({ event: { type: "session.deleted", properties: { info: { id: "ses_1" } } } });
  assert.equal(fetches, 1);
  assert.equal(remembered.length, 1);
  await assert.rejects(readFile(transcriptPath(stateRoot, "ses_1"), "utf8"), /ENOENT/);
});
