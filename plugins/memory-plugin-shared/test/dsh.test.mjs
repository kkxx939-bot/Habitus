import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import * as zlib from "node:zlib";

import { createDshAdapter } from "../../habitus-memory-dsh/scripts/host-adapter.mjs";
import { decodeLog, encodeSegment, mirrorEvents, readSessionLog } from "../../habitus-memory-dsh/scripts/session-log.mjs";
import { executePluginLifecycle } from "../../install-memory-plugin.mjs";
import { createContextInjection, promptInjection } from "../lib/hook-runner.mjs";
import { requireHostAdapter } from "../lib/host-adapter.mjs";
import { patchFilePath } from "../../registrations.mjs";
import { HARNESS_REGISTRY } from "../../harnesses.mjs";

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-dsh-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

const line = (value) => `${JSON.stringify(value)}\n`;
const header = (id, extra = {}) => ({ id, createdAt: "2026-10-22T07:00:00Z", cwd: "/w", ...extra });
const event = (seq, type, data = {}) => ({ type, seq, time: 1792998000000 + seq, data });

test("session ids are escaped the way dsh names their directories", () => {
  assert.equal(encodeSegment("session-019a_x.y"), "session-019a_x.y");
  assert.equal(encodeSegment("a/b~c"), "a~002Fb~007Ec");
  assert.equal(encodeSegment(".."), "~002E~002E");
});

test("the newest generation is decoded across zstd frames and a torn last frame keeps its complete lines", { skip: !zlib.zstdCompressSync && "Node without zstd" }, async (t) => {
  const home = await temporaryRoot(t);
  const directory = join(home, "sessions", "--w--", encodeSegment("s1"));
  await mkdir(directory, { recursive: true });
  // 旧一代与当前一代并存：取版本号最高的
  await writeFile(join(directory, "session.v1.jsonl"), line(header("s1")));
  const frames = Buffer.concat([
    zlib.zstdCompressSync(Buffer.from(line(header("s1")) + line(event(1, "turn/start")))),
    zlib.zstdCompressSync(Buffer.from(line(event(2, "user/message")) + line(event(3, "turn/end")))),
  ]);
  await writeFile(join(directory, "session.v3.jsonl.zstd"), frames);
  const { header: read, events } = await readSessionLog("s1", { DSH_HOME: home });
  assert.equal(read.id, "s1");
  assert.deepEqual(events.map((item) => item.seq), [1, 2, 3]);
  const torn = Buffer.concat([zlib.zstdCompressSync(Buffer.from(line(event(1, "turn/start")))), Buffer.from([0x28, 0xb5, 0x2f, 0xfd, 0x00])]);
  assert.deepEqual(decodeLog(torn, true).map((item) => item.seq), [1]);
  assert.deepEqual(await readSessionLog("missing", { DSH_HOME: home }), { header: null, events: [] });
  // dsh 写的帧带校验和；一批事件大时一帧里有好几个块
  const checksummed = { params: { [zlib.constants.ZSTD_c_checksumFlag]: 1 } };
  const big = Array.from({ length: 4000 }, (_, index) => line(event(index + 10, "assistant/message", { text: `段落 ${index} ${"字".repeat(40)}` }))).join("");
  const mixed = Buffer.concat([zlib.zstdCompressSync(Buffer.from(line(event(1, "turn/start"))), checksummed), zlib.zstdCompressSync(Buffer.from(big), checksummed)]);
  const decoded = decodeLog(mixed, true);
  assert.equal(decoded.length, 4001);
  assert.equal(decoded.at(-1).seq, 4009);
});

test("the local mirror only appends events past the last mirrored seq", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const first = await mirrorEvents(stateRoot, "s1", [event(1, "a"), event(2, "b")]);
  const again = await mirrorEvents(stateRoot, "s1", [event(1, "a"), event(2, "b"), event(3, "c")]);
  assert.equal(first.appended, 2);
  assert.equal(again.appended, 1);
  const seqs = (await readFile(first.path, "utf8")).trim().split("\n").map((value) => JSON.parse(value).seq);
  assert.deepEqual(seqs, [1, 2, 3]);
});

test("a torn last line in the mirror is cut before appending, so the session stays readable", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  const events = [event(1, "turn/start"), event(2, "user/message"), event(3, "turn/end"), event(4, "turn/start"), event(5, "user/message"), event(6, "turn/end")];
  const adapter = createDshAdapter({ readLog: async () => ({ header: header("torn"), events }) });
  const { path } = await mirrorEvents(stateRoot, "torn", events.slice(0, 3));
  await writeFile(path, `${await readFile(path, "utf8")}${JSON.stringify(events[3]).slice(0, 20)}`);
  const stop = await adapter.prepareInput({ session_id: "torn", transcript_path: "" }, config, "stop");
  const delta = await adapter.readTranscriptDelta(stop, null, config, []);
  assert.equal(delta.records.length, 6);
  const seqs = (await readFile(path, "utf8")).trim().split("\n").map((value) => JSON.parse(value).seq);
  assert.deepEqual(seqs, [1, 2, 3, 4, 5, 6]);
});

test("the dsh adapter mirrors on capture, recognises subagent sessions and skips their prompts", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  const logs = {
    main: { header: header("main"), events: [event(1, "user/message"), event(2, "turn/end")] },
    child: { header: header("child", { parentSession: "main", origin: "subagent" }), events: [] },
  };
  const adapter = createDshAdapter({ readLog: async (id) => logs[id] });
  assert.equal(requireHostAdapter(adapter), adapter);
  const stop = await adapter.prepareInput({ session_id: "main", transcript_path: "" }, config, "stop");
  assert.ok(stop.transcript_path.endsWith(".jsonl"));
  assert.equal((await readFile(stop.transcript_path, "utf8")).trim().split("\n").length, 2);
  assert.equal(adapter.conversationId(stop), "dsh-main");
  const child = await adapter.prepareInput({ session_id: "child", transcript_path: "", prompt: "去查" }, config, "user-prompt");
  assert.equal(adapter.isSubagent(child), true);
  assert.equal(adapter.isSubagent(stop), false);
  const core = { recall: () => assert.fail("a subagent's task prompt is not his words") };
  assert.equal(await promptInjection(core, adapter, child, { config, hook: "user-prompt" }), "");
});

// Q3：子代理的会话不进记忆——Stop 时既不镜像它的日志，也不送（真跑 hook 进程的那一条在 hook-runner.test）
test("a dsh subagent session's log is not mirrored when it stops", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  const logs = { child: { header: header("child", { parentSession: "main", origin: "subagent" }), events: [event(1, "user/message"), event(2, "turn/end")] } };
  const adapter = createDshAdapter({ readLog: async (id) => logs[id] });
  const child = await adapter.prepareInput({ session_id: "child", transcript_path: "" }, config, "stop");
  assert.equal(adapter.isSubagent(child), true);
  await assert.rejects(readFile(child.transcript_path, "utf8"), /ENOENT/);
});

test("installing for dsh adds the hooks bridge and a managed entry to the home patch, leaving the rest alone", async (t) => {
  const root = await temporaryRoot(t);
  const env = { HOME: join(root, "home"), DSH_HOME: join(root, "dsh") };
  const definition = HARNESS_REGISTRY.resolve("deepseek");
  const path = patchFilePath(definition, env);
  await mkdir(env.DSH_HOME, { recursive: true });
  const mine = "- name: '@deepseek-ai/dsh-tool-web'\n  config:\n    enabled: true\n";
  await writeFile(path, mine);
  const calls = [];
  const runner = { available: (command) => command === "dsh", json: () => [], run: (command, args) => calls.push([command, ...args].join(" ")) };
  const marketplace = join(root, "marketplace");
  const installed = await executePluginLifecycle({ action: "install", harnesses: ["dsh"], root: marketplace }, { runner, env });
  assert.deepEqual(calls, ["dsh plugin --profile web add @deepseek-ai/dsh-hooks-claude-code@next"]);
  assert.equal(installed.harnesses[0].installed, true);
  const patched = await readFile(path, "utf8");
  assert.ok(patched.startsWith(mine.trimEnd()));
  assert.match(patched, /- name: "@deepseek-ai\/dsh-hooks-claude-code@next"/);
  assert.ok(patched.includes(JSON.stringify(join(marketplace, "plugins", "habitus-memory-dsh", "hooks", "hooks.json"))));
  // 再装一次：替换受管那段，不重复
  await executePluginLifecycle({ action: "install", harnesses: ["dsh"], root: marketplace }, { runner, env });
  assert.equal((await readFile(path, "utf8")).split("habitus-memory: begin").length, 2);
  await executePluginLifecycle({ action: "remove", harnesses: ["dsh"], root: marketplace }, { runner, env });
  assert.equal(await readFile(path, "utf8"), mine);
  // 流式写法的 patch 不改，报错说怎么手动加
  await writeFile(path, "[{name: x}]\n");
  await assert.rejects(
    executePluginLifecycle({ action: "install", harnesses: ["dsh"], root: marketplace }, { runner, env }),
    /flow-style YAML list/,
  );
  assert.equal(await readFile(path, "utf8"), "[{name: x}]\n");
});

async function mirroredStop(t, events) {
  const stateRoot = await temporaryRoot(t);
  const config = { enabled: true, stateRoot, timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  const adapter = createDshAdapter({ readLog: async () => ({ header: header("cut"), events }) });
  return { adapter, config, input: await adapter.prepareInput({ session_id: "cut", transcript_path: "" }, config, "stop") };
}

// 杀变异 DS01（任意记录都能切）：dsh 只在 turn/end 之后切（裁定 69），一轮做到一半的事件留到下一批
test("dsh batches end only at turn/end", async (t) => {
  const events = [event(1, "turn/start"), event(2, "user/message"), event(3, "turn/end"), event(4, "turn/start"), event(5, "user/message")];
  const { adapter, config, input } = await mirroredStop(t, events);
  const delta = await adapter.readTranscriptDelta(input, null, config, []);
  assert.deepEqual(delta.records.map((record) => record.seq), [1, 2, 3]);
  const unfinished = await mirroredStop(t, events.slice(3));
  assert.equal(await unfinished.adapter.readTranscriptDelta(unfinished.input, null, unfinished.config, []), null);
});

// 杀变异 DS04（dsh 不剥注入回执）：桥把注入的上下文写进会话日志，送回服务前要按回执剥掉
test("blocks the plugin injected are stripped from the dsh log before it is sent", async (t) => {
  const prepared = createContextInjection("用户偏好简洁回答。", "6".repeat(32));
  const events = [event(1, "user/message", { text: `看报错\n${prepared.injection}` }), event(2, "turn/end")];
  const { adapter, config, input } = await mirroredStop(t, events);
  const delta = await adapter.readTranscriptDelta(input, null, config, [prepared.receipt]);
  assert.equal(delta.records[0].data.text, "看报错\n");
  assert.equal(JSON.stringify(delta.records).includes("habitus-memory-context"), false);
});

// 杀变异 DS04（只留最后一个轮边界）：一段里有几轮就给几个轮边界，按轮拼批时才能在每一轮的结束处拆开
test("a dsh delta holding two finished turns reports both turn ends", async (t) => {
  const events = [
    event(1, "turn/start"), event(2, "user/message"), event(3, "turn/end"),
    event(4, "turn/start"), event(5, "user/message"), event(6, "turn/end"),
  ];
  const { adapter, config, input } = await mirroredStop(t, events);
  const delta = await adapter.readTranscriptDelta(input, null, config, []);
  assert.deepEqual(delta.turns.map((turn) => turn.records.map((record) => record.seq)), [[1, 2, 3], [4, 5, 6]]);
});
