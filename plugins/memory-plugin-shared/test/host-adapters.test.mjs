import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { hostAdapter as claude } from "../../habitus-memory-claude-code/scripts/host-adapter.mjs";
import { hostAdapter as codex } from "../../habitus-memory/scripts/host-adapter.mjs";
import { createContextInjection } from "../lib/hook-runner.mjs";
import { hasMeaningfulMessageContent, requireInjectionReceipt, stripInjectionReceipts } from "../lib/injection-receipt.mjs";
import { listOutbox } from "../lib/outbox.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";

const FIXTURES = fileURLToPath(new URL("./fixtures", import.meta.url));

async function transcript(t, records) {
  const directory = await mkdtemp(join(tmpdir(), "habitus-host-adapter-test-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const path = join(directory, "transcript.jsonl");
  await writeFile(path, records.map((value) => JSON.stringify(value)).join("\n") + "\n");
  return path;
}

const config = { maxTranscriptBytes: 1024 * 1024 };

test("Codex capture waits for custom tool output and a terminal assistant message", async (t) => {
  const pending = await transcript(t, [
    { type: "response_item", payload: { type: "custom_tool_call", call_id: "c1", name: "shell", input: "pwd" } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: [] } },
  ]);
  assert.equal(await codex.readTranscriptDelta({ transcript_path: pending }, null, config), null);

  const complete = await transcript(t, [
    { type: "response_item", payload: { type: "custom_tool_call", call_id: "c1", name: "shell", input: "pwd" } },
    { type: "response_item", payload: { type: "custom_tool_call_output", call_id: "c1", output: "/tmp" } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: [] } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: complete }, null, config);
  assert.equal(delta.records.length, 3);
  assert.equal(delta.afterTurn, true);
});

test("Codex capture waits for tool search output and retains native agent events", async (t) => {
  const path = await transcript(t, [
    { type: "response_item", payload: { type: "agent_message", author: "worker", content: [{ type: "output_text", text: "evidence" }] } },
    { type: "response_item", payload: { type: "sub_agent_activity", event_id: "e1", kind: "completed" } },
    { type: "response_item", payload: { type: "tool_search_call", call_id: "s1", arguments: { query: "memory" } } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: "not terminal" } },
  ]);
  assert.equal(await codex.readTranscriptDelta({ transcript_path: path }, null, config), null);

  const complete = await transcript(t, [
    { type: "response_item", payload: { type: "agent_message", author: "worker", content: [{ type: "output_text", text: "evidence" }] } },
    { type: "response_item", payload: { type: "sub_agent_activity", event_id: "e1", kind: "completed" } },
    { type: "response_item", payload: { type: "tool_search_call", call_id: "s1", arguments: { query: "memory" } } },
    { type: "response_item", payload: { type: "tool_search_output", call_id: "s1", tools: [{ name: "github" }] } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: "terminal" } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: complete }, null, config);
  assert.equal(delta.records.length, 5);
});

test("checked-in host transcript fixtures remain consumable", async () => {
  const codexDelta = await codex.readTranscriptDelta(
    { transcript_path: join(FIXTURES, "codex-native.jsonl") }, null, config,
  );
  const claudeDelta = await claude.readTranscriptDelta(
    { session_id: "parent", transcript_path: join(FIXTURES, "claude-subagent.jsonl") },
    null,
    config,
  );
  assert.equal(codexDelta.records.length, 5);
  assert.equal(claudeDelta.records.length, 2);
});

test("Claude capture waits for tool_result and emits no legacy approve decision", async (t) => {
  const path = await transcript(t, [
    { type: "assistant", message: { role: "assistant", stop_reason: "tool_use", content: [{ type: "tool_use", id: "t1" }] } },
    { type: "user", message: { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "ok" }] } },
    { type: "assistant", message: { role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "done" }] } },
  ]);
  const delta = await claude.readTranscriptDelta({ transcript_path: path }, null, config);
  assert.equal(delta.records.length, 3);
  assert.deepEqual(claude.successOutput("stop"), {});
  assert.equal("decision" in claude.contextOutput("memory"), false);
});

test("Claude capture pairs tool calls only on user / assistant records: a blob repeating a paired tool_use does not stall", async (t) => {
  const path = await transcript(t, [
    { type: "user", message: { role: "user", content: "列一下目录" } },
    { type: "assistant", message: { role: "assistant", stop_reason: "tool_use", content: [{ type: "tool_use", id: "t1", name: "Bash", input: {} }] } },
    { type: "user", message: { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: "a b" }] } },
    { type: "api-request-blob", sessionId: "s", hash: "h", message: { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "Bash", input: {} }] } },
    { type: "assistant", message: { role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "目录里有 a b" }] } },
  ]);
  const delta = await claude.readTranscriptDelta({ transcript_path: path }, null, config);
  assert.notEqual(delta, null);
  assert.equal(delta.nextCursor.lineCount, 5);
  assert.equal(delta.records.length, 4);
});

test("a Claude hook fired inside a subagent is recognised as one, and the main session's transcript path is the only one read", () => {
  const input = { session_id: "parent", agent_id: "worker-1", transcript_path: "/main.jsonl", agent_transcript_path: "/sub.jsonl" };
  assert.equal(claude.isSubagent(input), true);
  assert.equal(claude.isSubagent({ session_id: "parent", transcript_path: "/main.jsonl" }), false);
  assert.equal(claude.nativeSessionId(input), "parent");
  assert.equal(claude.transcriptPath(input), "/main.jsonl");
});

// U23：只有脚本 / SDK 的入口算非交互；交互的 CLI、IDE、桌面照常进记忆
test("the Claude Code adapter tells a non-interactive entrypoint from an interactive one", () => {
  for (const entrypoint of ["sdk-cli", "sdk-ts", "sdk-py"]) {
    assert.equal(claude.isNonInteractive({}, { CLAUDE_CODE_ENTRYPOINT: entrypoint }), true, entrypoint);
  }
  for (const entrypoint of ["cli", "claude-vscode", "claude-desktop", undefined]) {
    assert.equal(claude.isNonInteractive({}, { CLAUDE_CODE_ENTRYPOINT: entrypoint }), false, String(entrypoint));
  }
});

test("the Codex adapter tells a codex exec rollout from an interactive one by its session_meta source", async (t) => {
  for (const [source, expected] of [["exec", true], ["cli", false], ["vscode", false]]) {
    const path = await transcript(t, [{ type: "session_meta", payload: { id: "s", source } }]);
    const prepared = await codex.prepareInput({ session_id: "s", transcript_path: path });
    assert.equal(codex.isNonInteractive(prepared), expected, source);
    assert.equal(codex.isSubagent(prepared), false, source);
  }
});

// B3：Claude Code 转写里夹着大段 api-request 之类的日志记录（本机一份 415 MB 的转写里一轮夹了 140 MiB）：它们与切点、送出无关，
// 不占读取额度，一轮照样切得出来
test("Claude capture is not stalled by log records larger than the read limit between two turn ends", async (t) => {
  const path = await transcript(t, [
    { type: "user", message: { role: "user", content: "看看" } },
    { type: "api-request", body: "x".repeat(8 * 1024) },
    { type: "assistant", message: { role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "好" }] } },
  ]);
  const delta = await claude.readTranscriptDelta({ transcript_path: path }, null, { maxTranscriptBytes: 4096 });
  assert.deepEqual(delta.records.map((record) => record.type), ["user", "assistant"]);
  assert.equal(delta.nextCursor.lineCount, 3);
});

// T3（裁定 75 第 1 条）：一行原样超过读取额度的跳过不读（jsonl.mjs）。跳过的若是挂着的那个工具调用的结果（截图），这一轮照样
// 在它之后第一条说完了的答复处结束：整轮其余的记录在同一轮里，不拆开、不丢，后面的一轮照常切出来
test("a tool result too large to read is skipped and its turn still ends at the reply after it, whole", async (t) => {
  const shot = "A".repeat(8 * 1024);
  const claudePath = await transcript(t, [
    { type: "user", message: { role: "user", content: "截个图" } },
    { type: "assistant", message: { id: "m1", role: "assistant", stop_reason: "tool_use", content: [{ type: "tool_use", id: "t1", name: "Screenshot", input: {} }] } },
    { type: "user", message: { role: "user", content: [{ type: "tool_result", tool_use_id: "t1", content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: shot } }] }] } },
    { type: "assistant", message: { id: "m2", role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "看到了" }] } },
    { type: "user", message: { role: "user", content: "下一个" } },
    { type: "assistant", message: { id: "m3", role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "好" }] } },
  ]);
  const claudeDelta = await claude.readTranscriptDelta({ transcript_path: claudePath }, null, { maxTranscriptBytes: 4096 });
  assert.deepEqual(claudeDelta.turns.map((turn) => turn.records.map((record) => record.message.content?.[0]?.text ?? record.message.content?.[0]?.type ?? record.message.content)), [
    ["截个图", "tool_use", "看到了"],
    ["下一个", "好"],
  ]);
  assert.equal(claudeDelta.turns[0].nextCursor.lineCount, 4);
  assert.deepEqual(claudeDelta.skippedLines.map((line) => line.lineCount), [3]);

  const codexPath = await transcript(t, [
    { type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "截个图" }] } },
    { type: "response_item", payload: { type: "function_call", call_id: "c1", name: "screenshot", arguments: "{}" } },
    { type: "response_item", payload: { type: "function_call_output", call_id: "c1", output: shot } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: [{ type: "output_text", text: "看到了" }] } },
    { type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "下一个" }] } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: [{ type: "output_text", text: "好" }] } },
  ]);
  const codexDelta = await codex.readTranscriptDelta({ transcript_path: codexPath }, null, { maxTranscriptBytes: 4096 });
  assert.deepEqual(codexDelta.turns.map((turn) => turn.records.map((record) => record.payload.content?.[0]?.text ?? record.payload.type)), [
    ["截个图", "function_call", "看到了"],
    ["下一个", "好"],
  ]);
  assert.deepEqual(codexDelta.skippedLines.map((line) => line.lineCount), [3]);
});

test("injected memory context is consumed but excluded from captured payload", async (t) => {
  const prepared = createContextInjection("hidden", "2".repeat(32));
  const path = await transcript(t, [
    { type: "response_item", payload: { type: "message", role: "user", content: prepared.injection } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: "done" } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: path }, null, config, [prepared.receipt]);
  assert.equal(delta.records.length, 1);
  assert.equal(delta.records[0].payload.role, "assistant");
});

test("receipt removes only the injected byte range from concatenated Codex content", async (t) => {
  const prepared = createContextInjection("hidden", "3".repeat(32));
  const path = await transcript(t, [
    {
      type: "response_item",
      payload: { type: "message", role: "user", content: `before\n${prepared.injection}\nafter` },
    },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: "done" } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: path }, null, config, [prepared.receipt]);
  assert.equal(delta.records[0].payload.content, "before\n\nafter");
});

test("receipt removes only the injected byte range from nested Claude content", async (t) => {
  const prepared = createContextInjection("hidden", "4".repeat(32));
  const path = await transcript(t, [
    {
      type: "user",
      message: { role: "user", content: [{ type: "text", text: `before ${prepared.injection} after` }] },
    },
    {
      type: "assistant",
      message: { role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "done" }] },
    },
  ]);
  const delta = await claude.readTranscriptDelta({ transcript_path: path }, null, config, [prepared.receipt]);
  assert.equal(delta.records[0].message.content[0].text, "before  after");
});

test("nonce marker with a modified body is preserved", async (t) => {
  const prepared = createContextInjection("hidden", "5".repeat(32));
  const modified = prepared.injection.replace("hidden", "user fact");
  const path = await transcript(t, [
    { type: "response_item", payload: { type: "message", role: "user", content: modified } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: "done" } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: path }, null, config, [prepared.receipt]);
  assert.equal(delta.records[0].payload.content, modified);
});

test("literal context marker without an injection receipt remains user content", async (t) => {
  const path = await transcript(t, [
    { type: "response_item", payload: { type: "message", role: "user", content: "literal <habitus-memory-context> fact" } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: "done" } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: path }, null, config, []);
  assert.equal(delta.records.length, 2);
});


test("Codex capture never cuts inside a turn: a progress reply is not where a batch ends", async (t) => {
  // 压缩前（PreCompact）来取时这一轮还在做：只有中途的进度答复，没有 final_answer —— 不送，整轮留到做完一起送
  const midTurn = await transcript(t, [
    { type: "event_msg", payload: { type: "task_started" } },
    { type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "看下报错" }] } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "commentary", content: "我先看一下。" } },
    { type: "response_item", payload: { type: "function_call", call_id: "c1", name: "exec", arguments: "{}" } },
    { type: "response_item", payload: { type: "function_call_output", call_id: "c1", output: "ok" } },
  ]);
  assert.equal(await codex.readTranscriptDelta({ transcript_path: midTurn }, null, config), null);

  const done = await transcript(t, [
    { type: "event_msg", payload: { type: "task_started" } },
    { type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "看下报错" }] } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "commentary", content: "我先看一下。" } },
    { type: "compacted", payload: { message: "", replacement_history: [] } },
    { type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "还是暂停吧" }] } },
    { type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: "已暂停。" } },
    { type: "event_msg", payload: { type: "task_complete" } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: done }, null, config);
  assert.equal(delta.records.length, 6);  // 压缩前后两半、中途插的话、最终答复，一批送；task_complete 留给下一批
  assert.equal(delta.afterTurn, true);
});

// 杀变异 CC02（isMeta 过滤去掉）：Claude Code 自己插进来的元消息（命令回显、caveat 之类，isMeta）不是他说的话，不送
test("Claude capture leaves out the harness's own meta messages", async (t) => {
  const path = await transcript(t, [
    { type: "user", isMeta: true, message: { role: "user", content: "Caveat: the messages below were generated by the user while running local commands." } },
    { type: "user", message: { role: "user", content: "帮我看看这个报错" } },
    { type: "assistant", message: { role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "好" }] } },
  ]);
  const delta = await claude.readTranscriptDelta({ transcript_path: path }, null, config);
  assert.deepEqual(delta.records.map((record) => record.message.content), ["帮我看看这个报错", [{ type: "text", text: "好" }]]);
});

// R1（裁定 69）：Claude Code 把一条答复的 thinking、text 各写成一条记录、都带 end_turn；同一 message.id 后面还有块时不是一轮结束
const thinkingThenText = [
  { type: "user", uuid: "u1", message: { role: "user", content: "x".repeat(900) } },
  { type: "assistant", uuid: "a1", message: { id: "m1", role: "assistant", stop_reason: "end_turn", content: [{ type: "thinking", thinking: "想一想", signature: "s" }] } },
  { type: "assistant", uuid: "a2", message: { id: "m1", role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: `答复正文${"y".repeat(900)}` }] } },
  { type: "user", uuid: "u2", message: { role: "user", content: "下一问" } },
  { type: "assistant", uuid: "a3", message: { id: "m2", role: "assistant", stop_reason: "end_turn", content: [{ type: "text", text: "好" }] } },
];

test("Claude capture ends a turn after the last block of a reply, not after its thinking block", async (t) => {
  const path = await transcript(t, thinkingThenText.slice(0, 3));
  const delta = await claude.readTranscriptDelta({ transcript_path: path }, null, config);
  assert.deepEqual(delta.turns.map((turn) => turn.records.map((record) => record.uuid)), [["u1", "a1", "a2"]]);
  assert.equal(delta.nextCursor.lineCount, 3);
});

test("batching by turn under a small request limit never ends a batch on a reply's thinking block", async (t) => {
  const path = await transcript(t, thinkingThenText);
  const stateRoot = await mkdtemp(join(tmpdir(), "habitus-host-adapter-core-"));
  t.after(() => rm(stateRoot, { recursive: true, force: true }));
  await writeFile(join(stateRoot, "service-limits.json"), JSON.stringify({ schemaVersion: 1, maxRequestBytes: 2048 }));
  const core = new PluginCore({ stateRoot, timeoutMs: 1000, maxTranscriptBytes: config.maxTranscriptBytes }, { service: {} });
  await core.enqueueCapture(claude, { session_id: "s", transcript_path: path });
  const batches = (await listOutbox(stateRoot, "claude-code", "s")).map((item) => item.payload.records.map((record) => record.uuid));
  assert.deepEqual(batches, [["u1", "a1", "a2"], ["u2", "a3"]]);
});

// T2（裁定 69）：读取额度正好停在一条答复的 thinking 记录之后（同一 message.id 的 text 还没读到）：这一段的段尾不算一轮结束，
// 前一轮照常入队，这一轮下一段整轮读进来
test("a read that stops on its byte budget right after a reply's thinking block does not end a turn there", async (t) => {
  const reply = (uuid, id, content) => ({ type: "assistant", uuid, message: { id, role: "assistant", stop_reason: "end_turn", content } });
  const records = [
    { type: "user", uuid: "u0", message: { role: "user", content: "z".repeat(900) } },
    reply("a0", "m0", [{ type: "text", text: "先答" }]),
    { type: "user", uuid: "u1", message: { role: "user", content: "第一问" } },
    reply("a1", "m1", [{ type: "thinking", thinking: "想一想", signature: "s" }]),
    reply("a2", "m1", [{ type: "text", text: "答复正文" }]),
    { type: "user", uuid: "u2", message: { role: "user", content: "下一问" } },
    reply("a3", "m2", [{ type: "text", text: "好" }]),
  ];
  const path = await transcript(t, records);
  // 额度正好是 u0、a0、u1、a1 四行：第一段读到 a1（m1 的 thinking）为止，a2（m1 的 text）在额度外
  const budget = records.slice(0, 4).reduce((sum, record) => sum + Buffer.byteLength(JSON.stringify(record)) + 1, 0);
  const stateRoot = await mkdtemp(join(tmpdir(), "habitus-host-adapter-budget-"));
  t.after(() => rm(stateRoot, { recursive: true, force: true }));
  const core = new PluginCore({ stateRoot, timeoutMs: 1000, maxTranscriptBytes: budget }, { service: {} });
  await core.enqueueCapture(claude, { session_id: "s", transcript_path: path });
  const batches = (await listOutbox(stateRoot, "claude-code", "s")).map((item) => item.payload.records.map((record) => record.uuid));
  assert.deepEqual(batches, [["u0", "a0"], ["u1", "a1", "a2", "u2", "a3"]]);
});

const sha256 = (value) => createHash("sha256").update(value).digest("hex");

// 杀变异 IR01（老格式回执不剥）：升级前记下的回执只有整段摘要（nonce 为空）：哪段文字整段等于它，就整段剥掉
test("a receipt kept from before nonces strips a text whose whole digest matches it", () => {
  const injected = "<habitus-memory-context>旧的注入</habitus-memory-context>";
  const receipts = [{ nonce: null, digest: sha256(injected) }];
  assert.deepEqual(stripInjectionReceipts({ content: [injected, "他说的话"] }, receipts), { value: { content: ["", "他说的话"] }, removed: true });
  assert.deepEqual(stripInjectionReceipts(`${injected}\n他说的话`, receipts), { value: `${injected}\n他说的话`, removed: false });
});

// 杀变异 IR06（遇到摘要对不上的同名块就停）：前面一段带着同一个 nonce、内容被改过（比如他贴回来又改了）的块留着，后面原样的注入照样剥
test("a modified block carrying the same nonce does not stop the genuine injection after it from being stripped", () => {
  const nonce = "a".repeat(32);
  const genuine = `<habitus-reminder receipt="${nonce}">提醒</habitus-reminder>`;
  const edited = `<habitus-reminder receipt="${nonce}">改过的提醒</habitus-reminder>`;
  const result = stripInjectionReceipts(`${edited} 中间 ${genuine} 结尾`, [{ nonce, digest: sha256(genuine) }]);
  assert.deepEqual(result, { value: `${edited} 中间  结尾`, removed: true });
});

// 杀变异 IR04（认不出的内容形状当成空的）：消息内容不是文字也不是块数组时当作有内容，留着不丢；没有内容才算空
test("message content of an unrecognised shape counts as meaningful; missing content does not", () => {
  assert.equal(hasMeaningfulMessageContent({ type: "image", source: {} }), true);
  assert.equal(hasMeaningfulMessageContent(42), true);
  assert.equal(hasMeaningfulMessageContent(null), false);
  assert.equal(hasMeaningfulMessageContent(undefined), false);
  // 杀变异 J49：数组里既没有 type 也没有 text 的对象（认不出的块）也算有内容
  assert.equal(hasMeaningfulMessageContent([{ data: "x" }]), true);
  assert.equal(hasMeaningfulMessageContent([{ type: "text", text: " " }]), false);
});

// 杀变异 IR07（记回执不核格式）：nonce 必须是 32 位十六进制、摘要 64 位，格式不对的回执不记（记了也永远对不上，还占 32 个名额）
test("an injection receipt must carry a 32-hex nonce and a 64-hex digest", () => {
  const nonce = "b".repeat(32);
  const digest = "c".repeat(64);
  assert.deepEqual(requireInjectionReceipt({ nonce, digest }), { nonce, digest });
  for (const value of [{ nonce: "short", digest }, { nonce, digest: "short" }, { nonce: null, digest }, { digest }, {}]) {
    assert.throws(() => requireInjectionReceipt(value), /injection receipt is invalid/);
  }
});

// 杀变异 CX06（Codex 的会话 id 只认 session_id）：hook 输入没有 session_id、只有 conversation_id 时用 conversation_id
test("Codex hook input without session_id falls back to conversation_id", () => {
  assert.equal(codex.nativeSessionId({ conversation_id: "conv-1" }), "conv-1");
  assert.equal(codex.nativeSessionId({ session_id: "sess-1", conversation_id: "conv-1" }), "sess-1");
});
