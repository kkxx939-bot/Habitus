import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { hostAdapter as claude } from "../../habitus-memory-claude-code/scripts/host-adapter.mjs";
import { hostAdapter as codex } from "../../habitus-memory/scripts/host-adapter.mjs";
import { createContextInjection } from "../lib/hook-runner.mjs";

const FIXTURES = join(new URL("./fixtures", import.meta.url).pathname);

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
    { type: "response_item", payload: { type: "message", role: "assistant", content: [] } },
  ]);
  assert.equal(await codex.readTranscriptDelta({ transcript_path: pending }, null, config), null);

  const complete = await transcript(t, [
    { type: "response_item", payload: { type: "custom_tool_call", call_id: "c1", name: "shell", input: "pwd" } },
    { type: "response_item", payload: { type: "custom_tool_call_output", call_id: "c1", output: "/tmp" } },
    { type: "response_item", payload: { type: "message", role: "assistant", content: [] } },
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
    { type: "response_item", payload: { type: "message", role: "assistant", content: "not terminal" } },
  ]);
  assert.equal(await codex.readTranscriptDelta({ transcript_path: path }, null, config), null);

  const complete = await transcript(t, [
    { type: "response_item", payload: { type: "agent_message", author: "worker", content: [{ type: "output_text", text: "evidence" }] } },
    { type: "response_item", payload: { type: "sub_agent_activity", event_id: "e1", kind: "completed" } },
    { type: "response_item", payload: { type: "tool_search_call", call_id: "s1", arguments: { query: "memory" } } },
    { type: "response_item", payload: { type: "tool_search_output", call_id: "s1", tools: [{ name: "github" }] } },
    { type: "response_item", payload: { type: "message", role: "assistant", content: "terminal" } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: complete }, null, config);
  assert.equal(delta.records.length, 5);
});

test("checked-in host transcript fixtures remain consumable", async () => {
  const codexDelta = await codex.readTranscriptDelta(
    { transcript_path: join(FIXTURES, "codex-native.jsonl") }, null, config,
  );
  const claudeDelta = await claude.readTranscriptDelta(
    { session_id: "parent", agent_id: "fixture", agent_transcript_path: join(FIXTURES, "claude-subagent.jsonl") },
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

test("Claude subagent stop captures its own transcript under an isolated session identity", async (t) => {
  const path = await transcript(t, [
    { type: "user", message: { role: "user", content: "investigate" } },
    { type: "assistant", message: { role: "assistant", stop_reason: "end_turn", content: "evidence" } },
  ]);
  const input = { session_id: "parent", agent_id: "worker-1", agent_transcript_path: path };
  const delta = await claude.readTranscriptDelta(input, null, config);
  assert.equal(claude.nativeSessionId(input), "parent.subagent.worker-1");
  assert.equal(claude.conversationId(input), "claude-parent.subagent.worker-1");
  assert.equal(delta.records.length, 2);
});

test("injected memory context is consumed but excluded from captured payload", async (t) => {
  const prepared = createContextInjection("hidden", "2".repeat(32));
  const path = await transcript(t, [
    { type: "response_item", payload: { type: "message", role: "user", content: prepared.injection } },
    { type: "response_item", payload: { type: "message", role: "assistant", content: "done" } },
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
    { type: "response_item", payload: { type: "message", role: "assistant", content: "done" } },
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
    { type: "response_item", payload: { type: "message", role: "assistant", content: "done" } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: path }, null, config, [prepared.receipt]);
  assert.equal(delta.records[0].payload.content, modified);
});

test("literal context marker without an injection receipt remains user content", async (t) => {
  const path = await transcript(t, [
    { type: "response_item", payload: { type: "message", role: "user", content: "literal <habitus-memory-context> fact" } },
    { type: "response_item", payload: { type: "message", role: "assistant", content: "done" } },
  ]);
  const delta = await codex.readTranscriptDelta({ transcript_path: path }, null, config, []);
  assert.equal(delta.records.length, 2);
});
