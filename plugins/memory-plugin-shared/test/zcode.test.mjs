import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { claudeCodeAdapter, selectHostAdapter } from "../../habitus-memory-claude-code/scripts/host-adapter.mjs";
import { createZCodeAdapter, readZCodeSession, zcodeAdapter } from "../../habitus-memory-claude-code/scripts/zcode-adapter.mjs";
import { HARNESS_REGISTRY } from "../../harnesses.mjs";
import { executePluginLifecycle } from "../../install-memory-plugin.mjs";
import { requireHostAdapter } from "../lib/host-adapter.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";

const NO_SQLITE = await import("node:sqlite").then(() => false, () => "Node without node:sqlite");

const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-zcode-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

// ZCode v3.15.1 的会话库：message / part 两张表，data 里拆掉了身份字段（saveMessage / savePart）
async function sessionDatabase(path) {
  const { DatabaseSync } = await import("node:sqlite");
  const db = new DatabaseSync(path);
  db.exec(`
    create table message (id text primary key, session_id text not null, time_created integer not null,
      time_updated integer not null, data text not null, sequence integer);
    create table part (id text primary key, message_id text not null, session_id text not null,
      time_created integer not null, time_updated integer not null, data text not null, sequence integer);
  `);
  const message = db.prepare("insert into message values (?, ?, ?, ?, ?, ?)");
  const part = db.prepare("insert into part values (?, ?, ?, ?, ?, ?, ?)");
  message.run("msg_1", "sess_1", 1, 1, JSON.stringify({ role: "user", time: { created: 1 } }), 0);
  part.run("prt_1", "msg_1", "sess_1", 1, 1, JSON.stringify({ type: "text", text: "帮我看看" }), 0);
  message.run("msg_2", "sess_1", 2, 3, JSON.stringify({ role: "assistant", time: { created: 2, completed: 3 } }), 1);
  part.run("prt_2", "msg_2", "sess_1", 2, 3, JSON.stringify({ type: "text", text: "看完了" }), 0);
  message.run("msg_x", "sess_other", 1, 1, JSON.stringify({ role: "user", time: { created: 1 } }), 0);
  db.close();
}

test("the Claude Code plugin switches to the ZCode adapter inside ZCode", () => {
  assert.equal(selectHostAdapter({}), claudeCodeAdapter);
  assert.equal(selectHostAdapter({ ZCODE_SESSION_ID: "sess_1" }), zcodeAdapter);
  assert.equal(requireHostAdapter(zcodeAdapter), zcodeAdapter);
  assert.equal(zcodeAdapter.conversationId({ session_id: "sess_1" }), "zcode-sess_1");
  assert.equal(zcodeAdapter.questionTool, "AskUserQuestion");
});

test("a ZCode session is read from its database with the identity columns put back", { skip: NO_SQLITE }, async (t) => {
  const root = await temporaryRoot(t);
  const db = join(root, "db.sqlite");
  await sessionDatabase(db);
  const messages = await readZCodeSession("sess_1", db);
  assert.deepEqual(messages.map((item) => [item.info.id, item.info.role, item.parts.map((p) => p.text)]), [
    ["msg_1", "user", ["帮我看看"]],
    ["msg_2", "assistant", ["看完了"]],
  ]);
  assert.equal(messages[0].parts[0].messageID, "msg_1");
});

test("before reading, the ZCode adapter mirrors the session and the turn goes out as protocol zcode", { skip: NO_SQLITE }, async (t) => {
  const root = await temporaryRoot(t);
  const db = join(root, "db.sqlite");
  await sessionDatabase(db);
  const config = { enabled: true, stateRoot: join(root, "state"), timeoutMs: 1000, maxTranscriptBytes: 1 << 20 };
  let reads = 0;
  const adapter = createZCodeAdapter({ dbPath: db, readSession: (id, path) => { reads += 1; return readZCodeSession(id, path); } });
  // ZCode 给的 transcript_path 是临时文件：每一步都换成本地镜像；人说话那一步不读库
  const prompt = await adapter.prepareInput({ session_id: "sess_1", transcript_path: "/tmp/zcode-hook-123.jsonl", prompt: "x" }, config, "user-prompt");
  assert.notEqual(prompt.transcript_path, "/tmp/zcode-hook-123.jsonl");
  assert.equal(reads, 0);
  const input = await adapter.prepareInput({ session_id: "sess_1" }, config, "stop");
  assert.equal(reads, 1);
  assert.equal(input.transcript_path, prompt.transcript_path);
  assert.equal((await readFile(input.transcript_path, "utf8")).trim().split("\n").length, 2);
  const remembered = [];
  const service = {
    capabilities: async () => ({ ok: true, result: { api_version: "1.0", protocols: ["zcode"], features: FEATURES } }),
    cursor: async () => ({ ok: true, result: { next_sequence: 0 } }),
    remember: async (item) => { remembered.push(item); return { ok: true, result: { next_sequence: 2 } }; },
  };
  const core = new PluginCore(config, { service });
  await core.enqueueCapture(adapter, input);
  await core.drain(adapter, input);
  assert.equal(remembered.length, 1);
  assert.equal(remembered[0].protocol, "zcode");
  assert.deepEqual(remembered[0].payload.records.map((item) => item.info.id), ["msg_1", "msg_2"]);
});

test("ZCode registers through its own marketplace commands and borrows Claude Code's manifest", async (t) => {
  const root = await temporaryRoot(t);
  const calls = [];
  const runner = {
    available: (command) => command === "zcode",
    json: () => [],
    run: (command, args) => calls.push([command, ...args].join(" ")),
  };
  const marketplace = join(root, "marketplace");
  await executePluginLifecycle({ action: "install", harnesses: ["zcode"], root: marketplace }, { runner, env: { HOME: join(root, "home") } });
  assert.deepEqual(calls, [
    `zcode plugins marketplace add ${marketplace}`,
    "zcode plugins install habitus-memory@habitus-local",
    "zcode plugins enable habitus-memory@habitus-local",
  ]);
  const manifest = JSON.parse(await readFile(join(marketplace, ".claude-plugin", "marketplace.json"), "utf8"));
  assert.equal(manifest.plugins[0].source, "./plugins/habitus-memory-claude-code");
  assert.equal(HARNESS_REGISTRY.resolve("zc").manifestOf, "claude-code");
});
