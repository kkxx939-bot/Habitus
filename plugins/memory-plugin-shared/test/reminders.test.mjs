import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { hostAdapter as codex } from "../../habitus-memory/scripts/host-adapter.mjs";
import { createContextInjection, renderReminderInstruction } from "../lib/hook-runner.mjs";
import { stripInjectionReceipts } from "../lib/injection-receipt.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";
import { readState } from "../lib/state-store.mjs";

const PLUGINS_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];
const REMINDER = {
  request_id: "e1",
  kind: "写周报",
  text: "周四下午你一般会写周报，这周的还没写。",
  options: ["现在就做", "这次不用", "知道了"],
  deadline: "2026-10-22T18:00:00+08:00",
  lease_until: "2026-10-22T16:22:10+08:00",
  claimed_at: "2026-10-22T15:52:10+08:00",
};

function adapter() {
  return {
    host: "codex",
    protocol: "codex_rollout",
    questionTool: "request_user_input",
    nativeSessionId: (input) => input.session_id,
    conversationId: (input) => `codex-${input.session_id}`,
    prompt: (input) => input.prompt || "",
    successOutput: () => ({}),
    contextOutput: (context) => ({ context }),
    readTranscriptDelta: async () => null,
  };
}

function service({ features = [...FEATURES, "reminders_v1"], claim } = {}) {
  const calls = [];
  return {
    calls,
    capabilities: async () => ({ ok: true, result: { api_version: "1.0", protocols: ["codex_rollout"], features } }),
    claimReminders: async (body) => { calls.push(body); return claim ? claim(body) : { ok: true, result: { reminders: [REMINDER] } }; },
  };
}

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-reminder-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test("the reminder block is wrapped like the memory block and stripped by the same receipts", () => {
  const instruction = renderReminderInstruction([REMINDER], "request_user_input");
  assert.match(instruction, /不是用户说的话/);
  assert.match(instruction, /request_user_input/);
  assert.match(instruction, /『周四下午你一般会写周报，这周的还没写。』/);
  assert.match(instruction, /现在就做 \/ 这次不用 \/ 知道了/);
  const block = createContextInjection(instruction, "2".repeat(32), "habitus-reminder");
  assert.match(block.injection, /^<habitus-reminder receipt="22222222222222222222222222222222">\n/);
  assert.match(block.injection, /<\/habitus-reminder>$/);
  assert.equal(block.injection.includes("<habitus-memory-context"), false);
  const memory = createContextInjection("remembered", "1".repeat(32));
  const transcript = `他说的话\n${memory.injection}\n${block.injection}\n后面的话`;
  const stripped = stripInjectionReceipts(transcript, [memory.receipt, block.receipt]);
  assert.equal(stripped.removed, true);
  assert.equal(stripped.value, "他说的话\n\n\n后面的话");
  assert.throws(() => createContextInjection("x", "3".repeat(32), "habitus-other"), /unknown/);
  // 两条提醒：一件一问、按顺序
  const two = renderReminderInstruction([REMINDER, { ...REMINDER, request_id: "e2", text: "账还没对。" }], "AskUserQuestion");
  assert.match(two, /这 2 件事（一件一问，按顺序）/);
  assert.match(two, /2\. 『账还没对。』/);
});

test("claiming asks the service with the conversation identity, dedupes repeats and is silent on failure", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const remote = service();
  const core = new PluginCore({ stateRoot, timeoutMs: 1000, reminderClaimLimit: 3 }, { service: remote });
  const input = { session_id: "thread-1", prompt: "帮我看看这个报错" };
  const first = await core.claimReminders(adapter(), input);
  assert.equal(first.length, 1);
  assert.equal(remote.calls[0].host, "codex");
  assert.equal(remote.calls[0].session, "codex-thread-1"); // 与送转写时的 conversation_id 同一个身份
  assert.equal(remote.calls[0].limit, 3);
  assert.match(remote.calls[0].at, /^\d{4}-\d{2}-\d{2}T.*Z$/);
  const state = await readState(stateRoot, "codex", "thread-1");
  assert.deepEqual(state.remindersClaimed, [{ request_id: "e1", claimed_at: REMINDER.claimed_at }]);
  // 租约内重取：服务返回同一条（claimed_at 不变）→ 注入过了，不再注入
  assert.deepEqual(await core.claimReminders(adapter(), input), []);
  // 放回之后再取到：claimed_at 变了，是新的一次
  remote.claimReminders = async () => ({ ok: true, result: { reminders: [{ ...REMINDER, claimed_at: "2026-10-22T16:30:00+08:00" }] } });
  assert.equal((await core.claimReminders(adapter(), input)).length, 1);
  // 失败、超时、形状不对：当这一轮没有提醒
  remote.claimReminders = async () => ({ ok: false, status: 0, retryable: true, error: "timeout" });
  assert.deepEqual(await core.claimReminders(adapter(), input), []);
  remote.claimReminders = async () => ({ ok: true, result: { reminders: [{ nope: true }] } });
  assert.deepEqual(await core.claimReminders(adapter(), input), []);
  // 没有人说的话（空提示）不取
  assert.deepEqual(await core.claimReminders(adapter(), { session_id: "thread-1", prompt: "" }), []);
});

test("a service without reminders_v1 is never asked for reminders", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const remote = service({ features: FEATURES });
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: remote });
  assert.deepEqual(await core.claimReminders(adapter(), { session_id: "old", prompt: "hi" }), []);
  assert.equal(remote.calls.length, 0);
});

test("real Codex hook process injects the reminder block after the memory block and records both receipts", async (t) => {
  const stateRoot = await temporaryRoot(t);
  let claimBody = null;
  const server = createServer((request, response) => {
    let payload = "";
    request.on("data", (chunk) => { payload += chunk; });
    request.on("end", () => {
      const result = request.url === "/api/v1/capabilities"
        ? { api_version: "1.0", service_version: "test", protocols: ["codex_rollout"], features: [...FEATURES, "reminders_v1"] }
        : request.url === "/ready"
          ? { ready: true, status: "healthy" }
          : request.url === "/api/v1/memory/recall"
            ? { context: "remembered context", query: "prompt", queries: ["prompt"], memories: [], summaries: [], degradations: [], budget_exhausted: false }
            : request.url === "/api/v1/reminders/claim"
              ? ((claimBody = JSON.parse(payload)), { reminders: [REMINDER] })
              : {};
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ status: "ok", result }));
    });
  });
  try {
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => { server.removeListener("error", reject); resolve(); });
    });
  } catch (error) {
    if (error?.code === "EPERM") { t.skip("sandbox does not permit a loopback listener"); return; }
    throw error;
  }
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const address = server.address();
  const script = join(PLUGINS_ROOT, "habitus-memory", "scripts", "user-prompt.mjs");
  const result = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [script], {
      env: { ...process.env, HABITUS_URL: `http://127.0.0.1:${address.port}`, HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_PLUGIN_TIMEOUT_MS: "1000" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = ""; let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.end(JSON.stringify({ session_id: "hook-reminder", prompt: "prompt" }));
  });
  assert.equal(result.code, 0, result.stderr);
  const context = JSON.parse(result.stdout).hookSpecificOutput.additionalContext;
  const memoryAt = context.indexOf("<habitus-memory-context");
  const reminderAt = context.indexOf("<habitus-reminder receipt=");
  assert.ok(memoryAt >= 0 && reminderAt > memoryAt);
  assert.match(context, /\\u8fd9\\u6b21\\u4e0d\\u7528|这次不用/); // 块里的 JSON 载荷带着选项
  assert.equal(claimBody.session, "codex-hook-reminder");
  assert.match(codex.questionTool, /request_user_input/);
  const stateFiles = await readdir(join(stateRoot, "sessions", "codex"));
  const state = JSON.parse(await readFile(join(stateRoot, "sessions", "codex", stateFiles[0]), "utf8"));
  assert.equal(state.injectionReceipts.length, 2);
  assert.equal(state.remindersClaimed.length, 1);
});
