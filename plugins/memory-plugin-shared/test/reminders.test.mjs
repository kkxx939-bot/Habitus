import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { appendFile, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { hostAdapter as codex } from "../../habitus-memory/scripts/host-adapter.mjs";
import { loadPluginConfig } from "../lib/config.mjs";
import { createContextInjection, promptInjection, renderReminderInstruction } from "../lib/hook-runner.mjs";
import { stripInjectionReceipts } from "../lib/injection-receipt.mjs";
import { listOutbox } from "../lib/outbox.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";
import { HabitusServiceClient } from "../lib/service-client.mjs";
import { readState } from "../lib/state-store.mjs";
import { isolatedEnv } from "./support/isolated-env.mjs";

const PLUGINS_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];
const REMINDER = {
  request_id: "e1",
  kind: "写周报",
  text: "周四下午你一般会写周报，这周的还没写。",
  options: ["现在就做", "这次不用", "知道了", "已经做过了"],
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
  assert.match(instruction, /现在就做 \/ 这次不用 \/ 知道了 \/ 已经做过了/);
  // 选项个数不写死在话术里：给的是服务端带来的全部选项（裁定 91 第 3 条②加了第四个）
  assert.match(instruction, /给下面列出的全部选项/);
  assert.equal(instruction.includes("三个选项"), false);
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
          ? { ready: true, recall_ready: true, status: "healthy" }
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
  const isolated = await isolatedEnv(t);
  const result = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [script], {
      env: { ...isolated.env, HABITUS_URL: `http://127.0.0.1:${address.port}`, HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_PLUGIN_TIMEOUT_MS: "1000" },
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

// 假的 fetch：routes 里有的路径照常回话，其余接了不回（直到客户端自己按超时放弃）
function fetchStub(routes) {
  const urls = [];
  const fetchImpl = (url, init) => {
    const path = new URL(url).pathname;
    urls.push(path);
    if (Object.hasOwn(routes, path)) {
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ status: "ok", result: routes[path] }) });
    }
    return new Promise((_resolve, reject) => {
      init.signal.addEventListener("abort", () => reject(new Error("aborted")));
    });
  };
  return { urls, fetchImpl };
}

async function promptCore(t, routes, env = {}) {
  const stateRoot = await temporaryRoot(t);
  const config = loadPluginConfig({ HABITUS_URL: "http://127.0.0.1:9", HABITUS_PLUGIN_STATE_DIR: stateRoot, ...env });
  const stub = fetchStub(routes);
  const core = new PluginCore(config, { service: new HabitusServiceClient(config, stub.fetchImpl) });
  return { config, core, urls: stub.urls };
}

const CAPABILITIES = { api_version: "1.0", service_version: "test", protocols: ["codex_rollout"], features: [...FEATURES, "reminders_v1"] };

// B6：服务接了连接不回话时，UserPromptSubmit 原先先后问两次 capabilities、各等 15 s（30.1 s，撞上 hooks.json 的 30 s）
test("a service that never answers costs the prompt one capabilities request, inside the prompt budget", { timeout: 10_000 }, async (t) => {
  const { config, core, urls } = await promptCore(t, {}, { HABITUS_PLUGIN_PROMPT_BUDGET_MS: "1500" });
  assert.equal(config.timeoutMs, 15_000); // 后台路径的超时不变，预算只管这一路
  const started = Date.now();
  const injection = await promptInjection(core, adapter(), { session_id: "hung", prompt: "你好" }, { config, hook: "user-prompt" });
  const elapsed = Date.now() - started;
  assert.equal(injection, "");
  assert.deepEqual(urls, ["/api/v1/capabilities"]);
  // 预算管的是等服务的时间；另留 1 s 给本地的会话锁与写回执（修复前这里是 30 s）
  assert.ok(elapsed < 1500 + 1000, `user-prompt path took ${elapsed} ms, budget 1500 ms`);
});

test("a recall that never answers still leaves the reminder claim its share of the prompt budget", { timeout: 10_000 }, async (t) => {
  const { config, core, urls } = await promptCore(t, {
    "/api/v1/capabilities": CAPABILITIES,
    "/ready": { ready: true, recall_ready: true, status: "healthy" },
    "/api/v1/reminders/claim": { reminders: [REMINDER] },
  }, { HABITUS_PLUGIN_PROMPT_BUDGET_MS: "2000", HABITUS_PLUGIN_REMINDER_TIMEOUT_MS: "500" });
  const started = Date.now();
  const injection = await promptInjection(core, adapter(), { session_id: "slow-recall", prompt: "你好" }, { config, hook: "user-prompt" });
  const elapsed = Date.now() - started;
  assert.match(injection, /^<habitus-reminder receipt=/);
  assert.deepEqual(urls, ["/api/v1/capabilities", "/ready", "/api/v1/memory/recall", "/api/v1/reminders/claim"]);
  assert.ok(elapsed < 2000 + 1000, `user-prompt path took ${elapsed} ms, budget 2000 ms`);
});

test("the default prompt budget leaves room inside every UserPromptSubmit hook timeout", async () => {
  const budget = loadPluginConfig({}).promptBudgetMs;
  for (const plugin of ["habitus-memory", "habitus-memory-claude-code", "habitus-memory-dsh"]) {
    const manifest = JSON.parse(await readFile(join(PLUGINS_ROOT, plugin, "hooks", "hooks.json"), "utf8"));
    for (const entry of manifest.hooks.UserPromptSubmit) {
      for (const hook of entry.hooks) {
        // 留 5 s 给 node 起进程、会话锁与写回执
        assert.ok(budget + 5_000 <= hook.timeout * 1000, `${plugin}: budget ${budget} ms vs hook timeout ${hook.timeout} s`);
      }
    }
  }
});

// 杀变异 PC07（不按配置的条数取）：一次取几条由 HABITUS_PLUGIN_REMINDER_LIMIT 定
test("claiming asks for as many reminders as the configuration allows", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const remote = service();
  const config = loadPluginConfig({ HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_PLUGIN_REMINDER_LIMIT: "5" });
  const core = new PluginCore(config, { service: remote });
  await core.claimReminders(adapter(), { session_id: "limit-1", prompt: "hi" });
  assert.equal(remote.calls[0].limit, 5);
});

// 杀变异 PC06（去重只记 1 条）：一轮取到几条，几条都记下；租约内服务把它们原样再给一次，一条也不再注入。去重记录有上限（64 条）
test("every reminder injected is remembered for dedupe, up to the kept window", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const two = [REMINDER, { ...REMINDER, request_id: "e2", text: "账还没对。" }];
  const remote = service({ claim: () => ({ ok: true, result: { reminders: two } }) });
  const core = new PluginCore({ stateRoot, timeoutMs: 1000, reminderClaimLimit: 3 }, { service: remote });
  const input = { session_id: "dedupe-1", prompt: "hi" };
  assert.equal((await core.claimReminders(adapter(), input)).length, 2);
  assert.deepEqual(await core.claimReminders(adapter(), input), []);
  let next = 0;
  remote.claimReminders = async () => ({
    ok: true, result: { reminders: Array.from({ length: 10 }, () => ({ ...REMINDER, request_id: `r${(next += 1)}` })) },
  });
  for (let round = 0; round < 7; round += 1) await core.claimReminders(adapter(), input);
  const kept = (await readState(stateRoot, "codex", "dedupe-1")).remindersClaimed;
  assert.equal(kept.length, 64);
  assert.equal(kept.at(-1).request_id, "r70");
});

// R14：召回 / 取提醒没成而没抛异常的（连不上、未就绪、非 2xx）也各记一条操作日志、带原因；同一会话接连同一原因只记第一次，
// 原因变了、隔了一小时、或这一环节成过一次之后再出错，照常记
async function operations(stateRoot) {
  try {
    return (await readFile(join(stateRoot, "logs", "operations.jsonl"), "utf8")).trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
}

test("recall and reminder failures that do not throw are logged with their reason, once per run of the same reason", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { stateRoot, timeoutMs: 1000, promptBudgetMs: 5000, reminderTimeoutMs: 500, operationLogMaxBytes: 1 << 20, operationLogKeep: 2 };
  let mode = "down";
  const remote = {
    capabilities: async () => (mode === "down"
      ? { ok: false, status: 0, retryable: true, error: "fetch failed" }
      : { ok: true, status: 200, result: { api_version: "1.0", protocols: ["codex_rollout"], features: [...FEATURES, "reminders_v1"] } }),
    ready: async () => (mode === "not-ready"
      ? { ok: true, status: 503, result: { ready: false, recall_ready: false, status: "degraded" } }
      : { ok: true, status: 200, result: { ready: true, recall_ready: true, status: "healthy" } }),
    recall: async () => (mode === "error"
      ? { ok: false, status: 500, retryable: true, error: { code: "INTERNAL", message: "boom" } }
      : { ok: true, status: 200, result: { context: "" } }),
    claimReminders: async () => (mode === "error"
      ? { ok: false, status: 500, retryable: true, error: { code: "INTERNAL", message: "boom" } }
      : { ok: true, status: 200, result: { reminders: [] } }),
  };
  const core = new PluginCore(config, { service: remote });
  const say = () => promptInjection(core, adapter(), { session_id: "s", prompt: "你好" }, { config, hook: "user-prompt" });
  const logged = async () => (await operations(stateRoot)).map((item) => `${item.stage}:${item.reason}`);
  await say();
  await say();
  assert.deepEqual(await logged(), ["recall:service_unavailable", "claim_reminders:service_unavailable"]);
  assert.equal((await operations(stateRoot))[0].detail, "fetch failed");
  mode = "not-ready";
  await say();
  assert.deepEqual((await logged()).slice(2), ["recall:service_not_ready"]);
  mode = "error";
  await say();
  await say();
  assert.deepEqual((await logged()).slice(3), ["recall:request_failed", "claim_reminders:request_failed"]);
  assert.match((await operations(stateRoot)).at(-1).detail, /HTTP 500: INTERNAL: boom/);
  mode = "ok";
  await say();
  assert.equal((await logged()).length, 5);
  mode = "error";
  await say();
  assert.deepEqual((await logged()).slice(5), ["recall:request_failed", "claim_reminders:request_failed"]);
});

// T10（裁定 75 第 3 条）：服务回的 retryable 照记进操作日志——请求被拒、不可重试（400 INVALID_ARGUMENT）记成 false，
// 服务说可重试的（500）记成 true；不是写死的 true
test("a failed prompt-time request is logged with the retryable the service answered", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { stateRoot, timeoutMs: 1000, promptBudgetMs: 5000, reminderTimeoutMs: 500, operationLogMaxBytes: 1 << 20, operationLogKeep: 2 };
  let retryable = false;
  const refused = () => ({ ok: false, status: retryable ? 500 : 400, retryable, error: { code: retryable ? "INTERNAL" : "INVALID_ARGUMENT", message: "no" } });
  const remote = {
    capabilities: async () => ({ ok: true, status: 200, result: { api_version: "1.0", protocols: ["codex_rollout"], features: [...FEATURES, "reminders_v1"] } }),
    ready: async () => ({ ok: true, status: 200, result: { ready: true, recall_ready: true, status: "healthy" } }),
    recall: async () => refused(),
    claimReminders: async () => refused(),
  };
  const core = new PluginCore(config, { service: remote });
  await promptInjection(core, adapter(), { session_id: "s", prompt: "你好" }, { config, hook: "user-prompt" });
  retryable = true;
  await promptInjection(core, adapter(), { session_id: "t", prompt: "你好" }, { config, hook: "user-prompt" });
  assert.deepEqual((await operations(stateRoot)).map((item) => `${item.stage}:${item.retryable}`), [
    "recall:false", "claim_reminders:false", "recall:true", "claim_reminders:true",
  ]);
});

// T4 / 杀变异 J14、J15、J18：经真实 HTTP 与真实客户端，每个请求都把"连不上、不回话"（status 0）与"回了非 2xx"分开记：
// capabilities、ready、recall、取提醒同一个口径；服务没有 reminders_v1 不算取提醒失败
test("every prompt-time request tells an unreachable or silent service apart from one that answers with an error", async (t) => {
  const stateRoot = await temporaryRoot(t);
  let behaviour = {};
  const capabilities = (features) => ({ api_version: "1.0", service_version: "test", protocols: ["codex_rollout"], features });
  const server = createServer((request, response) => {
    request.resume();
    request.on("end", () => {
      const mode = behaviour[request.url] ?? "ok";
      if (mode === "silent") return; // 接了连接不回话：等客户端超时
      if (mode === "500") {
        response.writeHead(500, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ status: "error", error: { code: "INTERNAL", message: "boom", retryable: true } }));
        return;
      }
      const result = request.url === "/api/v1/capabilities"
        ? capabilities(mode === "no-reminders" ? FEATURES : [...FEATURES, "reminders_v1"])
        : request.url === "/ready" ? { ready: true, recall_ready: true, status: "healthy" }
          : request.url === "/api/v1/memory/recall" ? { context: "" } : { reminders: [] };
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
  t.after(() => { server.closeAllConnections(); server.close(); });
  const config = {
    stateRoot, baseUrl: `http://127.0.0.1:${server.address().port}`, timeoutMs: 300, promptBudgetMs: 2000, reminderTimeoutMs: 300,
  };
  const core = new PluginCore(config, { service: new HabitusServiceClient(config) });
  const failures = async (modes) => {
    behaviour = modes;
    const turn = core.beginPrompt();
    const input = { session_id: "s", prompt: "你好" };
    await core.recall(adapter(), input, turn);
    await core.claimReminders(adapter(), input, turn);
    return turn.failures.map((item) => `${item.stage}:${item.reason}`);
  };
  const silent = { "/api/v1/memory/recall": "silent", "/api/v1/reminders/claim": "silent" };
  assert.deepEqual(await failures(silent), ["recall:service_unavailable", "claim_reminders:service_unavailable"]);
  const answered = { "/api/v1/memory/recall": "500", "/api/v1/reminders/claim": "500" };
  assert.deepEqual(await failures(answered), ["recall:request_failed", "claim_reminders:request_failed"]);
  assert.deepEqual(await failures({ "/ready": "500" }), ["recall:request_failed"]);
  assert.deepEqual(await failures({ "/ready": "silent" }), ["recall:service_unavailable"]);
  assert.deepEqual(await failures({ "/api/v1/capabilities": "500" }), ["recall:request_failed", "claim_reminders:request_failed"]);
  assert.deepEqual(await failures({ "/api/v1/capabilities": "no-reminders" }), []);
  // 不回话的那一条不写成 "HTTP 0"
  behaviour = silent;
  const turn = core.beginPrompt();
  await core.claimReminders(adapter(), { session_id: "s", prompt: "你好" }, turn);
  assert.doesNotMatch(turn.failures[0].detail, /HTTP 0/);
});

test("the same failure reason is logged again once an hour has passed", async (t) => {
  const { writeRepeatedFailureLogs, REPEAT_RELOG_MS } = await import("../lib/operation-log.mjs");
  const stateRoot = await temporaryRoot(t);
  const config = { stateRoot, operationLogMaxBytes: 1 << 20, operationLogKeep: 2 };
  const event = { hook: "user-prompt", failures: [{ stage: "recall", reason: "service_unavailable", detail: "fetch failed" }], settled: [] };
  const input = { session_id: "s" };
  const start = Date.now();
  await writeRepeatedFailureLogs(config, adapter(), input, event, start);
  await writeRepeatedFailureLogs(config, adapter(), input, event, start + REPEAT_RELOG_MS - 1);
  assert.equal((await operations(stateRoot)).length, 1);
  await writeRepeatedFailureLogs(config, adapter(), input, event, start + REPEAT_RELOG_MS);
  assert.equal((await operations(stateRoot)).length, 2);
  assert.equal(REPEAT_RELOG_MS, 60 * 60 * 1000);
});

// 杀变异 PC16 / PC18（预算用完后仍按各自的完整超时发请求）：capabilities 一问就用完了这一轮的预算，后面的 ready、recall、取提醒一个都不发，
// 两个环节各记一条"预算用完"
test("once the prompt budget is spent, no further request goes out and both stages say so", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const config = { stateRoot, timeoutMs: 1000, promptBudgetMs: 200, reminderTimeoutMs: 100, operationLogMaxBytes: 1 << 20, operationLogKeep: 2 };
  const calls = [];
  const remote = {
    capabilities: async () => {
      calls.push("capabilities");
      await new Promise((resolve) => setTimeout(resolve, 300));
      return { ok: true, status: 200, result: { api_version: "1.0", protocols: ["codex_rollout"], features: [...FEATURES, "reminders_v1"] } };
    },
    ready: async () => { calls.push("ready"); return { ok: true, status: 200, result: { ready: true, recall_ready: true } }; },
    recall: async () => { calls.push("recall"); return { ok: true, status: 200, result: { context: "x" } }; },
    claimReminders: async () => { calls.push("claim"); return { ok: true, status: 200, result: { reminders: [REMINDER] } }; },
  };
  const core = new PluginCore(config, { service: remote });
  const injection = await promptInjection(core, adapter(), { session_id: "spent", prompt: "你好" }, { config, hook: "user-prompt" });
  assert.equal(injection, "");
  assert.deepEqual(calls, ["capabilities"]);
  assert.deepEqual((await operations(stateRoot)).map((item) => `${item.stage}:${item.reason}`), [
    "recall:prompt_budget_exhausted", "claim_reminders:prompt_budget_exhausted",
  ]);
});

// 杀变异 PC21（默认预算改了）：README 两份写的超时与预算默认值就是插件实际用的
test("the timeout and budget defaults the READMEs give are the ones the plugin uses", async () => {
  const defaults = loadPluginConfig({});
  const expected = {
    HABITUS_PLUGIN_TIMEOUT_MS: defaults.timeoutMs,
    HABITUS_PLUGIN_PROMPT_BUDGET_MS: defaults.promptBudgetMs,
    HABITUS_PLUGIN_REMINDER_TIMEOUT_MS: defaults.reminderTimeoutMs,
  };
  for (const readme of ["README.md", "README.zh-CN.md"]) {
    const text = await readFile(join(PLUGINS_ROOT, "..", readme), "utf8");
    for (const [name, value] of Object.entries(expected)) {
      const row = text.match(new RegExp(`^\\| \`${name}\` \\| \`(\\d+)\` \\|`, "m"));
      assert.ok(row, `${readme} has no row for ${name}`);
      assert.equal(Number(row[1]), value, `${readme}: ${name}`);
    }
  }
  assert.equal(defaults.promptBudgetMs, 20_000);
});

// T16（README：注入的内容不会被当成你说的话送回服务）：读转写一直失败（一轮的文字超过读取额度）期间召回、取提醒照常注入；
// 这些块的回执一条不丢（平时只留最近 32 条），调大额度读成之后一个注入块都不进 outbox，回执再收回到 32 条
test("blocks injected while the transcript keeps failing to read are all stripped once it reads again", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = join(stateRoot, "rollout.jsonl");
  const line = (value) => `${JSON.stringify(value)}\n`;
  const message = (role, content, extra = {}) => line({ type: "response_item", payload: { type: "message", role, content, ...extra } });
  await writeFile(transcript, message("user", `第一句${"z".repeat(1500)}`) + message("user", "z".repeat(1500)) + message("assistant", "好", { phase: "final_answer" }));
  let recalled = 0;
  let claimed = 0;
  const remote = {
    capabilities: async () => ({ ok: true, result: { api_version: "1.0", protocols: ["codex_rollout"], features: [...FEATURES, "reminders_v1"] } }),
    ready: async () => ({ ok: true, result: { ready: true, recall_ready: true } }),
    recall: async () => ({ ok: true, result: { context: `MEM-${++recalled}` } }),
    claimReminders: async () => {
      claimed += 1;
      return { ok: true, result: { reminders: [{ ...REMINDER, request_id: `q${claimed}`, text: `REM-${claimed}`, claimed_at: `2026-10-26T00:00:${String(claimed).padStart(2, "0")}Z` }] } };
    },
  };
  const input = { session_id: "s16", transcript_path: transcript };
  const base = { stateRoot, timeoutMs: 1000, promptBudgetMs: 5000, reminderTimeoutMs: 500 };
  const failing = new PluginCore({ ...base, maxTranscriptBytes: 2048 }, { service: remote });
  // 17 轮：每轮一块记忆、一块提醒，共 34 块，超过平时留的 32 条
  for (let index = 0; index < 17; index += 1) {
    const injection = await promptInjection(failing, codex, { ...input, prompt: `q${index}` }, { config: failing.config, hook: "user-prompt" });
    await appendFile(transcript, message("user", injection) + message("user", `q${index}`) + message("assistant", `a${index}`, { phase: "final_answer" }));
    await assert.rejects(failing.enqueueCapture(codex, input), /read limit/);
  }
  assert.equal((await readState(stateRoot, "codex", "s16")).injectionReceipts.length, 34);
  await new PluginCore({ ...base, maxTranscriptBytes: 1 << 20 }, { service: remote }).enqueueCapture(codex, input);
  const sent = JSON.stringify((await listOutbox(stateRoot, "codex", "s16")).map((item) => item.payload.records));
  assert.match(sent, /q16/);
  assert.doesNotMatch(sent, /habitus-memory-context|habitus-reminder|MEM-|REM-/);
  assert.equal((await readState(stateRoot, "codex", "s16")).injectionReceipts.length, 32);
});
