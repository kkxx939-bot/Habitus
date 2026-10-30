// runHook 的分发：真跑宿主的 hook 脚本进程（假服务 + 临时状态目录），看每个 hook 动作落到盘上、送到服务的是什么。
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, rm, utimes, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";

import { encodeSegment } from "../../habitus-memory-dsh/scripts/session-log.mjs";
import { loadPluginConfig } from "../lib/config.mjs";
import { listOutbox } from "../lib/outbox.mjs";
import { readState, sessionKey, writeState } from "../lib/state-store.mjs";
import { isolatedEnv } from "./support/isolated-env.mjs";

const PLUGINS_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];
const UNREACHABLE = "http://127.0.0.1:1";
// hook 脚本（与它起的后台 worker）跑在隔离的 HOME / PATH 里
const ISOLATED = await isolatedEnv({ after });

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-hook-runner-test-"));
  t.after(() => rm(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 }));
  return root;
}

function runScript(plugin, script, input, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [join(PLUGINS_ROOT, plugin, "scripts", script)], {
      env: { ...ISOLATED.env, HABITUS_PLUGIN_DEBUG: "0", HABITUS_HOOK_WORKER: "", ...env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.on("error", () => {});
    child.stdin.end(typeof input === "string" ? input : JSON.stringify(input));
  });
}

async function waitFor(predicate, what, timeoutMs = 8_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  assert.fail(`timed out waiting for ${what}`);
}

// 假服务：认得会话协议、回游标 0、remember 一批就把序号往前推一格
async function fakeService(t, protocols) {
  const calls = [];
  const server = createServer((request, response) => {
    let body = "";
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      const path = new URL(request.url, "http://x").pathname;
      const payload = body ? JSON.parse(body) : null;
      calls.push({ path, body: payload });
      const result = path === "/api/v1/capabilities"
        ? { api_version: "1.0", service_version: "test", protocols, features: FEATURES }
        : path === "/api/v1/memory/conversations/cursor"
          ? { next_sequence: 0 }
          : path === "/api/v1/memory/remember"
            ? { next_sequence: payload.start_sequence + 1 }
            : null;
      response.writeHead(result ? 200 : 404, { "Content-Type": "application/json" });
      response.end(JSON.stringify(result ? { status: "ok", result } : { status: "error", error: { code: "NOT_FOUND", retryable: false } }));
    });
  });
  try {
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => { server.removeListener("error", reject); resolve(); });
    });
  } catch (error) {
    if (error?.code === "EPERM") { t.skip("sandbox does not permit a loopback listener"); return null; }
    throw error;
  }
  t.after(() => new Promise((resolve) => server.close(resolve)));
  return { url: `http://127.0.0.1:${server.address().port}`, calls };
}

// Claude Code 的一轮：人说一句、助手答完（end_turn）
async function claudeTranscript(root, text = "帮我看看这个报错") {
  const path = join(root, "transcript.jsonl");
  await writeFile(path, [
    { type: "user", message: { role: "user", content: text } },
    { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "好的" }], stop_reason: "end_turn" } },
  ].map((value) => `${JSON.stringify(value)}\n`).join(""));
  return path;
}

async function queuedFiles(stateRoot, host) {
  try { return (await readdir(join(stateRoot, "outbox", host), { recursive: true })).filter((name) => name.endsWith(".json")).length; }
  catch (error) { if (error?.code === "ENOENT") return 0; throw error; }
}

async function sessionFiles(stateRoot, host) {
  try { return await readdir(join(stateRoot, "sessions", host)); }
  catch (error) { if (error?.code === "ENOENT") return []; throw error; }
}

// 后台 worker 把这一批送完、放掉会话锁之后才算这次 Stop 收场（测试结束前等它，免得它往删掉的目录里写）。
// ack 之后它还会再拿一次锁看看还有没有下一条，所以要"outbox 空、没有锁"连续保持一小段时间
// （只数文件、不读内容：worker 正在 ack，列出来的文件读的时候可能已经删了）
async function drainedByWorker(stateRoot, host) {
  let quietSince = null;
  await waitFor(async () => {
    const idle = (await queuedFiles(stateRoot, host)) === 0
      && !(await sessionFiles(stateRoot, host)).some((name) => name.includes(".lock"));
    quietSince = idle ? quietSince ?? Date.now() : null;
    return quietSince != null && Date.now() - quietSince >= 300;
  }, "the worker to drain the outbox and release the session lock");
}

// 杀变异 HR11（Stop 不入队）：Stop 是主采集路径，这一轮要进 outbox，再由后台 worker 送到服务
test("Stop queues the finished turn and its worker delivers it", async (t) => {
  const service = await fakeService(t, ["claude_code"]);
  if (!service) return;
  const stateRoot = await temporaryRoot(t);
  const transcript = await claudeTranscript(stateRoot);
  const result = await runScript("habitus-memory-claude-code", "stop.mjs", { session_id: "stop-1", transcript_path: transcript }, {
    HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: stateRoot,
  });
  assert.equal(result.code, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), {});
  await waitFor(() => service.calls.some((call) => call.path === "/api/v1/memory/remember"), "the worker's remember");
  await drainedByWorker(stateRoot, "claude-code");
  const remembered = service.calls.find((call) => call.path === "/api/v1/memory/remember").body;
  assert.equal(remembered.conversation_id, "claude-stop-1");
  assert.equal(remembered.payload.records.length, 2);
  assert.equal((await readState(stateRoot, "claude-code", "stop-1")).nextSequence, 1);
});

// 杀变异 HR07（PreCompact 不入队）：压缩前先把这一轮落进 outbox，服务连不上也不丢
test("PreCompact queues the turn before the transcript is compacted, even when the service is down", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = await claudeTranscript(stateRoot);
  const result = await runScript("habitus-memory-claude-code", "pre-compact.mjs", { session_id: "compact-1", transcript_path: transcript }, {
    HABITUS_URL: UNREACHABLE, HABITUS_PLUGIN_STATE_DIR: stateRoot,
  });
  assert.equal(result.code, 0, result.stderr);
  const queued = await listOutbox(stateRoot, "claude-code", "compact-1");
  assert.equal(queued.length, 1);
  assert.equal(queued[0].payload.records.length, 2);
});

// B3：转写比 HABITUS_PLUGIN_MAX_TRANSCRIPT_BYTES 大时，这次 Stop 照样从游标往后一段段读完、全部落进 outbox（原先整份读、每次抛错永久停采）
test("a transcript over the read limit is still captured in full, in increments, by one PreCompact", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const path = join(stateRoot, "transcript.jsonl");
  const turns = Array.from({ length: 8 }, (_, index) => [
    { type: "user", message: { role: "user", content: `第 ${index} 句 ${"字".repeat(100)}` } },
    { type: "assistant", message: { role: "assistant", content: [{ type: "text", text: `答 ${index}` }], stop_reason: "end_turn" } },
  ]).flat();
  await writeFile(path, turns.map((value) => `${JSON.stringify(value)}\n`).join(""));
  const result = await runScript("habitus-memory-claude-code", "pre-compact.mjs", { session_id: "oversize-1", transcript_path: path }, {
    HABITUS_URL: UNREACHABLE, HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_PLUGIN_MAX_TRANSCRIPT_BYTES: "1024",
  });
  assert.equal(result.code, 0, result.stderr);
  const queued = await listOutbox(stateRoot, "claude-code", "oversize-1");
  assert.ok(queued.length > 1, `${queued.length} item(s)`);
  assert.deepEqual(queued.flatMap((item) => item.payload.records), turns);
  const log = await readFile(join(stateRoot, "logs", "operations.jsonl"), "utf8").catch(() => "");
  assert.ok(!log.includes("transcript"), log);
});

// 杀变异 HR10（不调 prepareInput）：dsh 的转写不在 hook 给的路径里（桥给的是空串），要适配器先镜像出本地转写
test("a dsh Stop mirrors the session log before reading it, so the turn reaches the service", async (t) => {
  const service = await fakeService(t, ["dsh"]);
  if (!service) return;
  const stateRoot = await temporaryRoot(t);
  const dshHome = join(stateRoot, "dsh");
  const directory = join(dshHome, "sessions", "--w--", encodeSegment("dsh-1"));
  await mkdir(directory, { recursive: true });
  const line = (value) => `${JSON.stringify(value)}\n`;
  await writeFile(join(directory, "session.jsonl"), [
    line({ id: "dsh-1", createdAt: "2026-10-22T07:00:00Z", cwd: "/w" }),
    line({ type: "turn/start", seq: 1, time: 1792998000001, data: {} }),
    line({ type: "user/message", seq: 2, time: 1792998000002, data: { text: "你好" } }),
    line({ type: "turn/end", seq: 3, time: 1792998000003, data: {} }),
  ].join(""));
  const result = await runScript("habitus-memory-dsh", "stop.mjs", { session_id: "dsh-1", transcript_path: "" }, {
    HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: stateRoot, DSH_HOME: dshHome,
  });
  assert.equal(result.code, 0, result.stderr);
  await waitFor(() => service.calls.some((call) => call.path === "/api/v1/memory/remember"), "the worker's remember");
  await drainedByWorker(stateRoot, "dsh");
  const remembered = service.calls.find((call) => call.path === "/api/v1/memory/remember").body;
  assert.equal(remembered.protocol, "dsh");
  assert.deepEqual(remembered.payload.records.map((record) => record.seq), [1, 2, 3]);
});

// Q3：子代理的会话不进记忆。各命令 hook 宿主的子代理里每个 hook（前台与后台 worker）都什么都不送、不召回、不留会话状态：
// Codex 从转写开头的 session_meta 认（子线程 rollout 的 session_id 是父线程的）或 hook 输入的 agent_id；Claude Code 认 agent_id；
// dsh 认会话头的 parentSession
test("hooks fired for a subagent session send nothing and leave no state", async (t) => {
  const service = await fakeService(t, ["codex_rollout", "claude_code", "dsh"]);
  if (!service) return;
  const stateRoot = await temporaryRoot(t);
  const line = (value) => `${JSON.stringify(value)}\n`;
  const codexChild = join(stateRoot, "rollout-child.jsonl");
  await writeFile(codexChild, [
    line({ type: "session_meta", payload: {
      session_id: "parent-thread", id: "child-thread", thread_source: "subagent",
      source: { subagent: { thread_spawn: { parent_thread_id: "parent-thread", depth: 1 } } },
    } }),
    line({ type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "去查一下" }] } }),
    line({ type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: [{ type: "output_text", text: "查到了" }] } }),
  ].join(""));
  const codexMain = join(stateRoot, "rollout-main.jsonl");
  await writeFile(codexMain, (await readFile(codexChild, "utf8")).split("\n").slice(1).join("\n"));
  const dshHome = join(stateRoot, "dsh");
  const dshDirectory = join(dshHome, "sessions", "--w--", encodeSegment("dsh-child"));
  await mkdir(dshDirectory, { recursive: true });
  await writeFile(join(dshDirectory, "session.jsonl"), [
    line({ id: "dsh-child", createdAt: "2026-10-22T07:00:00Z", cwd: "/w", parentSession: "dsh-main", origin: "subagent" }),
    line({ type: "turn/start", seq: 1, time: 1792998000001, data: {} }),
    line({ type: "user/message", seq: 2, time: 1792998000002, data: { text: "去查" } }),
    line({ type: "turn/end", seq: 3, time: 1792998000003, data: {} }),
  ].join(""));
  const claudeTranscriptPath = await claudeTranscript(stateRoot, "去查一下");
  const commandHooks = ["session-start.mjs", "user-prompt.mjs", "stop.mjs", "pre-compact.mjs", "session-end.mjs"];
  const cases = [
    ["habitus-memory", commandHooks, { session_id: "parent-thread", transcript_path: codexChild, prompt: "去查一下" }],
    ["habitus-memory", commandHooks, { session_id: "parent-thread", agent_id: "a1", transcript_path: codexMain, prompt: "去查一下" }],
    ["habitus-memory-claude-code", commandHooks, { session_id: "cc-parent", agent_id: "a1", transcript_path: claudeTranscriptPath, prompt: "去查一下" }],
    ["habitus-memory-dsh", ["session-start.mjs", "user-prompt.mjs", "stop.mjs"], { session_id: "dsh-child", transcript_path: "", prompt: "去查" }],
  ];
  for (const [plugin, scripts, input] of cases) {
    for (const script of scripts) {
      for (const worker of ["", "1"]) {
        const result = await runScript(plugin, script, input, {
          HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: stateRoot, DSH_HOME: dshHome, HABITUS_HOOK_WORKER: worker,
        });
        assert.equal(result.code, 0, `${plugin}/${script}: ${result.stderr}`);
        if (!worker) assert.deepEqual(JSON.parse(result.stdout), {}, `${plugin}/${script}`);
      }
    }
  }
  // 后台 worker 若被起了也要等它收场再看
  await new Promise((resolve) => setTimeout(resolve, 500));
  assert.deepEqual(service.calls.map((call) => call.path), []);
  const left = (await readdir(stateRoot)).filter((name) => ["sessions", "outbox", "transcripts"].includes(name));
  assert.deepEqual(left, []);
});

// U23（裁定 80 第 6 条）：非交互会话（脚本 / SDK 调宿主）不是他在说话，与子代理同样什么都不送、不召回、不领提醒、不留会话状态：
// Codex `codex exec` 从转写开头 session_meta 的 source = "exec" 认；Claude Code `claude -p` 从 hook 进程继承的
// CLAUDE_CODE_ENTRYPOINT = sdk-cli 认
test("hooks fired for a non-interactive session send nothing and leave no state", async (t) => {
  const service = await fakeService(t, ["codex_rollout", "claude_code"]);
  if (!service) return;
  const stateRoot = await temporaryRoot(t);
  const line = (value) => `${JSON.stringify(value)}\n`;
  const codexExec = join(stateRoot, "rollout-exec.jsonl");
  await writeFile(codexExec, [
    line({ type: "session_meta", payload: { id: "exec-thread", source: "exec", originator: "codex_exec" } }),
    line({ type: "response_item", payload: { type: "message", role: "user", content: [{ type: "input_text", text: "[system] 跑一下实验" }] } }),
    line({ type: "response_item", payload: { type: "message", role: "assistant", phase: "final_answer", content: [{ type: "output_text", text: "跑完了" }] } }),
  ].join(""));
  const claudeTranscriptPath = await claudeTranscript(stateRoot, "[system] 跑一下实验");
  const commandHooks = ["session-start.mjs", "user-prompt.mjs", "stop.mjs", "pre-compact.mjs", "session-end.mjs"];
  const cases = [
    ["habitus-memory", { session_id: "exec-thread", transcript_path: codexExec, prompt: "[system] 跑一下实验" }, {}],
    ["habitus-memory-claude-code", { session_id: "cc-print", transcript_path: claudeTranscriptPath, prompt: "[system] 跑一下实验" },
      { CLAUDE_CODE_ENTRYPOINT: "sdk-cli" }],
  ];
  for (const [plugin, input, env] of cases) {
    for (const script of commandHooks) {
      for (const worker of ["", "1"]) {
        const result = await runScript(plugin, script, input, {
          HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_HOOK_WORKER: worker, ...env,
        });
        assert.equal(result.code, 0, `${plugin}/${script}: ${result.stderr}`);
        if (!worker) assert.deepEqual(JSON.parse(result.stdout), {}, `${plugin}/${script}`);
      }
    }
  }
  await new Promise((resolve) => setTimeout(resolve, 500));
  assert.deepEqual(service.calls.map((call) => call.path), []);
  const left = (await readdir(stateRoot)).filter((name) => ["sessions", "outbox", "transcripts"].includes(name));
  assert.deepEqual(left, []);
});

// 杀变异 HR06（SessionEnd 的 worker 不退场旧会话）：转写已不在、outbox 空、过了宽限期的会话状态在 SessionEnd 时删掉
test("the SessionEnd worker retires idle sessions whose transcript is gone", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const idle = {
    nativeSessionId: "idle-1", conversationId: "claude-idle-1", startedOn: "2026-10-01", protocol: "claude_code",
    nextSequence: 0, acknowledgedTranscriptCursor: null, injectionReceipts: [], transcriptPath: join(stateRoot, "gone.jsonl"),
  };
  await writeState(stateRoot, "claude-code", "idle-1", idle);
  const when = new Date(Date.now() - 2 * 3_600_000);
  await utimes(join(stateRoot, "sessions", "claude-code", `${sessionKey("claude-code", "idle-1")}.json`), when, when);
  const result = await runScript("habitus-memory-claude-code", "session-end.mjs", { session_id: "ending-1" }, {
    HABITUS_URL: UNREACHABLE, HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_HOOK_WORKER: "1",
    HABITUS_PLUGIN_SESSION_GRACE_HOURS: "1",
  });
  assert.equal(result.code, 0, result.stderr);
  assert.equal(await readState(stateRoot, "claude-code", "idle-1"), null);
});

// 杀变异 HR05（HABITUS_MEMORY_ENABLED=0 不生效）：关掉之后 hook 什么都不碰，只回宿主一个空的成功
test("HABITUS_MEMORY_ENABLED=0 makes every hook a no-op", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = await claudeTranscript(stateRoot);
  for (const script of ["stop.mjs", "pre-compact.mjs"]) {
    const result = await runScript("habitus-memory-claude-code", script, { session_id: "off-1", transcript_path: transcript }, {
      HABITUS_URL: UNREACHABLE, HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_MEMORY_ENABLED: "0",
    });
    assert.equal(result.code, 0, result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), {});
  }
  assert.deepEqual((await readdir(stateRoot)).sort(), ["transcript.jsonl"]);
});

// 杀变异 HR08（去掉 1 MiB 输入上限）：超大的 hook 输入整个不读，fail open 并记一笔
test("hook input over 1 MiB is refused: nothing is captured and the failure is logged", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const transcript = await claudeTranscript(stateRoot);
  const input = JSON.stringify({ session_id: "big-1", transcript_path: transcript, padding: "x".repeat(1024 * 1024) });
  const result = await runScript("habitus-memory-claude-code", "pre-compact.mjs", input, {
    HABITUS_URL: UNREACHABLE, HABITUS_PLUGIN_STATE_DIR: stateRoot,
  });
  assert.equal(result.code, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), {});
  assert.deepEqual(await listOutbox(stateRoot, "claude-code", "big-1"), []);
  const events = (await readFile(join(stateRoot, "logs", "operations.jsonl"), "utf8")).trim().split("\n").map((value) => JSON.parse(value));
  assert.equal(events.at(-1).detail, "hook input exceeds 1 MiB");
});

// B17：配置读不出来（这里是 connection.json 写着 IPv4 映射地址）时还不知道往哪儿写操作日志，hook 照样放行，但要在 stderr 留一行
test("a hook whose plugin config cannot be read still succeeds and says why on stderr", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const connection = join(stateRoot, "connection.json");
  await writeFile(connection, `${JSON.stringify({ base_url: "http://[::ffff:127.0.0.1]:8787", schema_version: 1 })}\n`, { mode: 0o600 });
  const result = await runScript("habitus-memory-claude-code", "stop.mjs", { session_id: "cfg-1", transcript_path: join(stateRoot, "t.jsonl") }, {
    HABITUS_URL: "", HABITUS_PLUGIN_STATE_DIR: stateRoot,
  });
  assert.equal(result.code, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), {});
  assert.match(result.stderr, /^habitus-memory: hook skipped, plugin config unusable: HABITUS_URL must be an unauthenticated loopback HTTP URL\n$/);
  assert.deepEqual(await readdir(stateRoot), ["connection.json"]);
});

// T6：命令 hook 打不打堆栈与 config.debug 是同一个判断（HABITUS_PLUGIN_DEBUG 只解析一处）：开着的写法（1、2、true…）都打，
// 0 / off 不打；只有操作日志那一条照常记
test("a command hook prints the error stack exactly when config.debug is on", async (t) => {
  for (const value of ["1", "2", "true", "0", "off"]) {
    const stateRoot = await temporaryRoot(t);
    const env = { HABITUS_URL: UNREACHABLE, HABITUS_PLUGIN_STATE_DIR: stateRoot, HABITUS_PLUGIN_DEBUG: value };
    const result = await runScript("habitus-memory", "user-prompt.mjs", "not json", env);
    assert.equal(result.code, 0, result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), {});
    const debug = loadPluginConfig(env).debug;
    assert.equal(debug, !["0", "off"].includes(value), value);
    if (debug) assert.match(result.stderr, /\[habitus-memory\] SyntaxError: .*\n\s+at /, value);
    else assert.equal(result.stderr, "", value);
  }
});

// T5：DeepSeek Harness 没有 SessionEnd：每次 SessionStart 的后台顺带让会话退场。dsh 的会话日志没了、outbox 空、过了宽限期的
// 会话状态与插件镜像一起删；日志还在的、outbox 还有没送的都留着
test("a dsh SessionStart worker retires sessions whose session log is gone, together with their mirror", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const dshHome = join(stateRoot, "dsh");
  const line = (value) => `${JSON.stringify(value)}\n`;
  const log = async (id) => {
    const directory = join(dshHome, "sessions", "--w--", encodeSegment(id));
    await mkdir(directory, { recursive: true });
    await writeFile(join(directory, "session.jsonl"), [
      line({ id, createdAt: "2026-10-22T07:00:00Z", cwd: "/w" }),
      line({ type: "turn/start", seq: 1, time: 1792998000001, data: {} }),
      line({ type: "user/message", seq: 2, time: 1792998000002, data: { text: "你好" } }),
      line({ type: "turn/end", seq: 3, time: 1792998000003, data: {} }),
    ].join(""));
    return directory;
  };
  const env = { HABITUS_URL: UNREACHABLE, HABITUS_PLUGIN_STATE_DIR: stateRoot, DSH_HOME: dshHome, HABITUS_PLUGIN_SESSION_GRACE_HOURS: "1" };
  // 三个会话各 Stop 一次（服务没起，每个都进 outbox）：gone 之后送空了，unsent 的还在 outbox，alive 的日志还在
  const directories = {};
  for (const id of ["gone", "unsent", "alive"]) {
    directories[id] = await log(id);
    const result = await runScript("habitus-memory-dsh", "stop.mjs", { session_id: id, transcript_path: "" }, env);
    assert.equal(result.code, 0, result.stderr);
  }
  for (const id of ["gone", "alive"]) await rm(join(stateRoot, "outbox", "dsh", sessionKey("dsh", id)), { recursive: true, force: true });
  for (const id of ["gone", "unsent"]) await rm(directories[id], { recursive: true, force: true });
  const when = new Date(Date.now() - 2 * 3_600_000);
  for (const id of ["gone", "unsent", "alive"]) {
    await utimes(join(stateRoot, "sessions", "dsh", `${sessionKey("dsh", id)}.json`), when, when);
  }
  const mirror = (id) => (async () => { try { await readFile((await readState(stateRoot, "dsh", id))?.transcriptPath ?? ""); return true; } catch { return false; } })();
  const goneMirror = (await readState(stateRoot, "dsh", "gone")).transcriptPath;
  await log("new");
  const result = await runScript("habitus-memory-dsh", "session-start.mjs", { session_id: "new", transcript_path: "" }, { ...env, HABITUS_HOOK_WORKER: "1" });
  assert.equal(result.code, 0, result.stderr);
  assert.equal(await readState(stateRoot, "dsh", "gone"), null);
  await assert.rejects(readFile(goneMirror), /ENOENT/);
  assert.notEqual(await readState(stateRoot, "dsh", "unsent"), null);
  assert.notEqual(await readState(stateRoot, "dsh", "alive"), null);
  assert.equal(await mirror("alive"), true);
});
