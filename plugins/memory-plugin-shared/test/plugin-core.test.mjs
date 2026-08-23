import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { chmod, mkdir, mkdtemp, readFile, readdir, rm, stat, symlink, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { executePluginLifecycle, parseArgs, prepareMarketplace } from "../../install-memory-plugin.mjs";
import { HARNESS_REGISTRY, createHarnessRegistry } from "../../harnesses.mjs";
import { outboxInventory, runPluginDoctor } from "../doctor.mjs";
import { loadPluginConfig } from "../lib/config.mjs";
import { requireHostAdapter } from "../lib/host-adapter.mjs";
import { createContextInjection, encodeContextPayload } from "../lib/hook-runner.mjs";
import { listOutbox } from "../lib/outbox.mjs";
import { PluginCore } from "../lib/plugin-core.mjs";
import { HabitusServiceClient } from "../lib/service-client.mjs";
import { readState, withSessionLock } from "../lib/state-store.mjs";

const PLUGINS_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];

class FakeHostRunner {
  constructor() {
    this.marketplaceSource = null;
    this.installed = false;
    this.enabled = false;
    this.calls = [];
    this.failNextPluginAdd = false;
  }

  available(command) { return command === "codex"; }

  json(command, args) {
    assert.equal(command, "codex");
    if (args.includes("marketplace")) {
      return { marketplaces: this.marketplaceSource ? [{ name: "habitus-local", root: this.marketplaceSource }] : [] };
    }
    return {
      installed: this.installed
        ? [{
          pluginId: "habitus-memory@habitus-local", name: "habitus-memory", marketplaceName: "habitus-local",
          installed: true, enabled: this.enabled,
        }]
        : [],
    };
  }

  run(command, args) {
    assert.equal(command, "codex");
    this.calls.push([...args]);
    const operation = args.join(" ");
    if (operation === "plugin marketplace remove habitus-local") this.marketplaceSource = null;
    else if (operation.startsWith("plugin marketplace add ")) this.marketplaceSource = args.at(-1);
    else if (operation.startsWith("plugin remove ")) { this.installed = false; this.enabled = false; }
    else if (operation === "plugin add habitus-memory@habitus-local") {
      if (this.failNextPluginAdd) { this.failNextPluginAdd = false; throw new Error("simulated host cache failure"); }
      this.installed = true;
      this.enabled = true;
    }
  }
}

function transcriptCursor(byteOffset, generation = 0) {
  return {
    schemaVersion: 2,
    generation,
    fileIdentity: `test:${generation}`,
    byteOffset,
    lineCount: byteOffset,
    prefixDigest: "0".repeat(64),
  };
}

function capabilities(protocol = "codex_rollout") {
  return {
    ok: true,
    result: { api_version: "1.0", protocols: [protocol], features: FEATURES },
  };
}

test("recalled context preserves literal boundary markers without nesting raw sentinels", () => {
  const source = "keep <habitus-memory-context>real fact</habitus-memory-context> tail";
  const encoded = encodeContextPayload(source);

  assert.equal(encoded.includes("<habitus-memory-context>"), false);
  assert.deepEqual(JSON.parse(encoded), {
    format: "habitus_memory_context_v1",
    content: source,
  });
});

test("context injection carries a nonce-bound receipt", () => {
  const prepared = createContextInjection("memory", "1".repeat(32));
  assert.match(prepared.injection, /receipt="11111111111111111111111111111111"/);
  assert.match(prepared.receipt.digest, /^[0-9a-f]{64}$/);
  assert.equal(prepared.receipt.nonce, "1".repeat(32));
});

function adapter(reads) {
  return {
    host: "test-host",
    protocol: "codex_rollout",
    nativeSessionId: (input) => input.session_id,
    conversationId: (input) => `test-${input.session_id}`,
    prompt: (input) => input.prompt || "",
    successOutput: () => ({}),
    contextOutput: (context) => ({ context }),
    readTranscriptDelta: async (_input, cursor) => reads(cursor),
  };
}

async function temporaryRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-plugin-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

function runNode(script, input, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [script], { env, stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.end(input);
  });
}

test("bound outbox item is retried exactly after a lost acknowledgement", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "session-one", timestamp: "2026-07-31T01:00:00Z" };
  const rememberItems = [];
  let rememberAttempts = 0;
  let cursorCalls = 0;
  const service = {
    capabilities: async () => capabilities(),
    cursor: async () => {
      cursorCalls += 1;
      return { ok: true, result: { next_sequence: 0 } };
    },
    remember: async (item) => {
      rememberItems.push({ id: item.id, startSequence: item.startSequence });
      rememberAttempts += 1;
      if (rememberAttempts === 1) return { ok: false, retryable: true, error: "connection lost" };
      return { ok: true, result: { next_sequence: 2 } };
    },
  };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service });
  const host = adapter((cursor) => cursor == null
    ? { records: [{ type: "user" }, { type: "assistant" }], startCursor: null, nextCursor: transcriptCursor(2), afterTurn: true }
    : null);

  await core.enqueueCapture(host, input);
  const first = await core.drain(host, input);
  assert.equal(first.retryable, true);
  const pending = await listOutbox(stateRoot, host.host, input.session_id);
  assert.equal(pending.length, 1);
  assert.equal(pending[0].status, "bound");
  assert.equal(pending[0].startSequence, 0);

  await core.drain(host, input);
  assert.deepEqual(rememberItems[0], rememberItems[1]);
  assert.equal(cursorCalls, 1);
  assert.deepEqual(await listOutbox(stateRoot, host.host, input.session_id), []);
  const state = await readState(stateRoot, host.host, input.session_id);
  assert.equal(state.nextSequence, 2);
  assert.equal(state.acknowledgedTranscriptCursor.byteOffset, 2);
});

test("new capture begins after the largest queued transcript offset", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const offsets = [];
  const host = adapter((cursor) => {
    const offset = cursor?.byteOffset || 0;
    offsets.push(offset);
    return { records: [{ offset }], startCursor: cursor, nextCursor: transcriptCursor(offset + 2), afterTurn: true };
  });
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: {} });
  const input = { session_id: "session-two" };

  await core.enqueueCapture(host, input);
  await core.enqueueCapture(host, input);

  assert.deepEqual(offsets, [0, 2]);
  assert.equal((await listOutbox(stateRoot, host.host, input.session_id)).length, 2);
});

test("non-retryable service rejection blocks but never drops the outbox item", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "session-three" };
  const service = {
    capabilities: async () => capabilities(),
    cursor: async () => ({ ok: true, result: { next_sequence: 4 } }),
    remember: async () => ({ ok: false, retryable: false, error: { code: "INVALID_ARGUMENT" } }),
  };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service });
  const host = adapter(() => ({ records: [{ type: "assistant" }], startCursor: null, nextCursor: transcriptCursor(1), afterTurn: true }));

  await core.enqueueCapture(host, input);
  await core.drain(host, input);

  const pending = await listOutbox(stateRoot, host.host, input.session_id);
  assert.equal(pending.length, 1);
  assert.equal(pending[0].status, "blocked");
});

test("service client sends no authorization header", async () => {
  let request;
  const client = new HabitusServiceClient(
    { baseUrl: "http://127.0.0.1:8787", timeoutMs: 1000 },
    async (url, init) => {
      request = { url, init };
      return { ok: true, status: 200, json: async () => ({ status: "ok", result: {} }) };
    },
  );

  await client.health();
  assert.equal(request.url, "http://127.0.0.1:8787/health");
  assert.equal("Authorization" in request.init.headers, false);
  assert.equal("authorization" in request.init.headers, false);
});

test("configuration accepts only unauthenticated loopback service URLs", () => {
  assert.equal(loadPluginConfig({ HABITUS_URL: "http://localhost:8787" }).baseUrl, "http://localhost:8787");
  assert.equal(loadPluginConfig({ HABITUS_URL: "http://[::1]:8787" }).baseUrl, "http://[::1]:8787");
  assert.throws(() => loadPluginConfig({ HABITUS_URL: "http://192.168.1.5:8787" }), /loopback/);
  assert.throws(
    () => loadPluginConfig({ HABITUS_URL: "http://user:secret@127.0.0.1:8787" }),
    /unauthenticated/,
  );
  assert.throws(() => loadPluginConfig({ HABITUS_URL: "http://127.0.0.1:8787/api" }), /must not contain a path/);
  assert.throws(() => loadPluginConfig({ HABITUS_URL: "http://127.0.0.1:8787?x=1" }), /loopback/);
});

test("configuration reads the service URL projected beside a selected Habitus YAML", async (t) => {
  const root = await temporaryRoot(t);
  const configPath = join(root, "config.yaml");
  const stateRoot = join(root, "agent-plugin");
  await writeFile(configPath, "storage: {}\n", { mode: 0o600 });
  await mkdir(stateRoot, { recursive: true });
  await writeFile(
    join(stateRoot, "connection.json"),
    `${JSON.stringify({ schema_version: 1, base_url: "http://127.0.0.1:8899" })}\n`,
    { mode: 0o600 },
  );

  const config = loadPluginConfig({ HABITUS_CONFIG_FILE: configPath });

  assert.equal(config.baseUrl, "http://127.0.0.1:8899");
  assert.equal(config.stateRoot, stateRoot);
});

test("configuration rejects a group or world writable service connection file", async (t) => {
  const root = await temporaryRoot(t);
  const connection = join(root, "connection.json");
  await writeFile(
    connection,
    `${JSON.stringify({ schema_version: 1, base_url: "http://127.0.0.1:9999" })}\n`,
    { mode: 0o600 },
  );
  await chmod(connection, 0o666);

  assert.throws(
    () => loadPluginConfig({ HABITUS_PLUGIN_STATE_DIR: root }),
    /permissions|writable|private/,
  );
});

test("marketplace preparation is isolated and produces private manifests", async (t) => {
  const root = await temporaryRoot(t);
  await prepareMarketplace(root);

  const codex = JSON.parse(await readFile(join(root, ".agents", "plugins", "marketplace.json"), "utf8"));
  const claude = JSON.parse(await readFile(join(root, ".claude-plugin", "marketplace.json"), "utf8"));
  assert.equal(codex.plugins[0].source.path, "./plugins/habitus-memory");
  assert.equal(claude.plugins[0].source, "./plugins/habitus-memory-claude-code");
  const manifestMode = (await stat(join(root, ".agents", "plugins", "marketplace.json"))).mode & 0o777;
  assert.equal(manifestMode, 0o600);
  const hooks = JSON.parse(await readFile(join(root, "plugins", "habitus-memory-claude-code", "hooks", "hooks.json"), "utf8"));
  assert.ok(hooks.hooks.SubagentStart);
  assert.ok(hooks.hooks.SubagentStop);
});

test("plugin lifecycle is idempotent, migrates sources, rolls back, and removes cleanly", async (t) => {
  const root = await temporaryRoot(t);
  const runner = new FakeHostRunner();
  const options = { action: "install", host: "codex", root, prepareOnly: false, json: true };

  await executePluginLifecycle(options, { runner });
  assert.equal(runner.marketplaceSource, root);
  assert.equal(runner.installed, true);
  const firstCallCount = runner.calls.length;
  await executePluginLifecycle(options, { runner });
  assert.equal(runner.calls.length, firstCallCount);

  runner.marketplaceSource = join(root, "old-source");
  await executePluginLifecycle(options, { runner });
  assert.equal(runner.marketplaceSource, root);
  assert.equal(runner.installed, true);

  runner.failNextPluginAdd = true;
  await assert.rejects(
    executePluginLifecycle({ ...options, action: "update" }, { runner }),
    /simulated host cache failure/,
  );
  assert.equal(runner.marketplaceSource, root);
  assert.equal(runner.installed, true);
  assert.equal((await executePluginLifecycle({ ...options, action: "status" }, { runner })).marketplace.prepared, true);

  const removed = await executePluginLifecycle({ ...options, action: "remove" }, { runner });
  assert.equal(removed.removed, true);
  assert.equal(runner.marketplaceSource, null);
  assert.equal(runner.installed, false);
});

test("plugin lifecycle argument parser exposes all supported operations", () => {
  assert.equal(parseArgs(["status", "--host", "codex", "--json"]).action, "status");
  assert.deepEqual(parseArgs(["status", "--harness", "cc"]).harnesses, ["cc"]);
  assert.equal(parseArgs(["update", "--prepare-only"]).prepareOnly, true);
  assert.equal(parseArgs(["remove"]).action, "remove");
  assert.equal(parseArgs(["harnesses", "--json"]).action, "harnesses");
  assert.throws(() => parseArgs(["status", "--prepare-only"]), /prepare-only/);
});

test("Harness registry drives a future adapter through lifecycle orchestration", async (t) => {
  const codex = HARNESS_REGISTRY.resolve("codex");
  const future = {
    ...codex,
    id: "future-harness",
    aliases: ["future-harness", "fh"],
    displayName: "Future Harness",
    command: "future-harness",
    protocol: "future_protocol",
  };
  const registry = createHarnessRegistry([future]);
  const root = await temporaryRoot(t);
  const runner = {
    available: (command) => command === "future-harness",
    json: () => [],
    run: () => assert.fail("prepare-only must not register the plugin"),
  };

  assert.equal(registry.resolve("fh").id, "future-harness");
  const result = await executePluginLifecycle(
    {
      action: "install",
      harnesses: ["fh"],
      root,
      prepareOnly: true,
      json: true,
    },
    { registry, runner },
  );
  assert.equal(result.preparedOnly, true);
  assert.equal(result.harnesses[0].harness, "future-harness");
  assert.equal(result.harnesses[0].available, true);
});

test("realpath-aware entrypoints execute through filesystem symlinks", async (t) => {
  const directory = await temporaryRoot(t);
  const installer = join(directory, "install-link.mjs");
  const doctor = join(directory, "doctor-link.mjs");
  await symlink(join(PLUGINS_ROOT, "install-memory-plugin.mjs"), installer);
  await symlink(join(PLUGINS_ROOT, "memory-plugin-shared", "doctor.mjs"), doctor);
  const marketplace = join(directory, "marketplace");
  const installed = spawnSync(process.execPath, [installer, "install", "--prepare-only", "--root", marketplace, "--json"], {
    encoding: "utf8",
  });
  assert.equal(installed.status, 0, installed.stderr);
  assert.equal(JSON.parse(installed.stdout).preparedOnly, true);

  const diagnosed = spawnSync(process.execPath, [doctor], {
    encoding: "utf8",
    env: {
      ...process.env,
      HABITUS_URL: "http://127.0.0.1:1",
      HABITUS_PLUGIN_STATE_DIR: join(directory, "state"),
      HABITUS_PLUGIN_TIMEOUT_MS: "250",
    },
  });
  assert.equal(diagnosed.status, 1);
  assert.equal(Array.isArray(JSON.parse(diagnosed.stdout).checks), true);
});

test("real Codex hook process negotiates, recalls, and persists its injection receipt", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const server = createServer((request, response) => {
    const result = request.url === "/api/v1/capabilities"
      ? { api_version: "1.0", service_version: "test", protocols: ["codex_rollout"], features: FEATURES }
      : request.url === "/ready"
        ? { ready: true, status: "healthy" }
        : request.url === "/api/v1/memory/recall"
          ? { context: "remembered context", query: "prompt", queries: ["prompt"], memories: [], summaries: [], degradations: [], budget_exhausted: false }
          : {};
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ status: "ok", result }));
  });
  try {
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => {
        server.removeListener("error", reject);
        resolve();
      });
    });
  } catch (error) {
    if (error?.code === "EPERM") {
      t.skip("sandbox does not permit a loopback listener");
      return;
    }
    throw error;
  }
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const script = join(PLUGINS_ROOT, "habitus-memory", "scripts", "user-prompt.mjs");
  const result = await runNode(
    script,
    JSON.stringify({ session_id: "hook-smoke", prompt: "prompt" }),
    {
      ...process.env,
      HABITUS_URL: `http://127.0.0.1:${address.port}`,
      HABITUS_PLUGIN_STATE_DIR: stateRoot,
      HABITUS_PLUGIN_TIMEOUT_MS: "1000",
    },
  );
  assert.equal(result.code, 0, result.stderr);
  const output = JSON.parse(result.stdout);
  assert.match(output.hookSpecificOutput.additionalContext, /remembered context/);
  assert.match(output.hookSpecificOutput.additionalContext, /receipt="[0-9a-f]{32}"/);
  const stateFiles = await readdir(join(stateRoot, "sessions", "codex"));
  const state = JSON.parse(await readFile(join(stateRoot, "sessions", "codex", stateFiles[0]), "utf8"));
  assert.equal(state.injectionReceipts.length, 1);
});

test("session start rejects a service missing the host protocol or replay contract", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const service = {
    capabilities: async () => ({
      ok: true,
      result: { api_version: "1.0", protocols: ["claude_code"], features: ["recall", "remember"] },
    }),
  };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service });

  await assert.rejects(
    core.sessionStart(adapter(() => null), { session_id: "incompatible" }),
    /does not support protocol/,
  );
});

test("recall performs capability negotiation before readiness and never calls recall when incompatible", async (t) => {
  const stateRoot = await temporaryRoot(t);
  let readyCalls = 0;
  let recallCalls = 0;
  const service = {
    capabilities: async () => ({
      ok: true,
      result: { api_version: "2.0", protocols: ["codex_rollout"], features: FEATURES },
    }),
    ready: async () => { readyCalls += 1; return { ok: true, result: { ready: true } }; },
    recall: async () => { recallCalls += 1; return { ok: true, result: { context: "unsafe" } }; },
  };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service });

  assert.equal(await core.recall(adapter(() => null), { session_id: "incompatible-recall", prompt: "hello" }), "");
  assert.equal(readyCalls, 0);
  assert.equal(recallCalls, 0);
});

test("host adapter identifiers cannot escape the plugin state root", () => {
  const unsafe = { ...adapter(() => null), host: "../outside" };
  assert.throws(() => requireHostAdapter(unsafe), /safe lowercase identifier/);
});

test("successful pre-compaction flush preserves the generation-aware transcript cursor", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "compacted" };
  const host = adapter((cursor) => cursor == null
    ? { records: [{ type: "assistant" }], startCursor: null, nextCursor: transcriptCursor(5), afterTurn: true }
    : null);
  const service = {
    capabilities: async () => capabilities(),
    cursor: async () => ({ ok: true, result: { next_sequence: 10 } }),
    remember: async () => ({ ok: true, result: { next_sequence: 11 } }),
    flush: async () => ({ ok: true, result: {} }),
  };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service });

  await core.enqueueCapture(host, input);
  const flushed = await core.flush(host, input);
  assert.equal(flushed.ok, true);
  const state = await readState(stateRoot, host.host, input.session_id);

  assert.equal(state.acknowledgedTranscriptCursor.byteOffset, 5);
  assert.equal(state.nextSequence, 11);
});

test("service remember request carries the durable delivery identity", async () => {
  let body;
  const client = new HabitusServiceClient(
    { baseUrl: "http://127.0.0.1:8787", timeoutMs: 1000 },
    async (_url, init) => {
      body = JSON.parse(init.body);
      return { ok: true, status: 200, json: async () => ({ status: "ok", result: { next_sequence: 2 } }) };
    },
  );
  await client.remember({
    deliveryId: "a".repeat(64), conversationId: "c", startedOn: "2026-08-01", protocol: "codex_rollout",
    payload: { records: [] }, startSequence: 0, occurredAt: "2026-08-01T00:00:00Z", afterTurn: true,
  });
  assert.equal(body.delivery_id, "a".repeat(64));
});

test("session start recovers another pending session for the same host", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const host = adapter((cursor) => cursor == null
    ? { records: [{ type: "assistant" }], startCursor: null, nextCursor: transcriptCursor(1), afterTurn: true }
    : null);
  const delivered = [];
  const service = {
    capabilities: async () => capabilities(),
    cursor: async () => ({ ok: true, result: { next_sequence: 0 } }),
    remember: async (item) => { delivered.push(item.nativeSessionId); return { ok: true, result: { next_sequence: 1 } }; },
  };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service });
  await core.enqueueCapture(host, { session_id: "orphan" });
  await core.sessionStart(host, { session_id: "current" });
  await core.recoverPending(host, { session_id: "current" });
  assert.deepEqual(delivered, ["orphan"]);
});

test("remember network I/O does not hold the session lock", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const input = { session_id: "network-lock" };
  const host = adapter((cursor) => cursor == null
    ? { records: [{ type: "assistant" }], startCursor: null, nextCursor: transcriptCursor(1), afterTurn: true }
    : null);
  let notifyRemember;
  const rememberStarted = new Promise((resolve) => { notifyRemember = resolve; });
  let releaseRemember;
  const rememberGate = new Promise((resolve) => { releaseRemember = resolve; });
  const service = {
    capabilities: async () => capabilities(),
    cursor: async () => ({ ok: true, result: { next_sequence: 0 } }),
    remember: async () => {
      notifyRemember();
      await rememberGate;
      return { ok: true, result: { next_sequence: 1 } };
    },
  };
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service });
  await core.enqueueCapture(host, input);
  const draining = core.drain(host, input);
  await rememberStarted;
  let entered = false;
  await withSessionLock(stateRoot, host.host, input.session_id, async () => { entered = true; });
  assert.equal(entered, true);
  releaseRemember();
  await draining;
});

test("plugin doctor reports every durable outbox lifecycle state", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const host = adapter((cursor) => cursor == null
    ? { records: [{ type: "assistant" }], startCursor: null, nextCursor: transcriptCursor(1), afterTurn: true }
    : null);
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: {} });
  await core.enqueueCapture(host, { session_id: "doctor-outbox" });
  const service = {
    health: async () => ({ ok: true, result: { status: "healthy" } }),
    ready: async () => ({ ok: true, result: { ready: true, status: "healthy" } }),
    capabilities: async () => ({
      ok: true,
      result: {
        api_version: "1.0",
        service_version: "test",
        protocols: ["claude_code", "codex_rollout"],
        features: FEATURES,
      },
    }),
  };
  const report = await runPluginDoctor(
    { stateRoot },
    { service, inspectHarnesses: async () => [] },
  );
  const outbox = report.checks.find((check) => check.name === "outbox");
  assert.equal(report.ok, true);
  assert.match(outbox.detail, /queued_items=1/);
  assert.match(outbox.detail, /inflight_items=0/);
});

test("plugin doctor fails when an outbox session lost its durable state", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const host = adapter((cursor) => cursor == null
    ? { records: [{ type: "assistant" }], startCursor: null, nextCursor: transcriptCursor(1), afterTurn: true }
    : null);
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: {} });
  await core.enqueueCapture(host, { session_id: "missing-state" });
  const stateFiles = await readdir(join(stateRoot, "sessions", host.host));
  await rm(join(stateRoot, "sessions", host.host, stateFiles.find((name) => name.endsWith(".json"))));
  const service = {
    health: async () => ({ ok: true, result: { status: "healthy" } }),
    ready: async () => ({ ok: true, result: { ready: true, status: "healthy" } }),
    capabilities: async () => ({
      ok: true,
      result: {
        api_version: "1.0", service_version: "test",
        protocols: ["claude_code", "codex_rollout"], features: FEATURES,
      },
    }),
  };

  const report = await runPluginDoctor(
    { stateRoot },
    { service, inspectHarnesses: async () => [] },
  );
  const outbox = report.checks.find((check) => check.name === "outbox");
  assert.equal(report.ok, false);
  assert.equal(outbox.status, "fail");
  assert.match(outbox.detail, /missingState_items=1/);
});

test("plugin doctor inventory detects aged and expired inflight delivery", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const host = adapter((cursor) => cursor == null
    ? { records: [{ type: "assistant" }], startCursor: null, nextCursor: transcriptCursor(1), afterTurn: true }
    : null);
  const core = new PluginCore({ stateRoot, timeoutMs: 1000 }, { service: {} });
  await core.enqueueCapture(host, { session_id: "aged-inflight" });
  const sessionDirectories = await readdir(join(stateRoot, "outbox", host.host));
  const directory = join(stateRoot, "outbox", host.host, sessionDirectories[0]);
  const filenames = await readdir(directory);
  const path = join(directory, filenames[0]);
  const item = JSON.parse(await readFile(path, "utf8"));
  await writeFile(path, JSON.stringify({
    ...item,
    status: "inflight",
    claimToken: "claim",
    claimExpiresAt: "2026-07-01T00:00:00Z",
    createdAt: "2026-07-01T00:00:00Z",
  }));

  const inventory = await outboxInventory(stateRoot, {
    now: Date.parse("2026-08-01T00:00:00Z"),
    maxPendingAgeMs: 1000,
  });
  assert.equal(inventory.aged, 1);
  assert.equal(inventory.expiredInflight, 1);
  assert.equal(inventory.invalidState, 0);
});

test("hook boundary fails open and records a private structured failure", async (t) => {
  const stateRoot = await temporaryRoot(t);
  const script = join(PLUGINS_ROOT, "habitus-memory", "scripts", "stop.mjs");
  const result = spawnSync(process.execPath, [script], {
    input: "{not-json",
    encoding: "utf8",
    env: { ...process.env, HABITUS_PLUGIN_DEBUG: "0", HABITUS_PLUGIN_STATE_DIR: stateRoot },
  });

  assert.equal(result.status, 0);
  assert.deepEqual(JSON.parse(result.stdout), {});
  assert.equal(result.stderr, "");
  const logPath = join(stateRoot, "logs", "operations.jsonl");
  const lines = (await readFile(logPath, "utf8")).trim().split("\n");
  const event = JSON.parse(lines.at(-1));
  assert.equal(event.hook, "stop");
  assert.equal(event.stage, "hook");
  assert.equal(event.retryable, true);
  assert.equal((await stat(logPath)).mode & 0o777, 0o600);
});
