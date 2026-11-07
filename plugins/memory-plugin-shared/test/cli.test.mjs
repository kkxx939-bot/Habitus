import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { hookReviewState, main } from "../../bin/habitus.mjs";
import { HARNESS_REGISTRY, createHarnessRegistry } from "../../harnesses.mjs";
import { executePluginLifecycle, parseArgs } from "../../install-memory-plugin.mjs";
import { HOOK_REVIEW_KILL_GRACE_MS, fileEntryPath, inspectHookReview } from "../../registrations.mjs";
import {
  harnessFeatureChecks, harnessFeatureGaps, harnessHookReviewChecks, harnessRegistrationChecks, marketplaceCheck, parseFeatureList, runPluginDoctor,
} from "../doctor.mjs";
import { loadPluginConfig } from "../lib/config.mjs";
import { HabitusServiceClient } from "../lib/service-client.mjs";
import { isolatedEnv } from "./support/isolated-env.mjs";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";

function sink() {
  const chunks = [];
  return { write: (value) => { chunks.push(String(value)); }, text: () => chunks.join("") };
}

async function serve(t, routes, statuses = {}) {
  const calls = [];
  const server = createServer((request, response) => {
    let body = "";
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      calls.push({ url: request.url, body: body ? JSON.parse(body) : null });
      const result = routes[request.url];
      response.writeHead(result ? statuses[request.url] ?? 200 : 404, { "Content-Type": "application/json" });
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

async function stateRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "habitus-cli-test-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test("help, version and unknown commands", async () => {
  const out = sink(); const err = sink();
  assert.equal(await main(["--help"], { stdout: out, stderr: err }), 0);
  assert.match(out.text(), /habitus plugin install/);
  const version = sink();
  assert.equal(await main(["--version"], { stdout: version, stderr: err }), 0);
  assert.match(version.text(), /^\d+\.\d+\.\d+\n$/);
  const unknown = sink();
  assert.equal(await main(["nope"], { stdout: out, stderr: unknown }), 2);
  assert.match(unknown.text(), /unknown command: nope/);
  const badPlugin = sink();
  assert.equal(await main(["plugin", "explode"], { stdout: out, stderr: badPlugin }), 1);
  assert.match(badPlugin.text(), /usage: habitus plugin/);
});

test("status reads health, readiness and capabilities from the local service", async (t) => {
  const service = await serve(t, {
    "/health": { status: "healthy", ready: true, recall_ready: true, checked_at: "2026-10-21T00:00:00Z", checks: [] },
    "/ready": { status: "degraded", ready: true, recall_ready: true, checked_at: "2026-10-21T00:00:00Z", checks: [{ name: "foresight", status: "degraded", detail: "no table yet", critical: false }] },
    "/api/v1/capabilities": { api_version: "1.0", service_version: "0.1.0", protocols: ["codex_rollout"], features: ["recall", "reminders_v1"] },
  });
  if (!service) return;
  const env = { HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "1000" };
  const json = sink();
  assert.equal(await main(["status", "--json"], { stdout: json, stderr: sink(), env }), 0);
  const report = JSON.parse(json.text());
  assert.equal(report.reachable, true);
  assert.equal(report.ready, true);
  assert.deepEqual(report.features, ["recall", "reminders_v1"]);
  const human = sink();
  await main(["status"], { stdout: human, stderr: sink(), env });
  assert.match(human.text(), /就绪/);
  assert.match(human.text(), /foresight: degraded/);
});

test("status says how to start the service when it is not reachable", async (t) => {
  const env = { HABITUS_URL: "http://127.0.0.1:1", HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "250" };
  const out = sink();
  assert.equal(await main(["status"], { stdout: out, stderr: sink(), env }), 1);
  assert.match(out.text(), /habitus-server/);
});

test("recall posts the query and prints the hits", async (t) => {
  const service = await serve(t, {
    "/api/v1/memory/recall": {
      query: "回答风格", queries: ["回答风格"], context: "用户偏好简洁回答。",
      memories: [{ uri: "memory://preferences/response_style.md", score: 0.91, matched_queries: ["回答风格"] }],
      summaries: [], degradations: [], budget_exhausted: false,
    },
  });
  if (!service) return;
  const env = { HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "1000" };
  const out = sink();
  assert.equal(await main(["recall", "回答", "风格", "--limit", "3"], { stdout: out, stderr: sink(), env }), 0);
  assert.deepEqual(service.calls.at(-1).body, { query: "回答 风格", limit: 3 });
  assert.match(out.text(), /0\.91  memory:\/\/preferences\/response_style\.md/);
  assert.match(out.text(), /用户偏好简洁回答/);
  const err = sink();
  assert.equal(await main(["recall"], { stdout: sink(), stderr: err, env }), 1);
  assert.match(err.text(), /usage: habitus recall/);
  assert.equal(await main(["recall", "x", "--limit", "0"], { stdout: sink(), stderr: sink(), env }), 1);
});

test("harnesses lists the registered hosts", async (t) => {
  const isolated = await isolatedEnv(t);
  const out = sink();
  assert.equal(await main(["harnesses", "--json"], { stdout: out, stderr: sink(), env: isolated.env }), 0);
  const ids = JSON.parse(out.text()).harnesses.map((item) => item.id);
  assert.deepEqual(ids, HARNESS_REGISTRY.list().map((item) => item.id));
  // 每个宿主的 --version 都落到假宿主上（经 env 的 PATH），HOME 也是临时的那个
  const calls = await isolated.calls();
  assert.deepEqual(calls.map((call) => call.command).sort(), HARNESS_REGISTRY.list().map((item) => item.command).sort());
  assert.ok(calls.every((call) => call.home === isolated.env.HOME), JSON.stringify(calls));
});

test("the doctor fails a harness whose hook feature is off and skips hosts without a CLI", () => {
  const listing = [
    "apps                                 stable             true",
    "hooks                                stable             false",
    "plugin_hooks                         removed            false",
    "network_proxy                        experimental       false",
  ].join("\n");
  assert.equal(parseFeatureList(listing).get("hooks"), false);
  assert.equal(parseFeatureList(listing).get("apps"), true);
  const registry = createHarnessRegistry(HARNESS_REGISTRY.list().map((item) => ({ ...item })));
  const off = harnessFeatureChecks(registry, [], () => ({ status: 0, stdout: listing }));
  return off.then((checks) => {
    assert.deepEqual(checks.map((item) => [item.name, item.status]), [["features_codex", "fail"]]);
    assert.match(checks[0].detail, /features off: hooks/);
    return harnessFeatureChecks(registry, [], () => ({ status: 0, stdout: listing.replace("stable             false", "stable             true") }));
  }).then((checks) => {
    assert.equal(checks[0].status, "pass");
    return harnessFeatureChecks(registry, [], () => ({ error: Object.assign(new Error("missing"), { code: "ENOENT" }) }));
  }).then((checks) => {
    assert.deepEqual(checks, []);
  });
});

// B10：Codex CLI 的默认模式没有提问工具时，提醒在那里问不出去——Codex 已注册才报，只报 warn，不让体检失败
test("the doctor warns, without failing, when a registered Codex has no request_user_input in its default mode", async () => {
  const listing = (value) => [
    "default_mode_request_user_input      under development  " + value,
    "hooks                                stable             true",
  ].join("\n");
  const run = (stdout) => () => ({ status: 0, stdout });
  const registered = [{ name: "harness_codex", status: "pass" }];
  for (const stdout of [listing("false"), "hooks                                stable             true"]) {
    const [check] = await harnessFeatureChecks(HARNESS_REGISTRY, registered, run(stdout));
    assert.deepEqual([check.name, check.status], ["features_codex", "warn"]);
    assert.match(check.detail, /Codex CLI 的默认模式没有 request_user_input，提醒在那里问不出去；Codex Desktop 不受影响/);
  }
  // 开着、或 Codex 没注册：不提
  assert.equal((await harnessFeatureChecks(HARNESS_REGISTRY, registered, run(listing("true"))))[0].status, "pass");
  assert.equal((await harnessFeatureChecks(HARNESS_REGISTRY, [], run(listing("false"))))[0].status, "pass");
  // 杀变异 DR06：查过、但没注册上（warn / fail）的 Codex 也不提
  for (const status of ["warn", "fail"]) {
    assert.equal((await harnessFeatureChecks(HARNESS_REGISTRY, [{ name: "harness_codex", status }], run(listing("false"))))[0].status, "pass");
  }
  // hook 特性本身没开仍是 fail
  const hooksOff = listing("false").replace("stable             true", "stable             false");
  assert.equal((await harnessFeatureChecks(HARNESS_REGISTRY, registered, run(hooksOff)))[0].status, "fail");
  // habitus plugin status 那一行带同一句
  const lifecycle = async (options) => ({ action: options.action, root: "/tmp/m", harnesses: [{ harness: "codex", installed: true, enabled: true }] });
  const featureGaps = (definition) => harnessFeatureGaps(definition, run(listing("false")));
  const out = sink();
  assert.equal(await main(["plugin", "status", "--harness", "codex"], {
    stdout: out, stderr: sink(), lifecycle, reviewHooks: async () => null, featureGaps,
  }), 0);
  assert.match(out.text(), /codex +已注册；Codex CLI 的默认模式没有 request_user_input，提醒在那里问不出去；Codex Desktop 不受影响\n/);
});

test("the repository root carries the same marketplace manifests the installer stages", async () => {
  // 用户可以不经 npm、直接用宿主原生命令装：`codex plugin marketplace add <仓库>` / `claude plugin marketplace add <仓库>`。
  // 根目录那两份必须与安装器暂存的一模一样（相对路径 ./plugins/<目录> 两处都成立）
  const { readFile } = await import("node:fs/promises");
  const repository = join(fileURLToPath(new URL(".", import.meta.url)), "..", "..", "..");
  for (const definition of HARNESS_REGISTRY.list().filter((item) => item.marketplaceManifest)) {
    const written = JSON.parse(await readFile(join(repository, definition.marketplaceManifest.path), "utf8"));
    assert.deepEqual(written, definition.marketplaceManifest.document, definition.id);
  }
});

// ── review 确认的缺陷（2026-10-21）────────────────────────────────────────────────────────────────

// 只有 codex 有 CLI 的假宿主；OpenCode 走文件注册，不需要 CLI
class OnlyCodexRunner {
  constructor() { this.marketplaceSource = null; this.installed = false; }
  available(command) { return command === "codex"; }
  json(_command, args) {
    if (args.includes("marketplace")) return { marketplaces: this.marketplaceSource ? [{ name: "habitus-local", root: this.marketplaceSource }] : [] };
    return { installed: this.installed ? [{ pluginId: "habitus-memory@habitus-local", installed: true, enabled: true }] : [] };
  }
  run(_command, args) {
    const operation = args.join(" ");
    if (operation.startsWith("plugin marketplace add ")) this.marketplaceSource = args.at(-1);
    else if (operation === "plugin marketplace remove habitus-local") this.marketplaceSource = null;
    else if (operation.startsWith("plugin add ")) this.installed = true;
    else if (operation.startsWith("plugin remove ")) this.installed = false;
  }
}

async function isolated(t) {
  const root = await stateRoot(t);
  const env = { HOME: join(root, "home"), XDG_CONFIG_HOME: join(root, "config") };
  await mkdir(join(env.XDG_CONFIG_HOME, "opencode"), { recursive: true });
  return { env, marketplace: join(root, "marketplace"), runner: new OnlyCodexRunner() };
}

test("install and remove report the state after the action", async (t) => {
  const { env, marketplace, runner } = await isolated(t);
  const options = { action: "install", harnesses: ["codex", "opencode"], root: marketplace };
  const installed = await executePluginLifecycle(options, { runner, env });
  assert.deepEqual(installed.harnesses.map((item) => [item.harness, item.installed]), [["codex", true], ["opencode", true]]);
  const removed = await executePluginLifecycle({ ...options, action: "remove" }, { runner, env });
  assert.deepEqual(removed.harnesses.map((item) => [item.harness, item.installed]), [["codex", false], ["opencode", false]]);
  assert.equal(removed.removed, true);
});

test("removing one harness keeps the marketplace root the others still use", async (t) => {
  const { env, marketplace, runner } = await isolated(t);
  await executePluginLifecycle({ action: "install", harnesses: ["codex", "opencode"], root: marketplace }, { runner, env });
  const opencodeGone = await executePluginLifecycle({ action: "remove", harnesses: ["opencode"], root: marketplace }, { runner, env });
  assert.equal(opencodeGone.removed, false);
  assert.ok((await stat(join(marketplace, "plugins", "habitus-memory"))).isDirectory());
  assert.equal(runner.marketplaceSource, marketplace);

  await executePluginLifecycle({ action: "install", harnesses: ["opencode"], root: marketplace }, { runner, env });
  const codexGone = await executePluginLifecycle({ action: "remove", harnesses: ["codex"], root: marketplace }, { runner, env });
  assert.equal(codexGone.removed, false);
  const entry = await readFile(fileEntryPath(HARNESS_REGISTRY.resolve("opencode"), env), "utf8");
  const target = new URL(entry.match(/from "([^"]+)"/)[1]);
  assert.ok((await stat(target)).isFile(), "the OpenCode entry must still resolve");
});

test("OpenCode registers without its CLI on PATH (desktop installs) and removal never leaves a dangling entry", async (t) => {
  const { env, marketplace } = await isolated(t);
  const noCli = { available: () => false, json: () => { throw new Error("no CLI"); }, run: () => { throw new Error("no CLI"); } };
  const options = { action: "install", harnesses: ["opencode"], root: marketplace };
  assert.equal((await executePluginLifecycle(options, { runner: noCli, env })).harnesses[0].installed, true);
  const removed = await executePluginLifecycle({ ...options, action: "remove" }, { runner: noCli, env });
  assert.equal(removed.removed, true);
  await assert.rejects(readFile(fileEntryPath(HARNESS_REGISTRY.resolve("opencode"), env), "utf8"), /ENOENT/);
});

test("status and the doctor see an entry whose plugin is gone as dangling, not registered", async (t) => {
  const { env, marketplace, runner } = await isolated(t);
  await executePluginLifecycle({ action: "install", harnesses: ["opencode"], root: marketplace }, { runner, env });
  await rename(marketplace, `${marketplace}.moved`);
  const status = await executePluginLifecycle({ action: "status", harnesses: ["opencode"], root: marketplace }, { runner, env });
  assert.equal(status.harnesses[0].installed, false);
  assert.equal(status.harnesses[0].dangling, true);
  const registry = createHarnessRegistry([HARNESS_REGISTRY.resolve("opencode")]);
  const [check] = await harnessRegistrationChecks(registry, env);
  assert.equal(check.status, "fail");
  assert.match(check.detail, /habitus plugin install/);
  const out = sink();
  assert.equal(await main(["plugin", "status", "--harness", "opencode", "--root", marketplace], { stdout: out, stderr: sink(), env }), 0);
  assert.match(out.text(), /入口指向的插件已不在/);
});

test("the doctor flags installed plugins older than the npm package", async (t) => {
  const { env, runner } = await isolated(t);
  const root = join(env.HOME, ".habitus", "plugin-marketplace");
  await executePluginLifecycle({ action: "install", harnesses: ["codex"], root, prepareOnly: true }, { runner, env });
  assert.equal((await marketplaceCheck(HARNESS_REGISTRY, env)).status, "pass");
  const markerPath = join(root, ".habitus-install.json");
  const marker = JSON.parse(await readFile(markerPath, "utf8"));
  await writeFile(markerPath, JSON.stringify({ ...marker, sourceDigest: "0".repeat(64) }));
  const stale = await marketplaceCheck(HARNESS_REGISTRY, env);
  assert.equal(stale.status, "warn");
  assert.match(stale.detail, /habitus plugin update/);
  const status = await executePluginLifecycle({ action: "status", harnesses: ["codex"], root }, { runner, env });
  assert.equal(status.marketplace.stale, true);
});

test("the doctor requires protocols only for harnesses registered on this machine", async () => {
  const service = {
    health: async () => ({ ok: true }),
    ready: async () => ({ ok: true, result: { ready: true, recall_ready: true, status: "healthy" } }),
    capabilities: async () => ({
      ok: true,
      result: {
        api_version: "1.0", service_version: "old", protocols: ["codex_rollout"],
        features: ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"],
      },
    }),
  };
  const options = {
    service,
    inspectFeatures: async () => [],
    inspectMarketplace: async () => ({ name: "marketplace", status: "pass", detail: "" }),
    inspectHookReviews: async () => [],
  };
  const root = await mkdtemp(join(tmpdir(), "habitus-cli-test-"));
  try {
    const codexOnly = await runPluginDoctor({ stateRoot: root }, {
      ...options,
      inspectHarnesses: async () => [{ name: "harness_codex", status: "pass" }, { name: "harness_opencode", status: "warn" }],
    });
    assert.equal(codexOnly.ok, true, JSON.stringify(codexOnly.checks));
    const withOpenCode = await runPluginDoctor({ stateRoot: root }, {
      ...options,
      inspectHarnesses: async () => [{ name: "harness_codex", status: "pass" }, { name: "harness_opencode", status: "pass" }],
    });
    assert.equal(withOpenCode.ok, false);
    assert.match(withOpenCode.checks.find((check) => check.name === "capabilities").detail, /missing_protocols=opencode/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("the doctor reports an unreachable service once, with how to start it", async () => {
  const root = await mkdtemp(join(tmpdir(), "habitus-cli-test-"));
  try {
    const report = await runPluginDoctor({ stateRoot: root }, {
      service: { health: async () => ({ ok: false, error: "fetch failed" }) },
      inspectHarnesses: async () => [],
      inspectFeatures: async () => [],
      inspectMarketplace: async () => ({ name: "marketplace", status: "pass", detail: "" }),
    });
    const names = report.checks.map((check) => check.name);
    assert.ok(!names.includes("capabilities") && !names.includes("ready"));
    assert.match(report.checks.find((check) => check.name === "health").detail, /habitus-server/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("argument edges: plugin --help, --root without a value, and server commands", async () => {
  const out = sink(); const err = sink();
  assert.equal(await main(["plugin", "--help"], { stdout: out, stderr: err }), 0);
  assert.match(out.text(), /habitus plugin install/);
  // T12（裁定 75 第 5 条）：--root 不是用户选项，帮助里不出现（只留给测试）
  assert.doesNotMatch(out.text(), /--root/);
  assert.throws(() => parseArgs(["install", "--root"]), /--root requires a directory/);
  assert.throws(() => parseArgs(["install", "--root", "--json"]), /--root requires a directory/);
  const server = sink();
  assert.equal(await main(["init"], { stdout: out, stderr: server }), 2);
  assert.match(server.text(), /habitus-server init/);
  // 期望值用 join 拼：Windows 上是 \tmp\someone\.habitus\plugin-marketplace，POSIX 上与原来逐字相同
  assert.equal(parseArgs(["status"], { HOME: "/tmp/someone" }).root, join("/tmp/someone", ".habitus", "plugin-marketplace"));
});

// T13（裁定 72 的"本地服务"口径）：命令行说给人看的话里不再有"服务端"：habitus init / serve 的提示，以及 bin/ 下所有不是注释的行
test("the command line calls the Habitus service the local service, never 服务端", async () => {
  for (const command of ["init", "serve"]) {
    const err = sink();
    assert.equal(await main([command], { stdout: sink(), stderr: err }), 2);
    assert.match(err.text(), new RegExp(`本地服务.*habitus-server ${command}`));
    assert.doesNotMatch(err.text(), /服务端/);
  }
  const bin = fileURLToPath(new URL("../../bin/", import.meta.url));
  for (const name of (await readdir(bin)).filter((value) => value.endsWith(".mjs"))) {
    const lines = (await readFile(join(bin, name), "utf8")).split("\n");
    const spoken = lines.filter((line) => !line.trim().startsWith("//") && line.includes("服务端"));
    assert.deepEqual(spoken, [], name);
  }
});

// ── habitus memory（方案 36 第 2 步）──────────────────────────────────────────────────────────────

function tree(uri) { return `/api/v1/memory/tree${uri ? `?uri=${encodeURIComponent(uri)}` : ""}`; }
function documentAt(uri) { return `/api/v1/memory/document?uri=${encodeURIComponent(uri)}`; }

const MEMORY_ROUTES = {
  "/health": { status: "healthy" },
  "/api/v1/capabilities": { api_version: "1.0", service_version: "0.1.0", protocols: [], features: ["memory_tree_v1"] },
  [tree()]: { uri: "memory://", directories: ["memory://entities", "memory://preferences"], documents: ["memory://profile.md"], abstract: null },
  [tree("memory://preferences")]: { uri: "memory://preferences", directories: [], documents: ["memory://preferences/回答风格.md"], abstract: "偏好：回答简洁。" },
  [tree("memory://entities")]: { uri: "memory://entities", directories: ["memory://entities/项目"], documents: [], abstract: null },
  [tree("memory://entities/项目")]: { uri: "memory://entities/项目", directories: [], documents: ["memory://entities/项目/habitus.md"], abstract: null },
  [documentAt("memory://preferences/回答风格.md")]: {
    uri: "memory://preferences/回答风格.md", node: "document", kind: "preference", revision: 2,
    created_at: "2026-10-01T00:00:00Z", updated_at: "2026-10-20T00:00:00Z", last_confirmed_at: null,
    fields: { topic: "回答风格", content: "- 偏好简洁直接的回答" }, text: "",
    links: [{ from_uri: "memory://preferences/回答风格.md", to_uri: "memory://entities/项目/habitus.md", link_type: "related" }], backlinks: [],
  },
  [documentAt("memory://preferences/.overview.md")]: { uri: "memory://preferences/.overview.md", node: "layer", level: "overview", text: "## 偏好\n- 简洁" },
};

async function memoryService(t, routes = MEMORY_ROUTES) {
  const service = await serve(t, routes);
  if (!service) return null;
  return { ...service, env: { HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "1000" } };
}

test("memory ls lists a directory, with or without the memory:// prefix", async (t) => {
  const service = await memoryService(t);
  if (!service) return;
  const root = sink();
  assert.equal(await main(["memory", "ls"], { stdout: root, stderr: sink(), env: service.env }), 0);
  assert.equal(root.text(), "entities/\npreferences/\nprofile.md\n");
  const preferences = sink();
  assert.equal(await main(["memory", "ls", "preferences/"], { stdout: preferences, stderr: sink(), env: service.env }), 0);
  assert.equal(preferences.text(), "偏好：回答简洁。\n\n回答风格.md\n");
});

test("memory tree walks down to the requested depth and marks what it did not open", async (t) => {
  const service = await memoryService(t);
  if (!service) return;
  const shallow = sink();
  assert.equal(await main(["memory", "tree", "-L", "1"], { stdout: shallow, stderr: sink(), env: service.env }), 0);
  assert.equal(shallow.text(), "memory://\n├── entities/ …\n├── preferences/ …\n└── profile.md\n");
  const deep = sink();
  assert.equal(await main(["memory", "tree", "memory://entities"], { stdout: deep, stderr: sink(), env: service.env }), 0);
  assert.equal(deep.text(), "memory://entities\n└── 项目/\n    └── habitus.md\n");
  assert.equal(await main(["memory", "tree", "-L", "9"], { stdout: sink(), stderr: sink(), env: service.env }), 1);
});

test("memory read prints a document's fields and links, or a layer's text", async (t) => {
  const service = await memoryService(t);
  if (!service) return;
  const doc = sink();
  assert.equal(await main(["memory", "read", "memory://preferences/回答风格.md"], { stdout: doc, stderr: sink(), env: service.env }), 0);
  assert.match(doc.text(), /preference · 第 2 版/);
  assert.match(doc.text(), /topic: 回答风格/);
  assert.match(doc.text(), /→ memory:\/\/entities\/项目\/habitus\.md（related）/);
  const layer = sink();
  assert.equal(await main(["memory", "read", "preferences/.overview.md"], { stdout: layer, stderr: sink(), env: service.env }), 0);
  assert.equal(layer.text(), "## 偏好\n- 简洁\n");
  const json = sink();
  assert.equal(await main(["memory", "read", "preferences/.overview.md", "--json"], { stdout: json, stderr: sink(), env: service.env }), 0);
  assert.equal(JSON.parse(json.text()).level, "overview");
  const missing = sink();
  assert.equal(await main(["memory", "read"], { stdout: sink(), stderr: missing, env: service.env }), 1);
  assert.match(missing.text(), /usage: habitus memory/);
});

test("memory commands say when the service is too old or not running", async (t) => {
  const old = await memoryService(t, { ...MEMORY_ROUTES, "/api/v1/capabilities": { api_version: "1.0", service_version: "0.0.9", protocols: [], features: ["recall"] } });
  if (!old) return;
  const tooOld = sink();
  assert.equal(await main(["memory", "ls"], { stdout: sink(), stderr: tooOld, env: old.env }), 1);
  assert.match(tooOld.text(), /服务太老.*memory_tree_v1/);
  const down = sink();
  const env = { HABITUS_URL: "http://127.0.0.1:1", HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "250" };
  assert.equal(await main(["memory", "ls"], { stdout: sink(), stderr: down, env }), 1);
  assert.match(down.text(), /先启动：habitus-server/);
});

// ── habitus reminders（方案 36 第 2 步）─────────────────────────────────────────────────────────────

const REMINDER_DAY = {
  day: "2026-10-22",
  reminders: [
    {
      request_id: "e1", kind: "写周报", text: "周四下午你一般会写周报，这周的还没写。", deadline: "2026-10-22T18:00:00+08:00",
      stage: "说了", awaiting_at: "2026-10-22T15:40:00+08:00", spoken_at: "2026-10-22T15:52:10+08:00", spoken_via: "codex",
      response: "正", responded_at: "2026-10-22T15:53:00+08:00", outcome: "提醒后发生", attempts: 1,
      handoffs: [{ host: "codex", session: "codex-thread-1", claimed_at: "2026-10-22T15:41:00+08:00", ended: "released" }],
    },
    {
      request_id: "e2", kind: "跑备份", text: "今晚的备份还没跑。", deadline: "2026-10-22T23:00:00+08:00",
      stage: "取走了", awaiting_at: "2026-10-22T20:10:00+08:00", spoken_at: null, spoken_via: null,
      response: null, responded_at: null, outcome: null, attempts: 0,
      handoffs: [{ host: "claude-code", session: "claude-s2", claimed_at: "2026-10-22T20:12:00+08:00", ended: null }],
    },
  ],
  awaiting: 0, taken: 1, spoken: 1, never_claimed: 0, responded: 1,
};

// 段 6 Q15：「待取」之后他先做了的那条显示「已发生」，不再永远显示「待取」
test("reminders shows a reminder he did before being asked as happened", async (t) => {
  const happened = {
    ...REMINDER_DAY.reminders[1], request_id: "e3", stage: "已发生", handoffs: [], attempts: 0,
  };
  const service = await serve(t, {
    "/health": { status: "healthy" },
    "/api/v1/capabilities": { api_version: "1.0", service_version: "0.1.0", protocols: [], features: ["reminders_report_v1"] },
    "/api/v1/reminders?day=2026-10-22": { ...REMINDER_DAY, reminders: [happened], spoken: 0, taken: 0, responded: 0 },
  });
  if (!service) return;
  const env = { HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "1000" };
  const out = sink();
  assert.equal(await main(["reminders", "--day", "2026-10-22"], { stdout: out, stderr: sink(), env }), 0);
  assert.match(out.text(), /20:10 {2}跑备份 {2}待取（时限 23:00） → 他先做了（已发生）\n/);
});

test("reminders shows each reminder's stage and the counts R8 needs", async (t) => {
  const service = await serve(t, {
    "/health": { status: "healthy" },
    "/api/v1/capabilities": { api_version: "1.0", service_version: "0.1.0", protocols: [], features: ["reminders_report_v1"] },
    "/api/v1/reminders?day=2026-10-22": REMINDER_DAY,
    "/api/v1/reminders": { ...REMINDER_DAY, reminders: [], spoken: 0, taken: 0, responded: 0, day: "2026-10-21" },
  });
  if (!service) return;
  const env = { HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "1000" };
  const out = sink();
  assert.equal(await main(["reminders", "--day", "2026-10-22"], { stdout: out, stderr: sink(), env }), 0);
  const text = out.text();
  assert.match(text, /^2026-10-22 {2}2 条：说了 1 · 取走了 1 · 待取 0 · 宿主没取 0 · 回话 1 {2}问出口 1\/2 {2}放回 1 次\n/);
  assert.match(text, /15:40 {2}写周报 {2}待取（时限 18:00） → 说了（codex 15:52） → 回话 正 15:53 → 结果 提醒后发生 → 放回 1 次/);
  assert.match(text, /20:10 {2}跑备份 {2}待取（时限 23:00） → claude-code 取走 20:12\n/);
  const today = sink();
  assert.equal(await main(["reminders"], { stdout: today, stderr: sink(), env }), 0);
  assert.equal(today.text(), "2026-10-21  没有进出口的提醒。\n");
  const bad = sink();
  assert.equal(await main(["reminders", "--day", "yesterday"], { stdout: sink(), stderr: bad, env }), 1);
  assert.match(bad.text(), /--day must look like/);
});

test("the registry refuses two harnesses that would write the same manifest and names the supported ones", () => {
  const claude = HARNESS_REGISTRY.resolve("claude-code");
  assert.throws(
    () => createHarnessRegistry([claude, { ...claude, id: "zcode", aliases: ["zcode"], command: "zcode" }]),
    /claude-code and zcode would write the same marketplace:\.claude-plugin\/marketplace\.json/,
  );
  assert.throws(() => HARNESS_REGISTRY.resolve("zed"), /unknown Agent Harness: zed \(supported: codex, claude-code, zcode, opencode, pi, dsh\)/);
});


// 假的 `codex app-server`：按行收 JSON-RPC，initialize 之后答 hooks/list
function fakeAppServer(hooks) {
  return () => {
    const child = new EventEmitter();
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.kill = () => {};
    let buffer = "";
    child.stdin.on("data", (chunk) => {
      buffer += chunk;
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const message = JSON.parse(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
        if (message.id === 1) child.stdout.write(`${JSON.stringify({ id: 1, result: {} })}\n`);
        if (message.id === 2) child.stdout.write(`${JSON.stringify({ id: 2, result: { data: [{ cwd: "/", hooks }] } })}\n`);
      }
    });
    return child;
  };
}

test("Codex hooks the user has not approved are reported with how to approve them", async (t) => {
  const codex = HARNESS_REGISTRY.resolve("codex");
  const hooks = [
    { pluginId: "habitus-memory@habitus-local", eventName: "userPromptSubmit", trustStatus: "untrusted" },
    { pluginId: "habitus-memory@habitus-local", eventName: "stop", trustStatus: "modified" },
    { pluginId: "habitus-memory@habitus-local", eventName: "sessionStart", trustStatus: "trusted" },
    { pluginId: "someone-else@market", eventName: "stop", trustStatus: "untrusted" },
  ];
  const review = await inspectHookReview(codex, { spawnImpl: fakeAppServer(hooks), timeoutMs: 2000 });
  assert.deepEqual(review.pending, ["userPromptSubmit", "stop"]);
  assert.equal(review.hooks.length, 3);

  const registered = [{ name: "harness_codex", status: "pass" }, { name: "harness_claude-code", status: "pass" }];
  const [check] = await harnessHookReviewChecks(HARNESS_REGISTRY, registered, async () => review);
  assert.equal(check.name, "hooks_review_codex");
  assert.equal(check.status, "fail");
  assert.match(check.detail, /2 hook\(s\) not approved \(userPromptSubmit, stop\): 打开 Codex 输入 \/hooks/);

  const approved = await harnessHookReviewChecks(HARNESS_REGISTRY, registered, async () => ({ available: true, hooks: [{}], pending: [] }));
  assert.equal(approved[0].status, "pass");
  // 没注册的宿主不问
  assert.deepEqual(await harnessHookReviewChecks(HARNESS_REGISTRY, [{ name: "harness_codex", status: "warn" }], async () => review), []);
  // 宿主没装或不答：说查不到，不算批没批
  const missing = await inspectHookReview({ ...codex, command: "habitus-no-such-harness" }, { timeoutMs: 2000, env: (await isolatedEnv(t)).env });
  assert.equal(missing.available, false);
  const [unknown] = await harnessHookReviewChecks(HARNESS_REGISTRY, registered, async () => missing);
  assert.equal(unknown.status, "warn");
});


test("the doctor runs the hook approval check and reports a service error in words", async () => {
  const root = await mkdtemp(join(tmpdir(), "habitus-cli-test-"));
  try {
    const report = await runPluginDoctor({ stateRoot: root }, {
      service: { health: async () => ({ ok: false, error: { code: "UNAVAILABLE", message: "runtime is starting" } }) },
      inspectHarnesses: async () => [{ name: "harness_codex", status: "pass" }],
      inspectFeatures: async () => [],
      inspectMarketplace: async () => ({ name: "marketplace", status: "pass", detail: "" }),
      inspectHookReviews: async (_registry, registrations) => {
        assert.deepEqual(registrations.map((check) => check.name), ["harness_codex"]);  // 拿到的是注册检查的结果
        return [{ name: "hooks_review_codex", status: "fail", detail: "1 hook(s) not approved" }];
      },
    });
    const byName = Object.fromEntries(report.checks.map((check) => [check.name, check]));
    assert.equal(byName.hooks_review_codex.status, "fail");
    assert.equal(report.ok, false);
    assert.match(byName.health.detail, /^UNAVAILABLE: runtime is starting; start it with: habitus-server$/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("install and status say whether Codex will run the hooks, and never call an empty list approved", async () => {
  const item = (hookReview) => ({ harness: "codex", installed: true, enabled: true, hookReview });
  assert.equal(hookReviewState(item(null)), "");
  assert.equal(hookReviewState(item({ available: true, hooks: [{}, {}], pending: [] })), "；hook 已批准");
  assert.match(hookReviewState(item({ available: true, hooks: [{}], pending: ["stop"] })), /1 个 hook 还没批准，不会运行：打开 Codex 输入 \/hooks/);
  assert.match(hookReviewState(item({ available: true, hooks: [], pending: [] })), /没列出 Habitus 的 hook，不会运行/);
  assert.match(hookReviewState(item({ available: false, error: "x" })), /查不到/);
  // Codex 一条都没列：doctor 判失败，并先指向特性开关
  const codex = HARNESS_REGISTRY.resolve("codex");
  assert.equal(codex.hookReview.kind, "codex-app-server");
  return harnessHookReviewChecks(HARNESS_REGISTRY, [{ name: "harness_codex", status: "pass" }], async () => ({ available: true, hooks: [], pending: [] }))
    .then(([check]) => {
      assert.equal(check.status, "fail");
      assert.match(check.detail, /check features_codex above/);
    });
});

test("a harness whose hook approval protocol is unknown is a definition error", () => {
  assert.throws(() => inspectHookReview({ id: "x", hookReview: { kind: "nope" } }), /unknown hookReview kind/);
});


test("plugin install / status ask the harness whether its hooks are approved and say how to approve", async () => {
  const lifecycle = async (options) => ({
    action: options.action, root: "/tmp/m",
    harnesses: [{ harness: "codex", installed: true, enabled: true }, { harness: "opencode", installed: true, enabled: true }],
  });
  const asked = [];
  const reviewHooks = async (definition) => { asked.push(definition.id); return { available: true, hooks: [{}], pending: ["stop"] }; };
  for (const action of ["install", "status"]) {
    const out = sink();
    assert.equal(await main(["plugin", action, "--harness", "codex"], {
      stdout: out, stderr: sink(), lifecycle, reviewHooks, featureGaps: () => ({ available: true, missing: [], advisory: [] }),
    }), 0);
    assert.match(out.text(), /codex +已注册；1 个 hook 还没批准，不会运行：打开 Codex 输入 \/hooks/);
  }
  assert.deepEqual(asked, ["codex", "codex"]);  // 只问要求逐条批准的宿主（OpenCode 不问）
  // 只准备、不注册时不问
  assert.equal(await main(["plugin", "install", "--prepare-only"], { stdout: sink(), stderr: sink(), lifecycle, reviewHooks }), 0);
  assert.deepEqual(asked, ["codex", "codex"]);
});

test("the doctor reports a not-ready service in words", async () => {
  const root = await mkdtemp(join(tmpdir(), "habitus-cli-test-"));
  try {
    const report = await runPluginDoctor({ stateRoot: root }, {
      service: {
        health: async () => ({ ok: true }),
        ready: async () => ({ ok: false, error: { code: "NOT_READY", message: "vector index building" } }),
        capabilities: async () => ({ ok: false, error: "x" }),
      },
      inspectHarnesses: async () => [], inspectFeatures: async () => [], inspectHookReviews: async () => [],
      inspectMarketplace: async () => ({ name: "marketplace", status: "pass", detail: "" }),
    });
    const ready = report.checks.find((check) => check.name === "ready");
    assert.equal(ready.detail, "NOT_READY: vector index building");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

// ── /ready 的 503 带着各项检查（B9）────────────────────────────────────────────────────────────────

const NOT_READY = {
  status: "unhealthy", ready: false, recall_ready: false, checked_at: "2026-10-23T00:00:00Z",
  checks: [
    { name: "memory_worker", status: "unhealthy", detail: "worker stopped: VectorStoreError", critical: true },
    { name: "foresight", status: "healthy", detail: "ok", critical: false },
  ],
};

test("a not-ready service answers /ready with 503 and its checks; status and the doctor name the unhealthy ones", async (t) => {
  const service = await serve(t, {
    "/health": { status: "unhealthy", ready: false, recall_ready: false, checked_at: "2026-10-23T00:00:00Z", checks: [] },
    "/ready": NOT_READY,
    "/api/v1/capabilities": { api_version: "1.0", service_version: "0.1.0", protocols: ["codex_rollout"], features: ["recall"] },
  }, { "/ready": 503 });
  if (!service) return;
  const env = { HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "1000" };
  const human = sink();
  assert.equal(await main(["status"], { stdout: human, stderr: sink(), env }), 1);
  assert.match(human.text(), /未就绪/);
  assert.match(human.text(), /! memory_worker: unhealthy worker stopped: VectorStoreError/);
  const json = sink();
  assert.equal(await main(["status", "--json"], { stdout: json, stderr: sink(), env }), 1);
  const report = JSON.parse(json.text());
  assert.equal(report.ready, false);
  assert.deepEqual(report.checks.map((check) => check.name), ["memory_worker", "foresight"]);

  const client = new HabitusServiceClient({ baseUrl: service.url, timeoutMs: 1000 });
  const doctor = await runPluginDoctor({ stateRoot: env.HABITUS_PLUGIN_STATE_DIR }, {
    service: client,
    inspectHarnesses: async () => [], inspectFeatures: async () => [], inspectHookReviews: async () => [],
    inspectMarketplace: async () => ({ name: "marketplace", status: "pass", detail: "" }),
  });
  const ready = doctor.checks.find((check) => check.name === "ready");
  assert.equal(ready.status, "fail");
  assert.equal(ready.detail, "unhealthy; memory_worker: unhealthy worker stopped: VectorStoreError");
  // 别的接口回 503 仍是失败，不按成功读
  assert.equal((await client.request("/ready")).ok, false);
});

// ── 子命令的 --help（B19）──────────────────────────────────────────────────────────────────────────

test("every subcommand's --help prints its usage and sends nothing; a non-existent --day is refused locally", async (t) => {
  const service = await serve(t, {});
  if (!service) return;
  const env = { HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "1000" };
  const cases = [
    [["doctor", "--help"], /usage: habitus doctor/],
    [["status", "-h"], /usage: habitus status/],
    [["harnesses", "--help"], /usage: habitus harnesses/],
    [["recall", "--help"], /usage: habitus recall/],
    [["recall", "回答风格", "--help"], /usage: habitus recall/],
    [["memory", "ls", "--help"], /usage: habitus memory ls/],
    [["memory", "tree", "-L", "2", "--help"], /usage: habitus memory ls/],
    [["memory", "read", "--help"], /usage: habitus memory ls/],
    [["reminders", "--help"], /usage: habitus reminders/],
  ];
  for (const [argv, usage] of cases) {
    const out = sink(); const err = sink();
    assert.equal(await main(argv, { stdout: out, stderr: err, env }), 0, `${argv.join(" ")}: ${err.text()}`);
    assert.match(out.text(), usage, argv.join(" "));
  }
  for (const day of ["2026-02-30", "2026-02-29", "2026-13-01", "2026-04-31"]) {
    const err = sink();
    assert.equal(await main(["reminders", "--day", day], { stdout: sink(), stderr: err, env }), 1);
    assert.match(err.text(), new RegExp(`--day ${day} 日期不存在`));
  }
  assert.deepEqual(service.calls, []);
});

// 杀变异 BN16（memory ls 多给的参数不报错）：多出来的参数报出来，不当成没看见、也不去问服务
test("memory ls with two addresses names the extra argument and sends nothing", async (t) => {
  const service = await memoryService(t);
  if (!service) return;
  const err = sink();
  assert.equal(await main(["memory", "ls", "preferences", "entities"], { stdout: sink(), stderr: err, env: service.env }), 1);
  assert.match(err.text(), /unknown argument: entities/);
  assert.deepEqual(service.calls, []);
});

// 杀变异 BN13（reminders 不查 reminders_report_v1）：老服务没有这张报表，先说"服务太老"，不去撞接口
test("reminders says the service is too old when it has no reminders report", async (t) => {
  const service = await serve(t, {
    "/health": { status: "healthy" },
    "/api/v1/capabilities": { api_version: "1.0", service_version: "0.0.9", protocols: [], features: ["reminders_v1"] },
    "/api/v1/reminders": REMINDER_DAY,
  });
  if (!service) return;
  const env = { HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "1000" };
  const err = sink();
  assert.equal(await main(["reminders"], { stdout: sink(), stderr: err, env }), 1);
  assert.match(err.text(), /服务太老.*reminders_report_v1/);
  assert.equal(service.calls.some((call) => call.url.startsWith("/api/v1/reminders")), false);
});

// 杀变异 BN15（"谁取走了"取第一次交接）：放回过又被别的宿主取走的提醒，显示现在拿着它的那一个
test("reminders shows the host that holds a reminder now, not the one that put it back", async (t) => {
  const passedOn = {
    ...REMINDER_DAY.reminders[1],
    attempts: 1,
    handoffs: [
      { host: "codex", session: "codex-thread-1", claimed_at: "2026-10-22T20:11:00+08:00", ended: "released" },
      { host: "claude-code", session: "claude-s2", claimed_at: "2026-10-22T20:12:00+08:00", ended: null },
    ],
  };
  const service = await serve(t, {
    "/health": { status: "healthy" },
    "/api/v1/capabilities": { api_version: "1.0", service_version: "0.1.0", protocols: [], features: ["reminders_report_v1"] },
    "/api/v1/reminders?day=2026-10-22": { ...REMINDER_DAY, reminders: [passedOn], spoken: 0, responded: 0 },
  });
  if (!service) return;
  const env = { HABITUS_URL: service.url, HABITUS_PLUGIN_STATE_DIR: await stateRoot(t), HABITUS_PLUGIN_TIMEOUT_MS: "1000" };
  const out = sink();
  assert.equal(await main(["reminders", "--day", "2026-10-22"], { stdout: out, stderr: sink(), env }), 0);
  assert.match(out.text(), /跑备份 {2}待取（时限 23:00） → claude-code 取走 20:12 → 放回 1 次\n/);
});

// 杀变异 BN02（没装上的宿主也去问 hook 批没批）：只问已注册、已启用的宿主；没装上的问了也是白问（还要起宿主进程）
test("plugin status asks about hook approval only for harnesses that are registered", async () => {
  const lifecycle = async (options) => ({
    action: options.action, root: "/tmp/m",
    harnesses: [{ harness: "codex", available: true, installed: false, enabled: false }],
  });
  const asked = [];
  const reviewHooks = async (definition) => { asked.push(definition.id); return { available: true, hooks: [{}], pending: ["stop"] }; };
  const out = sink();
  assert.equal(await main(["plugin", "status"], { stdout: out, stderr: sink(), lifecycle, reviewHooks }), 0);
  assert.deepEqual(asked, []);
  assert.match(out.text(), /codex +未注册\n/);
});

// 杀变异 RG10（managed 的 hook 也算没批）：Codex 判成 managed（受管理员策略管）的 hook 与 trusted 一样算批了
test("Codex hooks that are managed count as approved", async () => {
  const codex = HARNESS_REGISTRY.resolve("codex");
  const hooks = [
    { pluginId: "habitus-memory@habitus-local", eventName: "stop", trustStatus: "managed" },
    { pluginId: "habitus-memory@habitus-local", eventName: "sessionStart", trustStatus: "trusted" },
  ];
  const review = await inspectHookReview(codex, { spawnImpl: fakeAppServer(hooks), timeoutMs: 2000 });
  assert.deepEqual(review.pending, []);
  assert.equal(review.hooks.length, 2);
});

// 杀变异 RG17（app-server 不答时不超时）：`codex app-server` 起来了却一直不答，到点给"没答"并把它杀掉，doctor / plugin status 不挂住
test("a Codex app-server that never answers times out and is killed", { timeout: 3000 }, async () => {
  let killed = false;
  const silent = () => {
    const child = new EventEmitter();
    child.stdin = new PassThrough();
    child.stdout = new PassThrough();
    child.kill = () => { killed = true; };
    return child;
  };
  const review = await inspectHookReview(HARNESS_REGISTRY.resolve("codex"), { spawnImpl: silent, timeoutMs: 50 });
  assert.deepEqual(review, { available: false, error: "harness did not answer" });
  assert.equal(killed, true);
});

const HABITUS = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "bin", "habitus.mjs");

// 真进程跑 habitus 命令：limitMs 到了还没退就 SIGKILL 并记下（挂住的那种，测试里不能无限等）
function runHabitus(args, env, limitMs) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const child = spawn(process.execPath, [HABITUS, ...args], { env, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr += chunk; });
    let hung = false;
    const limit = setTimeout(() => { hung = true; child.kill("SIGKILL"); }, limitMs);
    child.once("error", reject);
    child.once("exit", (code) => { clearTimeout(limit); resolve({ code, hung, elapsed: Date.now() - started, stdout, stderr }); });
  });
}

function alive(pid) {
  try { process.kill(pid, 0); return true; }
  catch (error) { return error?.code !== "ESRCH"; }
}

// 杀变异 T23（问完只发一次 SIGTERM、不等它退）：app-server 是真 OS 进程、忽略 SIGTERM（或退了但孙进程占着 stdout）时，
// habitus doctor / plugin status 照样在"hook review 超时 + 固定宽限"内退出，退出时 app-server 已不在。超时取 HABITUS_PLUGIN_TIMEOUT_MS
for (const mode of ["silent-ignore-term", "answer-ignore-term", "grandchild"]) {
  for (const args of [["doctor", "--json"], ["plugin", "status", "--json"]]) {
    test(`habitus ${args.filter((arg) => arg !== "--json").join(" ")} exits and leaves no app-server behind (app-server: ${mode})`, { timeout: 20_000 }, async (t) => {
      const isolated = await isolatedEnv(t, { fakeCodex: true });
      // 留够起 app-server 的时间：Windows 上是 cmd.exe 再起 node，忙的 runner 上 500 ms 内 node 还没记下自己的 pid 就被整树杀掉
      // （windows-latest 3.13 实测 "the fake app-server was never started"）
      const timeoutMs = 2000;
      const env = {
        ...isolated.env,
        FAKE_CODEX_APP_SERVER: mode,
        HABITUS_URL: "http://127.0.0.1:1",
        HABITUS_PLUGIN_TIMEOUT_MS: String(timeoutMs),
      };
      const result = await runHabitus(args, env, 15_000);
      assert.equal(result.hung, false, `habitus ${args.join(" ")} hung: ${result.stderr}`);
      // 超时 + 两段宽限，外加起进程、别的检查的余量（仍远小于默认的 15 s）
      assert.ok(result.elapsed < timeoutMs + 2 * HOOK_REVIEW_KILL_GRACE_MS + 5000, `took ${result.elapsed} ms`);
      const report = JSON.parse(result.stdout);
      const review = args[0] === "doctor"
        ? report.checks.find((check) => check.name === "hooks_review_codex")
        : report.harnesses.find((item) => item.harness === "codex").hookReview;
      assert.ok(review, result.stdout);
      const [server] = (await isolated.pids()).filter((item) => item.role === "app-server");
      assert.ok(server, "the fake app-server was never started");
      assert.equal(alive(server.pid), false, `app-server ${server.pid} is still running`);
    });
  }
}

// 杀变异 T23（spawnSync 的超时只发 SIGTERM）：`codex features list` 忽略 SIGTERM、一直不答，doctor 照样在它的 5 s 超时 + 余量内返回，记 warn
test("habitus doctor returns when the harness feature list ignores SIGTERM and never answers", { timeout: 30_000 }, async (t) => {
  const isolated = await isolatedEnv(t, { fakeCodex: true });
  const env = { ...isolated.env, FAKE_CODEX_FEATURES: "hang", HABITUS_URL: "http://127.0.0.1:1", HABITUS_PLUGIN_TIMEOUT_MS: "500" };
  const result = await runHabitus(["doctor", "--json"], env, 25_000);
  assert.equal(result.hung, false, result.stderr);
  assert.ok(result.elapsed < 5000 + 2 * HOOK_REVIEW_KILL_GRACE_MS + 5000, `took ${result.elapsed} ms`);
  const features = JSON.parse(result.stdout).checks.find((check) => check.name === "features_codex");
  assert.equal(features.status, "warn");
  const [listing] = (await isolated.pids()).filter((item) => item.role === "features");
  assert.equal(alive(listing.pid), false);
});

// 杀变异 T23（hook review 的超时写死 15 s）：doctor 与 plugin status 把 HABITUS_PLUGIN_TIMEOUT_MS 交给 hook review
test("the hook review uses HABITUS_PLUGIN_TIMEOUT_MS in doctor and plugin status", async (t) => {
  const isolated = await isolatedEnv(t);
  const env = { ...isolated.env, HABITUS_PLUGIN_TIMEOUT_MS: "250" };
  let doctorOptions = null;
  await runPluginDoctor(loadPluginConfig(env), {
    env,
    service: { health: async () => ({ ok: false, error: new Error("down") }) },
    inspectHarnesses: async () => [{ name: "harness_codex", status: "pass" }],
    inspectFeatures: async () => [],
    inspectMarketplace: async () => ({ name: "marketplace", status: "pass", detail: "" }),
    inspectHookReviews: async (_registry, _registrations, _review, options) => { doctorOptions = options; return []; },
  });
  assert.equal(doctorOptions.timeoutMs, 250);
  assert.equal(doctorOptions.env, env);

  const reviewed = [];
  const lifecycle = async (options) => ({
    action: options.action, root: "/tmp/m",
    harnesses: [{ harness: "codex", available: true, installed: true, enabled: true }],
  });
  const reviewHooks = async (_definition, options) => { reviewed.push(options); return { available: true, hooks: [{}], pending: [] }; };
  const featureGaps = () => ({ available: true, missing: [], advisory: [] });
  assert.equal(await main(["plugin", "status"], { stdout: sink(), stderr: sink(), env, lifecycle, reviewHooks, featureGaps }), 0);
  assert.equal(reviewed[0].timeoutMs, 250);
  assert.equal(reviewed[0].env, env);
});
