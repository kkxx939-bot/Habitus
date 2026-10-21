import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { main } from "../../bin/habitus.mjs";
import { HARNESS_REGISTRY, createHarnessRegistry } from "../../harnesses.mjs";
import { harnessFeatureChecks, parseFeatureList } from "../doctor.mjs";

function sink() {
  const chunks = [];
  return { write: (value) => { chunks.push(String(value)); }, text: () => chunks.join("") };
}

async function serve(t, routes) {
  const calls = [];
  const server = createServer((request, response) => {
    let body = "";
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      calls.push({ url: request.url, body: body ? JSON.parse(body) : null });
      const result = routes[request.url];
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
    "/health": { status: "healthy", ready: true, checked_at: "2026-10-21T00:00:00Z", checks: [] },
    "/ready": { status: "degraded", ready: true, checked_at: "2026-10-21T00:00:00Z", checks: [{ name: "foresight", status: "degraded", detail: "no table yet", critical: false }] },
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

test("harnesses lists the registered hosts", async () => {
  const out = sink();
  assert.equal(await main(["harnesses", "--json"], { stdout: out, stderr: sink() }), 0);
  const ids = JSON.parse(out.text()).harnesses.map((item) => item.id);
  assert.deepEqual(ids, HARNESS_REGISTRY.list().map((item) => item.id));
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
  const off = harnessFeatureChecks(registry, () => ({ status: 0, stdout: listing }));
  return off.then((checks) => {
    assert.deepEqual(checks.map((item) => [item.name, item.status]), [["features_codex", "fail"]]);
    assert.match(checks[0].detail, /features off: hooks/);
    return harnessFeatureChecks(registry, () => ({ status: 0, stdout: listing.replace("stable             false", "stable             true") }));
  }).then((checks) => {
    assert.equal(checks[0].status, "pass");
    return harnessFeatureChecks(registry, () => ({ error: Object.assign(new Error("missing"), { code: "ENOENT" }) }));
  }).then((checks) => {
    assert.deepEqual(checks, []);
  });
});

test("the repository root carries the same marketplace manifests the installer stages", async () => {
  // 用户可以不经 npm、直接用宿主原生命令装：`codex plugin marketplace add <仓库>` / `claude plugin marketplace add <仓库>`。
  // 根目录那两份必须与安装器暂存的一模一样（相对路径 ./plugins/<目录> 两处都成立）
  const { readFile } = await import("node:fs/promises");
  const repository = join(new URL(".", import.meta.url).pathname, "..", "..", "..");
  for (const definition of HARNESS_REGISTRY.list().filter((item) => item.marketplaceManifest)) {
    const written = JSON.parse(await readFile(join(repository, definition.marketplaceManifest.path), "utf8"));
    assert.deepEqual(written, definition.marketplaceManifest.document, definition.id);
  }
});
