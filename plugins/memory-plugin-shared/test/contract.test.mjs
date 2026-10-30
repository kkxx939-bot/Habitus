// 插件与本地服务之间的契约：插件这一侧的常量逐项对着与服务端共用的 fixture 断言。
// 服务端那一半在 tests/unit/integrations/test_service_contract.py，对着同一份 fixture；任何一侧单独改了契约，两边至少一边失败。
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { HARNESS_REGISTRY, createHarnessRegistry } from "../../harnesses.mjs";
import { claudeCodeAdapter } from "../../habitus-memory-claude-code/scripts/host-adapter.mjs";
import { zcodeAdapter } from "../../habitus-memory-claude-code/scripts/zcode-adapter.mjs";
import { hostAdapter as codexAdapter } from "../../habitus-memory/scripts/host-adapter.mjs";
import { hostAdapter as dshAdapter } from "../../habitus-memory-dsh/scripts/host-adapter.mjs";
import { hostAdapter as opencodeAdapter } from "../../habitus-memory-opencode/scripts/host-adapter.mjs";
import { hostAdapter as piAdapter } from "../../habitus-memory-pi/scripts/host-adapter.mjs";
import {
  API_VERSION, DEFAULT_MAX_REQUEST_BYTES, REMEMBER_PARTS_FEATURE, REMINDERS_FEATURE, REQUIRED_FEATURES, maxRequestBytes,
} from "../lib/compat.mjs";
import { rememberPartBody, splitRememberBody } from "../lib/remember-parts.mjs";
import { loadPluginConfig } from "../lib/config.mjs";
import { stripMedia } from "../lib/media.mjs";
import { HabitusServiceClient } from "../lib/service-client.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const CONTRACT = JSON.parse(await readFile(join(HERE, "fixtures", "service-contract.json"), "utf8"));

// 每个宿主在 hook 进程里实际用的适配器（ZCode 与 Claude Code 共用一个插件目录，按环境变量换适配器）
const ADAPTERS = {
  codex: codexAdapter,
  "claude-code": claudeCodeAdapter,
  zcode: zcodeAdapter,
  opencode: opencodeAdapter,
  pi: piAdapter,
  dsh: dshAdapter,
};

// 认得出子代理的宿主：hook 输入里哪个字段说明这是子代理（Codex 另从转写开头的 session_meta 认，见 hook-runner.test）
const SUBAGENT_INPUTS = {
  codex: { session_id: "parent", agent_id: "child" },
  "claude-code": { session_id: "parent", agent_id: "child" },
  opencode: { session_id: "child", parent_id: "parent" },
  dsh: { session_id: "child", parent_session: "parent" },
};

test("every registered harness speaks the protocol, host and question tools its adapter and the service agree on", () => {
  const ids = HARNESS_REGISTRY.list().map((definition) => definition.id);
  assert.deepEqual(Object.keys(ADAPTERS).sort(), [...ids].sort(), "every harness needs its adapter listed here");
  for (const definition of HARNESS_REGISTRY.list()) {
    const adapter = ADAPTERS[definition.id];
    const served = CONTRACT.protocols[definition.protocol];
    assert.equal(adapter.protocol, definition.protocol, `${definition.id}: registry protocol differs from its adapter`);
    assert.ok(served, `${definition.id}: the service does not serve protocol ${definition.protocol}`);
    assert.equal(adapter.host, definition.id, `${definition.id}: adapter host differs from the harness id`);
    assert.equal(served.host, adapter.host, `${definition.id}: the service names this protocol's host differently`);
    assert.ok(served.question_tools.length > 0, `${definition.id}: the service knows no question tool for ${definition.protocol}`);
    for (const tool of served.question_tools) {
      assert.ok(
        typeof adapter.questionTool === "string" && adapter.questionTool.includes(tool),
        `${definition.id}: adapter questionTool ${adapter.questionTool} does not name ${tool}`,
      );
    }
  }
});

test("the features the plugin requires are features the service declares, under the same API version", () => {
  assert.equal(API_VERSION, CONTRACT.api_version);
  for (const feature of [...REQUIRED_FEATURES, REMINDERS_FEATURE, REMEMBER_PARTS_FEATURE]) {
    assert.ok(CONTRACT.features.includes(feature), `the service does not declare ${feature}`);
  }
});

// 分块上传（remember_parts_v1）：插件把这一批请求体切成的块逐字等于 fixture，每块都在上限以内；
// 服务端那一半（收下这几块拼回原请求体、按 remember 处理）在 tests/unit/integrations/test_service_contract.py
test("a batch over the request limit is cut into exactly the parts of the shared fixture, each under the limit", () => {
  const { feature, max_request_bytes: limit, body, parts } = CONTRACT.remember_parts;
  assert.equal(feature, REMEMBER_PARTS_FEATURE);
  assert.ok(Buffer.byteLength(body) > limit);
  const cut = splitRememberBody(JSON.parse(body).delivery_id, body, limit);
  assert.deepEqual(cut, parts);
  assert.ok(cut.every((part) => Buffer.byteLength(rememberPartBody(part)) <= limit));
  assert.equal(Buffer.concat(cut.map((part) => Buffer.from(part.data, "base64"))).toString("utf8"), body);
});

// 子代理的会话插件一概不送（服务端会话 lane 另按 subagent_mark 跳过，那一半在服务端的契约测试里）
test("every host that can tell a subagent session recognises it, and its main session is not mistaken for one", () => {
  for (const [id, input] of Object.entries(SUBAGENT_INPUTS)) {
    const adapter = ADAPTERS[id];
    assert.equal(adapter.isSubagent(input), true, `${id}: a subagent input is not recognised`);
    assert.equal(adapter.isSubagent({ session_id: "parent" }), false, `${id}: the main session is taken for a subagent`);
  }
});

test("the plugin accepts exactly the loopback hosts of the shared table", () => {
  for (const { host, loopback } of CONTRACT.loopback_hosts) {
    const url = `http://${host.includes(":") ? `[${host}]` : host}:8787`;
    let accepted;
    try {
      loadPluginConfig({ HABITUS_URL: url, HABITUS_PLUGIN_STATE_DIR: join(HERE, "no-such-state") });
      accepted = true;
    } catch {
      accepted = false;
    }
    assert.equal(accepted, loopback, `${host}: expected loopback=${loopback}`);
  }
});

// 杀变异 SC02（分块送到了别的路由）：每一块 POST 到 fixture 里的路径、请求体逐字是 fixture 的那一块；服务端按同一个路径收
test("each remember part is posted to the path the shared fixture names, with the fixture's part as its body", async () => {
  const { path, parts } = CONTRACT.remember_parts;
  const requests = [];
  const fetchImpl = async (url, init) => {
    requests.push({ url, method: init.method, body: init.body });
    return new Response(JSON.stringify({ status: "ok", result: {} }), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  const client = new HabitusServiceClient({ baseUrl: "http://127.0.0.1:8787", timeoutMs: 1000 }, fetchImpl);
  for (const part of parts) await client.rememberPart(part);
  assert.deepEqual(requests, parts.map((part) => ({ url: `http://127.0.0.1:8787${path}`, method: "POST", body: rememberPartBody(part) })));
});

// 杀变异 CP02（0、负数的上限也收下）：服务给的 max_request_bytes 不是正整数时按默认上限拆批
test("a request limit that is not a positive integer falls back to the default", () => {
  for (const value of [0, -1, 1.5, "16777216", null]) {
    assert.equal(maxRequestBytes({ max_request_bytes: value }), DEFAULT_MAX_REQUEST_BYTES, String(value));
  }
  assert.equal(maxRequestBytes({ max_request_bytes: 1024 }), 1024);
});

// 杀变异 HN03（Claude Code 的必需 hook 少了 PreCompact）：插件 hooks.json 里挂的每个 hook 都得是共用这个插件目录的某个宿主的必需 hook
// （ZCode 借 Claude Code 的插件、只认其中一部分事件），每个宿主的必需 hook 都挂在 hooks.json 里；doctor 的 manifests 检查才拦得住少挂了一个的插件
test("the hooks a plugin's hooks.json registers are the required hooks of the harnesses sharing that plugin", async () => {
  const byDirectory = new Map();
  for (const item of HARNESS_REGISTRY.list().filter((value) => value.hooksManifest)) {
    byDirectory.set(item.pluginDirectory, [...(byDirectory.get(item.pluginDirectory) ?? []), item]);
  }
  for (const [directory, definitions] of byDirectory) {
    const manifest = JSON.parse(await readFile(join(HERE, "..", "..", directory, definitions[0].hooksManifest), "utf8"));
    const registered = Object.keys(manifest.hooks).sort();
    assert.deepEqual([...new Set(definitions.flatMap((item) => item.requiredHooks))].sort(), registered, directory);
  }
});

// 杀变异 HN02（特性清单里提示项的形状不再校验）：提示项要有 name 与 note，缺了是定义错误
test("a harness whose advisory feature lacks a name or a note is a definition error", () => {
  const codex = HARNESS_REGISTRY.resolve("codex");
  for (const advisory of [[{ name: "x" }], [{ note: "y" }], "x"]) {
    assert.throws(
      () => createHarnessRegistry([{ ...codex, featureInventory: { ...codex.featureInventory, advisory } }]),
      /harness feature inventory is invalid: codex/,
    );
  }
});

async function listen(t, handler) {
  const server = createServer(handler);
  try {
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => { server.removeListener("error", reject); resolve(); });
    });
  } catch (error) {
    if (error?.code === "EPERM") return null;
    throw error;
  }
  t.after(() => { server.closeAllConnections(); server.close(); });
  return server;
}

// T15（README：本地服务只认回环地址）：插件的请求不跟随重定向。回环端口上回 3xx 的，转写不会被送到 Location 指的别处，
// 当成这一次请求失败
test("the plugin never follows a redirect: a 3xx from the loopback port is a failed request and nothing reaches its target", async (t) => {
  const received = [];
  const target = await listen(t, (request, response) => {
    received.push(`${request.method} ${request.url}`);
    request.resume();
    request.on("end", () => { response.writeHead(200, { "Content-Type": "application/json" }); response.end('{"status":"ok","result":{"next_sequence":1}}'); });
  });
  if (target == null) { t.skip("sandbox does not permit a loopback listener"); return; }
  for (const status of [302, 307, 308]) {
    const redirecting = await listen(t, (request, response) => {
      request.resume();
      request.on("end", () => {
        response.writeHead(status, { Location: `http://127.0.0.1:${target.address().port}${request.url}` });
        response.end();
      });
    });
    const client = new HabitusServiceClient({ baseUrl: `http://127.0.0.1:${redirecting.address().port}`, timeoutMs: 2000 });
    const remembered = await client.remember({
      deliveryId: "d", conversationId: "c", startedOn: "2026-10-26", protocol: "codex_rollout",
      payload: { records: [{ text: "SECRET TRANSCRIPT TEXT" }] }, startSequence: 0, occurredAt: "2026-10-26T00:00:00Z", afterTurn: true,
    });
    assert.equal(remembered.ok, false, `remember after ${status}`);
    assert.equal((await client.capabilities()).ok, false, `capabilities after ${status}`);
  }
  assert.deepEqual(received, []);
});

// 杀变异 J31（音频不当媒体）：audio / input_audio 的 base64 也换成占位
test("audio payloads are replaced by placeholders like images", () => {
  for (const type of ["audio", "input_audio"]) {
    const data = "U".repeat(4000);
    const stripped = stripMedia({ type, data, mimeType: "audio/wav" });
    assert.deepEqual(Object.keys(stripped).sort(), ["media_type", "omitted_bytes", "omitted_sha256", "type"], type);
    assert.equal(stripped.omitted_bytes, 4000, type);
    assert.equal(JSON.stringify(stripped).includes(data), false, type);
  }
});
