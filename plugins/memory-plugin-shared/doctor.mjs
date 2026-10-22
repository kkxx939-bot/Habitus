import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { access, readdir, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  HARNESS_REGISTRY,
  PLUGIN_NAME,
} from "../harnesses.mjs";
import { defaultMarketplaceRoot, marketplaceStatus } from "../install-memory-plugin.mjs";
import { CommandRunner, inspectHookReview, registrationOf } from "../registrations.mjs";
import { loadPluginConfig } from "./lib/config.mjs";
import { errorText } from "./lib/service-client.mjs";
import { HabitusServiceClient } from "./lib/service-client.mjs";
import { readState, sessionKey } from "./lib/state-store.mjs";

const REQUIRED_FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];
const PLUGINS_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const GENERATED_HEADER = "// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.\n";
const MAX_PENDING_AGE_MS = 24 * 60 * 60 * 1000;

export async function runPluginDoctor(
  config = loadPluginConfig(),
  {
    service = new HabitusServiceClient(config),
    registry = HARNESS_REGISTRY,
    env = process.env,
    protocols = null,
    inspectHarnesses = harnessRegistrationChecks,
    inspectFeatures = harnessFeatureChecks,
    inspectMarketplace = marketplaceCheck,
    inspectHookReviews = harnessHookReviewChecks,
  } = {},
) {
  const checks = [];
  checks.push(nodeRuntimeCheck(), ...(await bundleChecks(registry)));
  const registrations = await inspectHarnesses(registry, env);
  checks.push(...registrations, await inspectMarketplace(registry, env));
  checks.push(...await inspectFeatures(registry));
  checks.push(...await inspectHookReviews(registry, registrations));
  const health = await service.health();
  if (!health.ok) {
    // 连不上就只报这一条：就绪、特性、提醒都无从谈起，别把它们报成一串不相干的失败
    checks.push({ name: "health", status: "fail", detail: `${errorText(health.error)}; start it with: habitus-server` });
  } else {
    checks.push({ name: "health", status: "pass", detail: "service reachable" });
    checks.push(...await serviceChecks(service, protocols ?? registeredProtocols(registry, registrations)));
  }
  try {
    const inventory = await outboxInventory(config.stateRoot);
    const failed = inventory.blocked > 0
      || inventory.expiredInflight > 0
      || inventory.missingState > 0
      || inventory.invalidState > 0;
    checks.push({
      name: "outbox",
      status: failed ? "fail" : inventory.aged > 0 ? "warn" : "pass",
      detail: Object.entries(inventory).map(([status, count]) => `${status}_items=${count}`).join("; "),
    });
  } catch (error) {
    checks.push({ name: "outbox", status: "fail", detail: error?.message || String(error) });
  }
  return { ok: checks.every((check) => check.status !== "fail"), checks };
}

// 只有本机已注册的宿主，它的会话协议缺了才算失败；别的宿主缺协议只提示（新 CLI 连老服务时只用 Codex 的人照常能用）
function registeredProtocols(registry, registrations) {
  const registered = new Set(registrations.filter((check) => check.status === "pass").map((check) => check.name));
  return registry.list().filter((definition) => registered.has(`harness_${definition.id}`)).map((definition) => definition.protocol);
}

async function serviceChecks(service, requiredProtocols) {
  const checks = [];
  const ready = await service.ready();
  checks.push({ name: "ready", status: ready.ok && ready.result?.ready === true ? "pass" : "fail", detail: ready.ok ? String(ready.result?.status) : errorText(ready.error) });
  const capabilities = await service.capabilities();
  const availableProtocols = new Set(capabilities.result?.protocols || []);
  const availableFeatures = new Set(capabilities.result?.features || []);
  const missingProtocols = requiredProtocols.filter((protocol) => !availableProtocols.has(protocol));
  const missingFeatures = REQUIRED_FEATURES.filter((feature) => !availableFeatures.has(feature));
  const compatible = capabilities.ok
    && capabilities.result?.api_version === "1.0"
    && missingProtocols.length === 0
    && missingFeatures.length === 0;
  const compatibilityDetail = compatible
    ? `service ${capabilities.result?.service_version}`
    : `api=${capabilities.result?.api_version || "unavailable"}; missing_protocols=${missingProtocols.join(",")}; missing_features=${missingFeatures.join(",")} (service is too old; upgrade habitus-server)`;
  checks.push({ name: "capabilities", status: compatible ? "pass" : "fail", detail: compatibilityDetail });
  // 提醒出口（方案 32）不是必需特性：没有就只做 memory，这里报 warn 让人知道提醒接不接得上
  checks.push({
    name: "reminders",
    status: availableFeatures.has("reminders_v1") ? "pass" : "warn",
    detail: availableFeatures.has("reminders_v1") ? "service claims reminders (reminders_v1)" : "service has no reminders_v1; memory only",
  });
  return checks;
}

// 宿主跑的是 marketplace 根里的暂存副本：npm 包升级后不 update，宿主一直跑旧插件
export async function marketplaceCheck(registry = HARNESS_REGISTRY, env = process.env) {
  try {
    const status = await marketplaceStatus(defaultMarketplaceRoot(env), { sourceRoot: PLUGINS_ROOT, registry });
    if (!status.prepared) return { name: "marketplace", status: "warn", detail: "plugins are not installed; run: habitus plugin install" };
    if (status.stale) return { name: "marketplace", status: "warn", detail: "installed plugins are older than this npm package; run: habitus plugin update" };
    return { name: "marketplace", status: "pass", detail: `prepared ${status.preparedAt}` };
  } catch (error) {
    return { name: "marketplace", status: "warn", detail: error?.message || String(error) };
  }
}

function nodeRuntimeCheck() {
  const major = Number(process.versions.node.split(".")[0]);
  return {
    name: "node_runtime",
    status: Number.isSafeInteger(major) && major >= 18 ? "pass" : "fail",
    detail: `version=${process.versions.node}; minimum=18`,
  };
}

async function bundleChecks(registry = HARNESS_REGISTRY) {
  const required = registry.list().flatMap((definition) => (
    definition.requiredAssets.map((asset) => `${definition.pluginDirectory}/${asset}`)
  ));
  try {
    await Promise.all(required.map((value) => access(join(PLUGINS_ROOT, value))));
  } catch {
    return [{ name: "bundle", status: "fail", detail: "plugin bundle is incomplete" }];
  }
  return [
    { name: "bundle", status: "pass", detail: `${required.length} required assets available` },
    await manifestCheck(registry),
    await sharedBundleCheck(registry),
  ];
}

async function manifestCheck(registry = HARNESS_REGISTRY) {
  try {
    for (const definition of registry.list()) {
      const pluginRoot = join(PLUGINS_ROOT, definition.pluginDirectory);
      const manifest = JSON.parse(await readFile(join(pluginRoot, definition.pluginManifest), "utf8"));
      if (manifest.name !== PLUGIN_NAME || typeof manifest.version !== "string" || !manifest.version) {
        throw new Error("plugin identity is invalid");
      }
      if (!definition.hooksManifest) continue;
      const hooks = JSON.parse(await readFile(join(pluginRoot, definition.hooksManifest), "utf8"));
      for (const name of definition.requiredHooks) {
        if (!Array.isArray(hooks.hooks?.[name])) {
          throw new Error(`required hook is missing for ${definition.id}: ${name}`);
        }
      }
    }
    return { name: "manifests", status: "pass", detail: "plugin identities and required hooks are valid" };
  } catch (error) {
    return { name: "manifests", status: "fail", detail: error?.message || String(error) };
  }
}

async function sharedBundleCheck(registry = HARNESS_REGISTRY) {
  try {
    const sourceRoot = join(PLUGINS_ROOT, "memory-plugin-shared", "lib");
    const files = (await readdir(sourceRoot)).filter((name) => name.endsWith(".mjs")).sort();
    const digest = createHash("sha256");
    for (const filename of files) {
      const source = await readFile(join(sourceRoot, filename), "utf8");
      digest.update(filename).update("\0").update(source).update("\0");
      for (const plugin of registry.list().map((definition) => definition.pluginDirectory)) {
        const generated = await readFile(join(PLUGINS_ROOT, plugin, "scripts", "shared", filename), "utf8");
        if (generated !== GENERATED_HEADER + source) throw new Error(`${plugin}/${filename} is stale`);
      }
    }
    return { name: "shared_bundle", status: "pass", detail: `files=${files.length}; digest=${digest.digest("hex")}` };
  } catch (error) {
    return { name: "shared_bundle", status: "fail", detail: error?.message || String(error) };
  }
}

export async function harnessFeatureChecks(registry = HARNESS_REGISTRY, run = spawnSync) {
  // 插件装上了但宿主的 hook 特性没开，记忆与提醒的 hook 一个都不会跑：按注册表里声明的特性逐项查（宿主没装 CLI 的跳过）
  const checks = [];
  for (const definition of registry.list().filter((item) => item.featureInventory)) {
    const result = run(definition.command, definition.featureInventory.args, { encoding: "utf8", timeout: 5000 });
    if (result.error?.code === "ENOENT") continue;
    if (result.error || result.status !== 0) {
      checks.push({ name: `features_${definition.id}`, status: "warn", detail: "Harness feature list is unavailable" });
      continue;
    }
    const enabled = parseFeatureList(result.stdout);
    const missing = definition.featureInventory.required.filter((name) => enabled.get(name) !== true);
    checks.push({
      name: `features_${definition.id}`,
      status: missing.length === 0 ? "pass" : "fail",
      detail: missing.length === 0
        ? `required features enabled: ${definition.featureInventory.required.join(", ")}`
        : `features off: ${missing.join(", ")} (enable them in the harness config, then restart it)`,
    });
  }
  return checks;
}

// 插件注册了、hook 却没被用户批准，宿主照样一个都不跑：只查已注册、且宿主要求逐条批准的
export async function harnessHookReviewChecks(registry = HARNESS_REGISTRY, registrations = [], review = inspectHookReview) {
  const registered = new Set(registrations.filter((check) => check.status === "pass").map((check) => check.name));
  const checks = [];
  for (const definition of registry.list().filter((item) => item.hookReview && registered.has(`harness_${item.id}`))) {
    const name = `hooks_review_${definition.id}`;
    const result = await review(definition);
    if (!result?.available) {
      checks.push({ name, status: "warn", detail: `cannot ask the harness which hooks are approved: ${result?.error || "unknown"}` });
    } else if (result.hooks.length === 0) {
      // 列不出来多半是宿主的 hook 特性关着（看上面的 features_ 那一项），特性开着才是插件没装上
      checks.push({ name, status: "fail", detail: `the harness lists no Habitus hooks, so none will run: check features_${definition.id} above; if hooks are enabled, run: habitus plugin update` });
    } else if (result.pending.length > 0) {
      checks.push({ name, status: "fail", detail: `${result.pending.length} hook(s) not approved (${result.pending.join(", ")}): ${definition.hookReview.instruction}` });
    } else {
      checks.push({ name, status: "pass", detail: `all ${result.hooks.length} hooks approved` });
    }
  }
  return checks;
}

export function parseFeatureList(text) {
  // `codex features list` 一行一项：名字、阶段（可能含空格）、true / false
  const found = new Map();
  for (const line of String(text).split("\n")) {
    const match = line.trim().match(/^([a-z0-9_]+)\s+.*\s(true|false)$/);
    if (match) found.set(match[1], match[2] === "true");
  }
  return found;
}

export async function harnessRegistrationChecks(registry = HARNESS_REGISTRY, env = process.env, runner = undefined) {
  return Promise.all(registry.list().map((definition) => inspectHarness(definition, env, runner)));
}

// 和安装器用同一组注册方式去查（不再自己解析一遍宿主清单或入口文件）；宿主 CLI 报错只记 warn，不算体检失败
async function inspectHarness(definition, env, runner = new CommandRunner()) {
  const name = `harness_${definition.id}`;
  let snapshot;
  try {
    snapshot = await registrationOf(definition).inspect(definition, runner, env);
  } catch (error) {
    return { name, status: "warn", detail: `Harness plugin inventory is unavailable: ${error?.message || error}` };
  }
  if (!snapshot.available) return { name, status: "warn", detail: "Harness is not installed" };
  if (snapshot.installed && snapshot.enabled) return { name, status: "pass", detail: "Habitus plugin is registered" };
  if (snapshot.dangling) return { name, status: "fail", detail: "plugin entry points at a missing plugin; run: habitus plugin install" };
  if (snapshot.foreignEntry) return { name, status: "warn", detail: "plugin entry exists but is not managed by Habitus" };
  return { name, status: "warn", detail: "Habitus plugin is not registered" };
}

export async function outboxInventory(root, { now = Date.now(), maxPendingAgeMs = MAX_PENDING_AGE_MS } = {}) {
  const outbox = join(root, "outbox");
  let hosts;
  try { hosts = await readdir(outbox, { withFileTypes: true }); }
  catch (error) {
    if (error?.code === "ENOENT") {
      return {
        queued: 0, bound: 0, inflight: 0, blocked: 0,
        aged: 0, expiredInflight: 0, missingState: 0, invalidState: 0,
      };
    }
    throw error;
  }
  const counts = {
    queued: 0, bound: 0, inflight: 0, blocked: 0,
    aged: 0, expiredInflight: 0, missingState: 0, invalidState: 0,
  };
  const statuses = new Set(["queued", "bound", "inflight", "blocked"]);
  for (const host of hosts.filter((entry) => entry.isDirectory())) {
    const sessions = await readdir(join(outbox, host.name), { withFileTypes: true });
    for (const session of sessions.filter((entry) => entry.isDirectory())) {
      const directory = join(outbox, host.name, session.name);
      const files = (await readdir(directory)).filter((name) => name.endsWith(".json"));
      if (files.length === 0) continue;
      let stateChecked = false;
      for (const filename of files) {
        const item = JSON.parse(await readFile(join(directory, filename), "utf8"));
        if (item?.schemaVersion !== 2 || !/^[0-9a-f]{64}$/.test(item?.deliveryId || "")) {
          throw new Error("outbox item schema is invalid");
        }
        if (!statuses.has(item.status)) throw new Error(`unsupported outbox status: ${item.status}`);
        if (sessionKey(host.name, item.nativeSessionId) !== session.name) throw new Error("outbox session key is invalid");
        if (!stateChecked) {
          try {
            const state = await readState(root, host.name, item.nativeSessionId);
            if (state == null) counts.missingState += 1;
          } catch {
            counts.invalidState += 1;
          }
          stateChecked = true;
        }
        counts[item.status] += 1;
        const createdAt = Date.parse(item.createdAt);
        if (!Number.isFinite(createdAt)) throw new Error("outbox item createdAt is invalid");
        if (now - createdAt > maxPendingAgeMs) counts.aged += 1;
        if (item.status === "inflight") {
          const expiresAt = Date.parse(item.claimExpiresAt);
          if (!Number.isFinite(expiresAt) || expiresAt <= now) counts.expiredInflight += 1;
        }
      }
    }
  }
  return counts;
}
