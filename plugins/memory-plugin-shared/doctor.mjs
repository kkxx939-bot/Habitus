import { createHash } from "node:crypto";
import { access, readdir, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  HARNESS_REGISTRY,
  PLUGIN_NAME,
} from "../harnesses.mjs";
import { defaultMarketplaceRoot, marketplaceStatus } from "../install-memory-plugin.mjs";
import { runHostCommand } from "../host-command.mjs";
import { CommandRunner, inspectHookReview, registrationOf } from "../registrations.mjs";
import { REMINDERS_FEATURE, hasFeature, serviceGaps } from "./lib/compat.mjs";
import { loadPluginConfig } from "./lib/config.mjs";
import { errorText } from "./lib/service-client.mjs";
import { HabitusServiceClient } from "./lib/service-client.mjs";
import { inventoryOutbox } from "./lib/outbox.mjs";
import { listCaptureFailures, listSkippedTranscriptLines, readState } from "./lib/state-store.mjs";

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
  checks.push(...await inspectFeatures(registry, registrations, runHostCommand, env));
  checks.push(...await inspectHookReviews(registry, registrations, inspectHookReview, { timeoutMs: config.timeoutMs, env }));
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
      detail: [
        ...Object.entries(inventory).map(([status, count]) => `${status}_items=${count}`),
        ...(await blockedOutboxItems(config.stateRoot)),
      ].join("; "),
    });
  } catch (error) {
    checks.push({ name: "outbox", status: "fail", detail: error?.message || String(error) });
  }
  checks.push(await captureCheck(config.stateRoot));
  return { ok: checks.every((check) => check.status !== "fail"), checks };
}

// 只有本机已注册的宿主，它的会话协议缺了才算失败；别的宿主缺协议只提示（新 CLI 连老服务时只用 Codex 的人照常能用）
function registeredProtocols(registry, registrations) {
  const registered = new Set(registrations.filter((check) => check.status === "pass").map((check) => check.name));
  return registry.list().filter((definition) => registered.has(`harness_${definition.id}`)).map((definition) => definition.protocol);
}

async function serviceChecks(service, requiredProtocols) {
  const checks = [];
  checks.push(readinessCheck(await service.ready()));
  const capabilities = await service.capabilities();
  const gaps = serviceGaps(capabilities.result, requiredProtocols);
  const compatible = capabilities.ok
    && gaps.apiVersion
    && gaps.missingProtocols.length === 0
    && gaps.missingFeatures.length === 0;
  const compatibilityDetail = compatible
    ? `service ${capabilities.result?.service_version}`
    : `api=${capabilities.result?.api_version || "unavailable"}; missing_protocols=${gaps.missingProtocols.join(",")}; missing_features=${gaps.missingFeatures.join(",")} (service is too old; upgrade habitus-server)`;
  checks.push({ name: "capabilities", status: compatible ? "pass" : "fail", detail: compatibilityDetail });
  // 提醒出口（方案 32）不是必需特性：没有就只做 memory，这里报 warn 让人知道提醒接不接得上
  const reminders = hasFeature(capabilities.result, REMINDERS_FEATURE);
  checks.push({
    name: "reminders",
    status: reminders ? "pass" : "warn",
    // 服务只在预测层开着时公布 reminders_v1（裁定 80 第 1 条）：没有就是提醒没开，说清怎么开
    detail: reminders
      ? `service claims reminders (${REMINDERS_FEATURE})`
      : `reminders are off: the service's prediction layer is not enabled (no ${REMINDERS_FEATURE}); memory only. `
        + "To turn reminders on, set behavior.primary_subject, scene.enabled: true and foresight.enabled: true "
        + "in config.yaml (commented examples in habitus/config/example.yaml), then restart habitus-server",
  });
  return checks;
}

// 未就绪时服务照样给出各项检查（/ready 的 503 带 HealthResult）：报出哪几项不健康，而不是只说一个总状态。
// 记忆写入在等模型恢复（memory_queue 报 waiting_for_model：额度 / 余额、鉴权、断网）不是失败——Job 排着队按长退避自己重试，
// 召回照常（recall_ready）——报 warn 并说该看哪里；Job 真失败待处置、召回本身不可用报 fail
function readinessCheck(ready) {
  if (!ready.ok) return { name: "ready", status: "fail", detail: errorText(ready.error) };
  const checks = Array.isArray(ready.result?.checks) ? ready.result.checks : [];
  const unhealthy = checks
    .filter((check) => check?.status !== "healthy")
    .map((check) => `${check.name}: ${check.status} ${check.detail}`);
  const waiting = checks.find((check) => check?.name === "memory_queue" && String(check?.detail).startsWith("waiting_for_model:"));
  const otherCritical = checks.filter((check) => check?.critical !== false && check?.status !== "healthy" && check !== waiting);
  const detail = [String(ready.result?.status), ...unhealthy];
  if (ready.result?.ready === true) return { name: "ready", status: "pass", detail: detail.join("; ") };
  if (waiting && otherCritical.length === 0 && ready.result?.recall_ready === true) {
    detail.push("memory writes are waiting for the chat model to recover (quota or balance, credentials, network); "
      + "queued jobs retry on their own and recall still works; check the model credentials and account");
    return { name: "ready", status: "warn", detail: detail.join("; ") };
  }
  if (ready.result?.recall_ready === true) detail.push("recall still works; memory writes are stopped (see memory_queue / memory_worker above)");
  return { name: "ready", status: "fail", detail: detail.join("; ") };
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
    await Promise.all(required.map((value) => access(resolve(PLUGINS_ROOT, value))));
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
      const pluginRoot = resolve(PLUGINS_ROOT, definition.pluginDirectory);
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
        const generated = await readFile(join(resolve(PLUGINS_ROOT, plugin), "scripts", "shared", filename), "utf8");
        if (generated !== GENERATED_HEADER + source) throw new Error(`${plugin}/${filename} is stale`);
      }
    }
    return { name: "shared_bundle", status: "pass", detail: `files=${files.length}; digest=${digest.digest("hex")}` };
  } catch (error) {
    return { name: "shared_bundle", status: "fail", detail: error?.message || String(error) };
  }
}

// 宿主的特性清单对照注册表：必需的缺了 hook 一个都不跑；提醒相关的（advisory）缺了只是提醒问不出去。宿主没装 CLI 给 null。
// 到点直接 SIGKILL：宿主接管了 SIGTERM 时，spawnSync 默认的超时信号兜不住
export function harnessFeatureGaps(definition, run = runHostCommand, env = process.env) {
  const result = run(definition.command, definition.featureInventory.args, { encoding: "utf8", timeout: 5000, killSignal: "SIGKILL", env });
  if (result.error?.code === "ENOENT") return null;
  if (result.error || result.status !== 0) return { available: false };
  const enabled = parseFeatureList(result.stdout);
  return {
    available: true,
    missing: definition.featureInventory.required.filter((name) => enabled.get(name) !== true),
    advisory: definition.featureInventory.advisory.filter((item) => enabled.get(item.name) !== true),
  };
}

export async function harnessFeatureChecks(registry = HARNESS_REGISTRY, registrations = [], run = runHostCommand, env = process.env) {
  // 插件装上了但宿主的 hook 特性没开，记忆与提醒的 hook 一个都不会跑：按注册表里声明的特性逐项查（宿主没装 CLI 的跳过）。
  // 提醒相关的特性只对已注册的宿主查，缺了报 warn
  const registered = new Set(registrations.filter((check) => check.status === "pass").map((check) => check.name));
  const checks = [];
  for (const definition of registry.list().filter((item) => item.featureInventory)) {
    const gaps = harnessFeatureGaps(definition, run, env);
    if (gaps == null) continue;
    if (!gaps.available) {
      checks.push({ name: `features_${definition.id}`, status: "warn", detail: "Harness feature list is unavailable" });
      continue;
    }
    const advisory = registered.has(`harness_${definition.id}`) ? gaps.advisory : [];
    const detail = gaps.missing.length === 0
      ? `required features enabled: ${definition.featureInventory.required.join(", ")}`
      : `features off: ${gaps.missing.join(", ")} (enable them in the harness config, then restart it)`;
    checks.push({
      name: `features_${definition.id}`,
      status: gaps.missing.length > 0 ? "fail" : advisory.length > 0 ? "warn" : "pass",
      detail: [detail, ...advisory.map((item) => `${item.name} off: ${item.note}`)].join("; "),
    });
  }
  return checks;
}

// 插件注册了、hook 却没被用户批准，宿主照样一个都不跑：只查已注册、且宿主要求逐条批准的。
// options（超时 timeoutMs、环境 env）原样交给 review
export async function harnessHookReviewChecks(registry = HARNESS_REGISTRY, registrations = [], review = inspectHookReview, options = {}) {
  const registered = new Set(registrations.filter((check) => check.status === "pass").map((check) => check.name));
  const checks = [];
  for (const definition of registry.list().filter((item) => item.hookReview && registered.has(`harness_${item.id}`))) {
    const name = `hooks_review_${definition.id}`;
    const result = await review(definition, options);
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
async function inspectHarness(definition, env, runner = new CommandRunner(env)) {
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

// 转写读不出来的会话（没进 outbox，outbox 那一项看不见）：每次 Stop 都照样失败、这个会话停采，直到读成功一次（fail）。
// 原样超过读取额度、跳过没读的行：采集照常往下走，那一行的内容没进记忆（warn，列出是哪个会话的第几行、多大）
export async function captureCheck(root) {
  try {
    const failures = await listCaptureFailures(root);
    if (failures.length > 0) {
      return {
        name: "capture",
        status: "fail",
        detail: failures.map((item) => `${item.host} session ${item.nativeSessionId}: transcript not read since ${item.since} (last tried ${item.at}): ${item.reason}`).join("; "),
      };
    }
    const skipped = await listSkippedTranscriptLines(root);
    if (skipped.length > 0) {
      return {
        name: "capture",
        status: "warn",
        detail: skipped.map((item) => `${item.host} session ${item.nativeSessionId}: transcript line ${item.lineCount} (${item.bytes} bytes) skipped at ${item.at}: over the read limit (HABITUS_PLUGIN_MAX_TRANSCRIPT_BYTES)`).join("; "),
      };
    }
    return { name: "capture", status: "pass", detail: "no session is stuck reading its transcript" };
  } catch (error) {
    return { name: "capture", status: "fail", detail: error?.message || String(error) };
  }
}

// 卡住的每一条：哪个宿主、哪个会话、为什么（超过服务请求上限的写着这一批多大、超了多少）
export async function blockedOutboxItems(root) {
  const lines = [];
  for (const session of await inventoryOutbox(root)) {
    for (const item of session.items.filter((value) => value.status === "blocked")) {
      lines.push(`blocked ${session.host} session ${session.nativeSessionId}: ${item.blockedReason || "non-retryable service error"}`);
    }
  }
  return lines;
}

// 列 outbox 用 outbox.mjs 那一份（与 hook worker 并发时读到一半消失的会话、条目跳过），这里只数状态
export async function outboxInventory(root, { now = Date.now(), maxPendingAgeMs = MAX_PENDING_AGE_MS } = {}) {
  const counts = {
    queued: 0, bound: 0, inflight: 0, blocked: 0,
    aged: 0, expiredInflight: 0, missingState: 0, invalidState: 0,
  };
  const statuses = new Set(["queued", "bound", "inflight", "blocked"]);
  for (const session of await inventoryOutbox(root)) {
    try {
      if (await readState(root, session.host, session.nativeSessionId) == null) counts.missingState += 1;
    } catch {
      counts.invalidState += 1;
    }
    for (const item of session.items) {
      if (!statuses.has(item.status)) throw new Error(`unsupported outbox status: ${item.status}`);
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
  return counts;
}
