#!/usr/bin/env node

import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { access, readdir, readFile, realpath } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPluginConfig } from "./lib/config.mjs";
import { M2BOSServiceClient } from "./lib/service-client.mjs";
import { readState, sessionKey } from "./lib/state-store.mjs";

const REQUIRED_FEATURES = ["conversation_cursor", "flush", "recall", "remember", "remember_idempotency_v1"];
const REQUIRED_PROTOCOLS = ["claude_code", "codex_rollout"];
const PLUGINS_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const GENERATED_HEADER = "// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.\n";
const MAX_PENDING_AGE_MS = 24 * 60 * 60 * 1000;

export async function runPluginDoctor(
  config = loadPluginConfig(),
  {
    service = new M2BOSServiceClient(config),
    protocols = REQUIRED_PROTOCOLS,
    inspectHosts = hostRegistrationChecks,
  } = {},
) {
  const checks = [];
  checks.push(nodeRuntimeCheck(), ...(await bundleChecks()));
  checks.push(...await inspectHosts());
  const health = await service.health();
  checks.push({ name: "health", status: health.ok ? "pass" : "fail", detail: health.ok ? "service reachable" : String(health.error) });
  const ready = await service.ready();
  checks.push({ name: "ready", status: ready.ok && ready.result?.ready === true ? "pass" : "fail", detail: ready.ok ? String(ready.result?.status) : String(ready.error) });
  const capabilities = await service.capabilities();
  const availableProtocols = new Set(capabilities.result?.protocols || []);
  const availableFeatures = new Set(capabilities.result?.features || []);
  const missingProtocols = protocols.filter((protocol) => !availableProtocols.has(protocol));
  const missingFeatures = REQUIRED_FEATURES.filter((feature) => !availableFeatures.has(feature));
  const compatible = capabilities.ok
    && capabilities.result?.api_version === "1.0"
    && missingProtocols.length === 0
    && missingFeatures.length === 0;
  const compatibilityDetail = compatible
    ? `service ${capabilities.result?.service_version}`
    : `api=${capabilities.result?.api_version || "unavailable"}; missing_protocols=${missingProtocols.join(",")}; missing_features=${missingFeatures.join(",")}`;
  checks.push({ name: "capabilities", status: compatible ? "pass" : "fail", detail: compatibilityDetail });
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

function nodeRuntimeCheck() {
  const major = Number(process.versions.node.split(".")[0]);
  return {
    name: "node_runtime",
    status: Number.isSafeInteger(major) && major >= 18 ? "pass" : "fail",
    detail: `version=${process.versions.node}; minimum=18`,
  };
}

async function bundleChecks() {
  const required = [
    "m2bos-memory/.codex-plugin/plugin.json",
    "m2bos-memory/hooks/hooks.json",
    "m2bos-memory/scripts/shared/plugin-core.mjs",
    "m2bos-memory-claude-code/.claude-plugin/plugin.json",
    "m2bos-memory-claude-code/hooks/hooks.json",
    "m2bos-memory-claude-code/scripts/shared/plugin-core.mjs",
    "m2bos-memory-claude-code/scripts/subagent-start.mjs",
    "m2bos-memory-claude-code/scripts/subagent-stop.mjs",
  ];
  try {
    await Promise.all(required.map((value) => access(join(PLUGINS_ROOT, value))));
  } catch {
    return [{ name: "bundle", status: "fail", detail: "plugin bundle is incomplete" }];
  }
  return [
    { name: "bundle", status: "pass", detail: `${required.length} required assets available` },
    await manifestCheck(),
    await sharedBundleCheck(),
  ];
}

async function manifestCheck() {
  try {
    const codex = JSON.parse(await readFile(join(PLUGINS_ROOT, "m2bos-memory", ".codex-plugin", "plugin.json"), "utf8"));
    const claude = JSON.parse(await readFile(join(PLUGINS_ROOT, "m2bos-memory-claude-code", ".claude-plugin", "plugin.json"), "utf8"));
    const codexHooks = JSON.parse(await readFile(join(PLUGINS_ROOT, "m2bos-memory", "hooks", "hooks.json"), "utf8"));
    const claudeHooks = JSON.parse(await readFile(join(PLUGINS_ROOT, "m2bos-memory-claude-code", "hooks", "hooks.json"), "utf8"));
    for (const manifest of [codex, claude]) {
      if (manifest.name !== "m2bos-memory" || typeof manifest.version !== "string" || !manifest.version) {
        throw new Error("plugin identity is invalid");
      }
    }
    for (const name of ["SessionStart", "UserPromptSubmit", "Stop", "PreCompact", "SessionEnd"]) {
      if (!Array.isArray(codexHooks.hooks?.[name]) || !Array.isArray(claudeHooks.hooks?.[name])) {
        throw new Error(`required hook is missing: ${name}`);
      }
    }
    for (const name of ["SubagentStart", "SubagentStop"]) {
      if (!Array.isArray(claudeHooks.hooks?.[name])) throw new Error(`required Claude hook is missing: ${name}`);
    }
    return { name: "manifests", status: "pass", detail: "plugin identities and required hooks are valid" };
  } catch (error) {
    return { name: "manifests", status: "fail", detail: error?.message || String(error) };
  }
}

async function sharedBundleCheck() {
  try {
    const sourceRoot = join(PLUGINS_ROOT, "memory-plugin-shared", "lib");
    const files = (await readdir(sourceRoot)).filter((name) => name.endsWith(".mjs")).sort();
    const digest = createHash("sha256");
    for (const filename of files) {
      const source = await readFile(join(sourceRoot, filename), "utf8");
      digest.update(filename).update("\0").update(source).update("\0");
      for (const plugin of ["m2bos-memory", "m2bos-memory-claude-code"]) {
        const generated = await readFile(join(PLUGINS_ROOT, plugin, "scripts", "shared", filename), "utf8");
        if (generated !== GENERATED_HEADER + source) throw new Error(`${plugin}/${filename} is stale`);
      }
    }
    return { name: "shared_bundle", status: "pass", detail: `files=${files.length}; digest=${digest.digest("hex")}` };
  } catch (error) {
    return { name: "shared_bundle", status: "fail", detail: error?.message || String(error) };
  }
}

export async function hostRegistrationChecks() {
  return [
    inspectHost("codex", ["plugin", "list", "--json", "--available"]),
    inspectHost("claude", ["plugin", "list", "--json"]),
  ];
}

function inspectHost(command, args) {
  const result = spawnSync(command, args, { encoding: "utf8", timeout: 5000, maxBuffer: 8 * 1024 * 1024 });
  if (result.error?.code === "ENOENT") {
    return { name: `host_${command}`, status: "warn", detail: "host CLI is not installed" };
  }
  if (result.error || result.status !== 0) {
    return { name: `host_${command}`, status: "warn", detail: "host plugin inventory is unavailable" };
  }
  try {
    const inventory = JSON.parse(result.stdout);
    const installed = Array.isArray(inventory?.installed)
      ? inventory.installed
      : Array.isArray(inventory?.plugins)
        ? inventory.plugins
        : Array.isArray(inventory)
          ? inventory
          : [];
    const plugin = installed.find((entry) => (
      entry?.pluginId === "m2bos-memory@m2bos-local"
      || (entry?.name === "m2bos-memory" && (entry?.marketplaceName === "m2bos-local" || entry?.marketplace === "m2bos-local"))
    ));
    const registered = Boolean(plugin && plugin.installed !== false && plugin.enabled !== false);
    return {
      name: `host_${command}`,
      status: registered ? "pass" : "warn",
      detail: registered ? "m2bOS plugin is registered" : "m2bOS plugin is not registered",
    };
  } catch {
    return { name: `host_${command}`, status: "warn", detail: "host plugin inventory returned invalid JSON" };
  }
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

async function isMainModule() {
  if (!process.argv[1]) return false;
  try {
    return await realpath(fileURLToPath(import.meta.url)) === await realpath(process.argv[1]);
  } catch {
    return false;
  }
}

if (await isMainModule()) {
  let report;
  try { report = await runPluginDoctor(); }
  catch (error) {
    report = { ok: false, checks: [{ name: "config", status: "fail", detail: error?.message || String(error) }] };
  }
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!report.ok) process.exitCode = 1;
}
