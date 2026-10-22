#!/usr/bin/env node
// Habitus 命令行（方案 36）：npm 装、经 HTTP 连本地 Habitus 服务；插件安装走宿主原生的 marketplace 命令。
// 只认这一个入口：插件生命周期、体检、服务状态、搜记忆都从这里分发，不再经 Python 绕一圈。

import { readFile, realpath } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { HARNESS_REGISTRY } from "../harnesses.mjs";
import { executePluginLifecycle, parseArgs as parsePluginArgs } from "../install-memory-plugin.mjs";
import { runPluginDoctor } from "../memory-plugin-shared/doctor.mjs";
import { loadPluginConfig } from "../memory-plugin-shared/lib/config.mjs";
import { memory } from "./memory.mjs";
import { reminders } from "./reminders.mjs";
import { call, connect } from "./service.mjs";

const PACKAGE_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PLUGIN_ACTIONS = new Set(["install", "update", "remove", "status"]);
const SERVER_COMMANDS = new Set(["init", "serve"]);

const HELP = `usage: habitus <command> [options]

插件（宿主原生 marketplace）：
  habitus plugin install [--harness ID ...] [--root PATH] [--prepare-only] [--json]
  habitus plugin update  [--harness ID ...]          重新注册（插件升级后用）
  habitus plugin remove  [--harness ID ...]
  habitus plugin status  [--harness ID ...]
  habitus harnesses [--json]                        支持哪些宿主、本机装了哪些 CLI

本地服务：
  habitus doctor [--json]                           服务、特性、宿主注册、宿主 hook 开关、outbox
  habitus status [--json]                           服务健康、就绪、版本与特性
  habitus recall <问题> [--limit N] [--json]         搜记忆
  habitus memory ls [地址]                           列记忆树的一个目录（省略是根）
  habitus memory tree [地址] [-L 层数]                记忆树一览（默认 3 层）
  habitus memory read <地址>                         读一条记忆，或目录的 .abstract.md / .overview.md
  habitus reminders [--day YYYY-MM-DD] [--json]     某天（默认今天）每条提醒走到哪一步：待取 / 取走了 / 说了 / 宿主没取

  habitus --version | --help
服务地址取 HABITUS_URL，或 ~/.habitus/agent-plugin/connection.json（habitus-server init 写的），默认 http://127.0.0.1:8787。
`;

export async function main(argv, { stdout = process.stdout, stderr = process.stderr, env = process.env } = {}) {
  const args = [...argv];
  const command = args.shift();
  try {
    if (!command || command === "-h" || command === "--help" || command === "help") {
      stdout.write(HELP);
      return 0;
    }
    if (command === "--version" || command === "-v" || command === "version") {
      stdout.write(`${await packageVersion()}\n`);
      return 0;
    }
    if (command === "plugin") return await plugin(args, { stdout, env });
    if (command === "harnesses") return await harnesses(args, { stdout });
    if (command === "doctor") return await doctor(args, { stdout, env });
    if (command === "status") return await status(args, { stdout, env });
    if (command === "recall") return await recall(args, { stdout, env });
    if (command === "memory") return await memory(args, { stdout, env });
    if (command === "reminders") return await reminders(args, { stdout, env });
    if (SERVER_COMMANDS.has(command)) {
      stderr.write(`habitus ${command} 是服务端命令，现在叫 habitus-server ${command}（pip 装的 Habitus 服务带的）。\n`);
      return 2;
    }
    stderr.write(`unknown command: ${command}\n\n${HELP}`);
    return 2;
  } catch (error) {
    stderr.write(`habitus ${command}: ${error?.message || error}\n`);
    return 1;
  }
}

async function plugin(args, { stdout, env }) {
  const action = args[0];
  if (args.includes("-h") || args.includes("--help")) { stdout.write(HELP); return 0; }
  if (!PLUGIN_ACTIONS.has(action)) throw new Error("usage: habitus plugin <install|update|remove|status> [options]");
  const options = parsePluginArgs(args, env);
  const result = await executePluginLifecycle(options, { env });
  if (options.json) { writeJson(stdout, result); return 0; }
  stdout.write(`Habitus 插件 ${action}：${result.root}\n`);
  for (const item of result.harnesses || []) {
    stdout.write(`  ${item.harness.padEnd(14)} ${harnessState(item)}\n`);
  }
  if (result.marketplace) {
    const { prepared, stale } = result.marketplace;
    stdout.write(`  marketplace：${!prepared ? "未准备" : stale ? "比当前 npm 包旧，运行 habitus plugin update" : "已就绪"}\n`);
  }
  return 0;
}

function harnessState(item) {
  if (item.installed && item.enabled) return "已注册";
  if (item.dangling) return "入口指向的插件已不在，运行 habitus plugin install 修复";
  if (item.foreignEntry) return "同名入口不是 Habitus 的，未注册";
  return item.available ? "未注册" : "本机没有这个宿主";
}

async function harnesses(args, { stdout }) {
  const json = consumeFlag(args, "--json");
  rejectRest(args);
  const result = await executePluginLifecycle({ action: "harnesses" });
  if (json) { writeJson(stdout, result); return 0; }
  for (const item of result.harnesses) {
    stdout.write(`${item.id.padEnd(14)} ${item.display_name.padEnd(16)} ${item.available ? "本机已装" : "本机没有"}\n`);
  }
  return 0;
}

async function doctor(args, { stdout, env }) {
  const json = consumeFlag(args, "--json");
  rejectRest(args);
  const report = await runPluginDoctor(loadPluginConfig(env), { registry: HARNESS_REGISTRY, env });
  if (json) writeJson(stdout, report);
  else {
    for (const check of report.checks) {
      const mark = check.status === "pass" ? "✓" : check.status === "warn" ? "!" : "✗";
      stdout.write(`${mark} ${check.name.padEnd(28)} ${check.detail}\n`);
    }
    stdout.write(report.ok ? "\n全部通过。\n" : "\n有检查没过（✗）。\n");
  }
  return report.ok ? 0 : 1;
}

async function status(args, { stdout, env }) {
  const json = consumeFlag(args, "--json");
  rejectRest(args);
  const { config, service } = connect(env);
  const [health, ready, capabilities] = await Promise.all([service.health(), service.ready(), service.capabilities()]);
  const report = {
    base_url: config.baseUrl,
    reachable: health.ok,
    status: health.ok ? health.result?.status ?? null : null,
    ready: ready.ok ? ready.result?.ready === true : false,
    service_version: capabilities.ok ? capabilities.result?.service_version ?? null : null,
    api_version: capabilities.ok ? capabilities.result?.api_version ?? null : null,
    features: capabilities.ok && Array.isArray(capabilities.result?.features) ? capabilities.result.features : [],
    checks: ready.ok && Array.isArray(ready.result?.checks) ? ready.result.checks : [],
    error: health.ok ? null : String(health.error?.message || health.error || "unreachable"),
  };
  if (json) writeJson(stdout, report);
  else if (!report.reachable) stdout.write(`连不上 Habitus 服务（${report.base_url}）：${report.error}\n先启动：habitus-server\n`);
  else {
    stdout.write(`服务 ${report.base_url}  ${report.status}  ${report.ready ? "就绪" : "未就绪"}\n`);
    stdout.write(`版本 ${report.service_version}（API ${report.api_version}）\n`);
    stdout.write(`特性 ${report.features.join(", ")}\n`);
    for (const check of report.checks) {
      if (check.status !== "healthy") stdout.write(`  ! ${check.name}: ${check.status} ${check.detail}\n`);
    }
  }
  return report.reachable && report.ready ? 0 : 1;
}

async function recall(args, { stdout, env }) {
  const json = consumeFlag(args, "--json");
  const limit = consumeValue(args, "--limit");
  const query = args.join(" ").trim();
  if (!query) throw new Error("usage: habitus recall <问题> [--limit N] [--json]");
  if (limit != null && !(Number.isSafeInteger(Number(limit)) && Number(limit) > 0)) throw new Error("--limit must be a positive integer");
  const found = await call(connect(env), "recall", "/api/v1/memory/recall", {
    method: "POST",
    body: JSON.stringify(limit == null ? { query } : { query, limit: Number(limit) }),
  });
  if (json) { writeJson(stdout, found); return 0; }
  const result = found || {};
  if (!result.memories?.length && !result.context) { stdout.write("没有相关记忆。\n"); return 0; }
  for (const item of result.memories || []) stdout.write(`${Number(item.score).toFixed(2)}  ${item.uri}\n`);
  if (result.context) stdout.write(`\n${result.context}\n`);
  return 0;
}

function consumeFlag(args, name) {
  const index = args.indexOf(name);
  if (index < 0) return false;
  args.splice(index, 1);
  return true;
}

function consumeValue(args, name) {
  const index = args.indexOf(name);
  if (index < 0) return null;
  const value = args[index + 1];
  if (value == null) throw new Error(`${name} requires a value`);
  args.splice(index, 2);
  return value;
}

function rejectRest(args) {
  if (args.length > 0) throw new Error(`unknown argument: ${args[0]}`);
}

function writeJson(stdout, value) {
  stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

async function packageVersion() {
  const manifest = JSON.parse(await readFile(join(PACKAGE_ROOT, "package.json"), "utf8"));
  return manifest.version;
}

async function isMainModule() {
  if (!process.argv[1]) return false;
  try { return await realpath(fileURLToPath(import.meta.url)) === await realpath(process.argv[1]); }
  catch { return false; }
}

if (await isMainModule()) process.exitCode = await main(process.argv.slice(2));
