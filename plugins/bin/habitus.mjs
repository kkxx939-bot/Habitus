#!/usr/bin/env node
// Habitus 命令行（方案 36）：npm 装、经 HTTP 连本地 Habitus 服务；插件安装走宿主原生的 marketplace 命令。
// 只认这一个入口：插件生命周期、体检、服务状态、搜记忆都从这里分发，不再经 Python 绕一圈。

import { readFile, realpath } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { HARNESS_REGISTRY } from "../harnesses.mjs";
import { runHostCommand } from "../host-command.mjs";
import { executePluginLifecycle, parseArgs as parsePluginArgs } from "../install-memory-plugin.mjs";
import { inspectHookReview } from "../registrations.mjs";
import { harnessFeatureGaps, runPluginDoctor } from "../memory-plugin-shared/doctor.mjs";
import { loadPluginConfig, pluginTimeoutMs } from "../memory-plugin-shared/lib/config.mjs";
import { memory } from "./memory.mjs";
import { reminders } from "./reminders.mjs";
import { call, connect } from "./service.mjs";

const PACKAGE_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PLUGIN_ACTIONS = new Set(["install", "update", "remove", "status"]);
const SERVER_COMMANDS = new Set(["init", "serve"]);

const HELP = `usage: habitus <command> [options]

插件（宿主原生 marketplace）：
  habitus plugin install [--harness ID ...] [--prepare-only] [--json]
  habitus plugin update  [--harness ID ...] [--prepare-only]
                                                    重新暂存并重新注册已注册的宿主（插件升级后用）；--prepare-only 只暂存
  habitus plugin remove  [--harness ID ...]
  habitus plugin status  [--harness ID ...]
  habitus harnesses [--json]                        支持哪些宿主、本机有哪些

本地服务：
  habitus doctor [--json]                           服务、特性、宿主注册、宿主 hook 开关、outbox
  habitus status [--json]                           服务健康、就绪、版本与特性
  habitus recall <问题> [--limit N] [--json]         搜记忆
  habitus memory ls [地址]                           列记忆树的一个目录（省略是根）
  habitus memory tree [地址] [-L 层数]                记忆树一览（默认 3 层）
  habitus memory read <地址>                         读一条记忆，或目录的 .abstract.md / .overview.md
  habitus reminders [--day YYYY-MM-DD] [--json]     某天（默认今天）每条提醒走到哪一步：待取 / 取走了 / 说了 / 宿主没取 / 已发生

  habitus --version | --help
  出错时加 --verbose（放在哪都行）看调用栈与底层原因。
服务地址取 HABITUS_URL，或 ~/.habitus/agent-plugin/connection.json（habitus-server init 写的），默认 http://127.0.0.1:8787。
`;

// 子命令自己的 --help：只印用法，不连服务、不碰宿主
const USAGES = Object.freeze({
  harnesses: "usage: habitus harnesses [--json]",
  doctor: "usage: habitus doctor [--json]",
  status: "usage: habitus status [--json]",
  recall: "usage: habitus recall <问题> [--limit N] [--json]",
});

function wantsHelp(args) {
  return args.includes("-h") || args.includes("--help");
}

// lifecycle / reviewHooks / featureGaps 只为测试可替换（默认就是真的安装器与宿主查询）
export async function main(
  argv,
  {
    stdout = process.stdout, stderr = process.stderr, env = process.env,
    lifecycle = executePluginLifecycle, reviewHooks = inspectHookReview, featureGaps = harnessFeatureGaps,
  } = {},
) {
  // --verbose 是全局开关：出错时多印调用栈（W5）；从参数里摘掉，子命令看不到它
  const verbose = argv.includes("--verbose");
  const args = argv.filter((value) => value !== "--verbose");
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
    if (Object.hasOwn(USAGES, command) && wantsHelp(args)) {
      stdout.write(`${USAGES[command]}\n`);
      return 0;
    }
    if (command === "plugin") return await plugin(args, { stdout, env, lifecycle, reviewHooks, featureGaps });
    if (command === "harnesses") return await harnesses(args, { stdout, env });
    if (command === "doctor") return await doctor(args, { stdout, env });
    if (command === "status") return await status(args, { stdout, env });
    if (command === "recall") return await recall(args, { stdout, env });
    if (command === "memory") return await memory(args, { stdout, env });
    if (command === "reminders") return await reminders(args, { stdout, env });
    if (SERVER_COMMANDS.has(command)) {
      stderr.write(`habitus ${command} 是本地服务的命令，现在叫 habitus-server ${command}（pip 装的 Habitus 服务带的）。\n`);
      return 2;
    }
    stderr.write(`unknown command: ${command}\n\n${HELP}`);
    return 2;
  } catch (error) {
    stderr.write(formatError(command, error, verbose));
    return 1;
  }
}

// 报错写清卡在哪（W5）：哪个宿主、哪一步（安装器记在 error.habitusStep），底层原因链，回滚没恢复成的宿主；
// --verbose 时再附调用栈
export function formatError(command, error, verbose = false) {
  const step = error?.habitusStep;
  const where = step ? `[${[step.harness, step.action].filter(Boolean).join(" · ")}] ` : "";
  const lines = [`habitus ${command}: ${where}${error?.message || error}`];
  const seen = new Set([error]);
  for (let cause = error?.cause; cause && !seen.has(cause); cause = cause.cause) {
    seen.add(cause);
    lines.push(`  原因：${cause?.message || cause}`);
  }
  for (const failure of error?.restoreFailures || []) {
    lines.push(`  回滚时 ${failure.harness} 的注册没能恢复：${failure.message}（运行 habitus plugin status 查看，必要时手动修）`);
  }
  const rollback = error?.marketplaceRollbackFailure;
  if (rollback) {
    lines.push(`  回滚时插件副本 ${rollback.root} 没能恢复：${rollback.message}（修好权限后重新运行 habitus plugin install）`);
  }
  if (verbose) {
    for (const item of seen) if (item?.stack) lines.push("", item.stack);
  } else if (step || error?.cause || error?.restoreFailures || rollback) {
    lines.push("  （加 --verbose 看调用栈）");
  }
  return `${lines.join("\n")}\n`;
}

async function plugin(args, { stdout, env, lifecycle, reviewHooks, featureGaps }) {
  const action = args[0];
  if (args.includes("-h") || args.includes("--help")) { stdout.write(HELP); return 0; }
  if (!PLUGIN_ACTIONS.has(action)) throw new Error("usage: habitus plugin <install|update|remove|status> [options]");
  const options = parsePluginArgs(args, env);
  const result = await lifecycle(options, { env });
  const reviewOptions = { timeoutMs: pluginTimeoutMs(env), env };
  // 宿主要求逐条批准 hook 的（Codex）：装好、更新后与查状态时顺手问一句批没批，没批就说怎么批
  if (["install", "update", "status"].includes(action) && !options.prepareOnly) {
    for (const item of result.harnesses || []) {
      const definition = HARNESS_REGISTRY.resolve(item.harness);
      if (item.installed && item.enabled && definition.hookReview) item.hookReview = await reviewHooks(definition, reviewOptions);
    }
  }
  // 查状态时，已注册宿主的提醒相关特性（注册表 featureInventory.advisory）没开的，在那一行说一句（与 doctor 的 warn 同一句）
  if (action === "status") {
    for (const item of result.harnesses || []) {
      const definition = HARNESS_REGISTRY.resolve(item.harness);
      if (!item.installed || !item.enabled || !definition.featureInventory?.advisory.length) continue;
      const gaps = featureGaps(definition, runHostCommand, env);
      if (gaps?.available) item.featureNotes = gaps.advisory.map((feature) => feature.note);
    }
  }
  if (options.json) { writeJson(stdout, result); return 0; }
  stdout.write(`Habitus 插件 ${action}：${result.root}\n`);
  for (const item of result.harnesses || []) {
    stdout.write(`  ${item.harness.padEnd(14)} ${harnessState(item)}${hookReviewState(item)}${(item.featureNotes || []).map((note) => `；${note}`).join("")}\n`);
  }
  if (result.marketplace) {
    const { prepared, stale } = result.marketplace;
    stdout.write(`  marketplace：${!prepared ? "未准备" : stale ? "比当前 npm 包旧，运行 habitus plugin update" : "已就绪"}\n`);
  }
  return 0;
}

export function hookReviewState(item) {
  const review = item.hookReview;
  if (!review) return "";
  if (!review.available) return "；hook 批没批查不到";
  if (review.hooks.length === 0) return "；宿主没列出 Habitus 的 hook，不会运行（运行 habitus doctor 查原因）";
  if (review.pending.length === 0) return "；hook 已批准";
  return `；${review.pending.length} 个 hook 还没批准，不会运行：${HARNESS_REGISTRY.resolve(item.harness).hookReview.instruction}`;
}

function harnessState(item) {
  if (item.installed && item.enabled) return "已注册";
  if (item.skipped) return `跳过：${item.skipped}`;
  if (item.dangling) return "入口指向的插件已不在，运行 habitus plugin install 修复";
  if (item.foreignEntry) return "同名入口不是 Habitus 的，未注册";
  return item.available ? "未注册" : "本机没有这个宿主";
}

async function harnesses(args, { stdout, env }) {
  const json = consumeFlag(args, "--json");
  rejectRest(args);
  const result = await executePluginLifecycle({ action: "harnesses" }, { env });
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
  if (!query) throw new Error(USAGES.recall);
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
