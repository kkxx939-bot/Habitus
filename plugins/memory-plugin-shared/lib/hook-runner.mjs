import { createHash, randomBytes } from "node:crypto";
import { spawn } from "node:child_process";
import { debugEnabled, loadPluginConfig } from "./config.mjs";
import { INJECTION_TAGS } from "./injection-receipt.mjs";
import { writeOperationLog, writeRepeatedFailureLogs } from "./operation-log.mjs";
import { PluginCore } from "./plugin-core.mjs";

const [MEMORY_TAG, REMINDER_TAG] = INJECTION_TAGS;

async function readInput() {
  const chunks = [];
  let bytes = 0;
  for await (const chunk of process.stdin) {
    bytes += chunk.length;
    if (bytes > 1024 * 1024) throw new Error("hook input exceeds 1 MiB");
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString();
  if (!raw.trim()) return { raw: "{}", input: {} };
  return { raw, input: JSON.parse(raw) };
}

function detach(raw, config, adapter, input, action) {
  try {
    const child = spawn(process.execPath, [process.argv[1]], {
      detached: true,
      stdio: ["pipe", "ignore", "ignore"],
      env: { ...process.env, HABITUS_HOOK_WORKER: "1" },
    });
    child.on("error", (error) => {
      void writeOperationLog(config, adapter, input, {
        hook: action, stage: "worker_spawn", status: "error", retryable: true, error,
      });
    });
    child.stdin.on("error", (error) => {
      void writeOperationLog(config, adapter, input, {
        hook: action, stage: "worker_stdin", status: "error", retryable: true, error,
      });
    });
    child.stdin.end(raw);
    child.unref();
  } catch (error) {
    void writeOperationLog(config, adapter, input, {
      hook: action, stage: "worker_spawn", status: "error", retryable: true, error,
    });
  }
}

// 出错时的堆栈（config.debug，即 HABITUS_PLUGIN_DEBUG）：命令 hook 的前台与 OpenCode / pi 进程内插件都走这一处；
// 后台 worker 起的时候 stdout / stderr 都接到 ignore，打不出来，只有操作日志
export function writeDebugStack(debug, error) {
  if (debug) process.stderr.write(`[habitus-memory] ${error?.stack || error}\n`);
}

function write(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

export function encodeContextPayload(value) {
  return JSON.stringify({
    format: "habitus_memory_context_v1",
    content: String(value),
  }).replaceAll("<", "\\u003c").replaceAll(">", "\\u003e");
}

export function createContextInjection(value, nonce = randomBytes(16).toString("hex"), tag = MEMORY_TAG) {
  if (!/^[0-9a-f]{32}$/.test(nonce)) throw new Error("injection nonce is invalid");
  if (!INJECTION_TAGS.includes(tag)) throw new Error("injection tag is unknown");
  const injection = `<${tag} receipt="${nonce}">\n${encodeContextPayload(value)}\n</${tag}>`;
  return Object.freeze({
    injection,
    receipt: Object.freeze({ nonce, digest: createHash("sha256").update(injection).digest("hex") }),
  });
}

// 给 agent 的指令块（方案 32 第 5.1 条；措辞要对着真实宿主调，R1 / R8）：先问再做；题目原样、给出服务端带来的固定选项
// （现在就做 / 这次不用 / 知道了 / 已经做过了，裁定 91 第 3 条②；个数不写死）、允许他自己填；
// 他怎么答都不用另外处理——系统从转写里认提问与作答。
export function renderReminderInstruction(reminders, questionTool) {
  const tool = questionTool || "向用户提问的工具";
  const lines = [
    `Habitus 的提醒，不是用户说的话。在做任何别的事之前，先用 ${tool} 问用户下面${reminders.length > 1 ? `这 ${reminders.length} 件事（一件一问，按顺序）` : "这件事"}：`,
    "题目原样用这句话、给下面列出的全部选项、允许他自己填别的话。问完照常做他让你做的事；他怎么答你都不用另外处理，系统自己会记。",
  ];
  reminders.forEach((item, index) => {
    lines.push(`${index + 1}. 『${String(item.text).trim()}』`);
    lines.push(`   选项：${item.options.join(" / ")}`);
  });
  return lines.join("\n");
}

// 他说了一句话：召回记忆、取提醒，拼成要注入的块（没有就是空串）。命令 hook 与进程内插件（OpenCode）共用这一份：
// 取提醒出错只丢提醒、记一笔日志，召回的记忆照样注入；召回出错由调用方兜（那时什么都不注入）。
// 召回、取提醒没成而没抛异常的（连不上、不兼容、未就绪、预算用完、非 2xx）也各记一条，接连同一原因只记第一次（operation-log.mjs）。
export function isSubagent(adapter, input) {
  return typeof adapter.isSubagent === "function" && adapter.isSubagent(input) === true;
}

// 非交互会话（脚本 / SDK 调宿主：Claude Code `claude -p`、Codex `codex exec`）不是他在说话：不召回、不领提醒、不送转写、
// 不留会话状态（裁定 80 第 6 条）。宿主认得出时由适配器说
export function isNonInteractive(adapter, input) {
  return typeof adapter.isNonInteractive === "function" && adapter.isNonInteractive(input) === true;
}

// 这个会话整个不进记忆：子代理，或非交互会话
export function skipsSession(adapter, input) {
  return isSubagent(adapter, input) || isNonInteractive(adapter, input);
}

export async function promptInjection(core, adapter, input, { config, hook }) {
  // 子代理的"一句话"是主 agent 写的任务、非交互会话的是脚本写的，都不是他说的话（宿主认得出时由适配器说）
  if (skipsSession(adapter, input)) return "";
  // 召回与取提醒同属这一轮：capabilities 只问一次，整条路径落在 config.promptBudgetMs 之内（宿主给 UserPromptSubmit 30 s）
  const turn = core.beginPrompt();
  const context = await core.recall(adapter, input, turn);
  const prepared = context ? createContextInjection(context) : null;
  if (prepared) await core.recordInjection(adapter, input, prepared.receipt);
  // 提醒出口（方案 32 第五节）：recall 之后取，取到的拼成指令块、同一套回执包装；取不到当这一轮没有提醒
  let reminderInjection = "";
  try {
    const reminders = await core.claimReminders(adapter, input, turn);
    if (reminders.length > 0) {
      const block = createContextInjection(renderReminderInstruction(reminders, adapter.questionTool), undefined, REMINDER_TAG);
      await core.recordInjection(adapter, input, block.receipt);
      reminderInjection = block.injection;
    }
  } catch (error) {
    await writeOperationLog(config, adapter, input, {
      hook, stage: "claim_reminders", status: "error", retryable: true, error,
    }).catch(() => {});
  }
  const failed = new Set(turn.failures.map((failure) => failure.stage));
  await writeRepeatedFailureLogs(config, adapter, input, {
    hook, failures: turn.failures, settled: ["recall", "claim_reminders"].filter((stage) => !failed.has(stage)),
  });
  return [prepared?.injection || "", reminderInjection].filter(Boolean).join("\n");
}

export async function runHook(action, adapter) {
  let config;
  let input = {};
  try {
    config = loadPluginConfig();
    const hookInput = await readInput();
    const raw = hookInput.raw;
    input = hookInput.input;
    if (!config.enabled) {
      write(adapter.successOutput(action));
      return;
    }
    // 宿主的转写不在 hook 给的那个文件里时（ZCode：真会话在它的库里），适配器先把输入准备好（比如镜像出转写、换掉路径）
    if (typeof adapter.prepareInput === "function") input = await adapter.prepareInput(input, config, action);
    // 子代理的会话不进记忆（他的提示是主 agent 写的任务，结果已回到主会话的转写里），非交互会话也不进：
    // 不召回、不领提醒、不送、不留会话状态
    if (skipsSession(adapter, input)) {
      write(adapter.successOutput(action));
      return;
    }
    const core = new PluginCore(config);
    if (action === "user-prompt") {
      const injection = await promptInjection(core, adapter, input, { config, hook: action });
      write(injection ? adapter.contextOutput(injection) : adapter.successOutput(action));
      return;
    }
    if (action === "session-start") {
      if (process.env.HABITUS_HOOK_WORKER === "1") {
        const recovered = await core.recoverPending(adapter, input);
        if (recovered?.retryable) await writeOperationLog(config, adapter, input, {
          hook: action, stage: "recover", status: "pending", retryable: true,
        });
        // 宿主没有 SessionEnd（ZCode、DeepSeek Harness）：会话状态与转写镜像的退场放在每次 SessionStart 的后台顺带扫一遍
        if (adapter.retiresOnSessionStart === true) await core.retireIdleSessions(adapter);
        return;
      }
      await core.sessionStart(adapter, input);
      write(adapter.successOutput(action));
      detach(raw, config, adapter, input, action);
      return;
    }
    if (action === "stop") {
      if (process.env.HABITUS_HOOK_WORKER === "1") {
        const drained = await core.drain(adapter, input);
        if (drained?.pending || drained?.blocked) await writeOperationLog(config, adapter, input, {
          hook: action,
          delivery: drained.pending || drained.blocked,
          stage: "drain",
          status: drained.blocked ? "blocked" : "pending",
          retryable: Boolean(drained.retryable),
        });
        return;
      }
      await core.enqueueCapture(adapter, input);
      write(adapter.successOutput(action));
      detach(raw, config, adapter, input, action);
      return;
    }
    if (action === "session-end") {
      if (process.env.HABITUS_HOOK_WORKER === "1") {
        const flushed = await core.flush(adapter, input);
        if (!flushed?.ok) await writeOperationLog(config, adapter, input, {
          hook: action, stage: "flush", status: "pending", retryable: true,
        });
        await core.retireIdleSessions(adapter);
        return;
      }
      await core.enqueueCapture(adapter, input);
      write(adapter.successOutput(action));
      detach(raw, config, adapter, input, action);
      return;
    }
    if (action === "pre-compact") {
      await core.enqueueCapture(adapter, input);
      const flushed = await core.flush(adapter, input);
      if (!flushed?.ok) await writeOperationLog(config, adapter, input, {
        hook: action, stage: "flush", status: "pending", retryable: true,
      });
      write(adapter.successOutput(action));
      return;
    }
    throw new Error(`unsupported hook action: ${action}`);
  } catch (error) {
    if (config) await writeOperationLog(config, adapter, input, {
      hook: action,
      stage: process.env.HABITUS_HOOK_WORKER === "1" ? "worker" : "hook",
      status: "error",
      retryable: true,
      error,
    });
    // 配置读不出来（连接文件不对、HABITUS_URL 不是本机）时不知道状态目录在哪、写不了操作日志：在 stderr 留一行，
    // 宿主的 hook 日志里看得见（与 OpenCode / pi 入口读不了配置时同一个做法）
    else process.stderr.write(`habitus-memory: hook skipped, plugin config unusable: ${error?.message || error}\n`);
    writeDebugStack(config ? config.debug : debugEnabled(), error);
    write(adapter.successOutput(action));
  }
}
