import { createHash, randomBytes } from "node:crypto";
import { spawn } from "node:child_process";
import { loadPluginConfig } from "./config.mjs";
import { INJECTION_TAGS } from "./injection-receipt.mjs";
import { writeOperationLog } from "./operation-log.mjs";
import { PluginCore } from "./plugin-core.mjs";

const [MEMORY_TAG, REMINDER_TAG] = INJECTION_TAGS;
// 提醒的三个固定选项由服务给；没给时用这三个（产品定措辞，所有宿主一样，方案 32 第 4.1 条）。
const DEFAULT_OPTIONS = ["现在就做", "这次不用", "知道了"];

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

// 给 agent 的指令块（方案 32 第 5.1 条；措辞要对着真实宿主调，R1 / R8）：先问再做；题目原样、三个选项、允许他自己填；
// 他怎么答都不用另外处理——系统从转写里认提问与作答。
export function renderReminderInstruction(reminders, questionTool) {
  const tool = questionTool || "向用户提问的工具";
  const lines = [
    `Habitus 的提醒，不是用户说的话。在做任何别的事之前，先用 ${tool} 问用户下面${reminders.length > 1 ? `这 ${reminders.length} 件事（一件一问，按顺序）` : "这件事"}：`,
    "题目原样用这句话、给这三个选项、允许他自己填别的话。问完照常做他让你做的事；他怎么答你都不用另外处理，系统自己会记。",
  ];
  reminders.forEach((item, index) => {
    const options = Array.isArray(item.options) && item.options.length > 0 ? item.options : DEFAULT_OPTIONS;
    lines.push(`${index + 1}. 『${String(item.text).trim()}』`);
    lines.push(`   选项：${options.join(" / ")}`);
  });
  return lines.join("\n");
}

// 他说了一句话：召回记忆、取提醒，拼成要注入的块（没有就是空串）。命令 hook 与进程内插件（OpenCode）共用这一份：
// 取提醒出错只丢提醒、记一笔日志，召回的记忆照样注入；召回出错由调用方兜（那时什么都不注入）。
export async function promptInjection(core, adapter, input, { config, hook }) {
  // 子代理的"一句话"是主 agent 写的任务，不是他说的话（宿主认得出子代理时由适配器说）
  if (typeof adapter.isSubagent === "function" && adapter.isSubagent(input)) return "";
  const context = await core.recall(adapter, input);
  const prepared = context ? createContextInjection(context) : null;
  if (prepared) await core.recordInjection(adapter, input, prepared.receipt);
  // 提醒出口（方案 32 第五节）：recall 之后取，取到的拼成指令块、同一套回执包装；取不到当这一轮没有提醒
  let reminderInjection = "";
  try {
    const reminders = await core.claimReminders(adapter, input);
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
    const core = new PluginCore(config);
    if (action === "user-prompt") {
      const injection = await promptInjection(core, adapter, input, { config, hook: action });
      write(injection ? adapter.contextOutput(injection) : adapter.successOutput(action));
      return;
    }
    if (["session-start", "subagent-start"].includes(action)) {
      if (process.env.HABITUS_HOOK_WORKER === "1") {
        const recovered = await core.recoverPending(adapter, input);
        if (recovered?.retryable) await writeOperationLog(config, adapter, input, {
          hook: action, stage: "recover", status: "pending", retryable: true,
        });
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
    if (["session-end", "subagent-stop"].includes(action)) {
      if (process.env.HABITUS_HOOK_WORKER === "1") {
        const flushed = await core.flush(adapter, input);
        if (!flushed?.ok) await writeOperationLog(config, adapter, input, {
          hook: action, stage: "flush", status: "pending", retryable: true,
        });
        if (action === "session-end") await core.retireIdleSessions(adapter);
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
    if (["1", "true", "yes", "on"].includes(String(process.env.HABITUS_PLUGIN_DEBUG || "").toLowerCase())) {
      process.stderr.write(`[habitus-memory] ${error?.stack || error}\n`);
    }
    write(adapter.successOutput(action));
  }
}
