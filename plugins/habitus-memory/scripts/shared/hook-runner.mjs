// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
import { createHash, randomBytes } from "node:crypto";
import { spawn } from "node:child_process";
import { loadPluginConfig } from "./config.mjs";
import { writeOperationLog } from "./operation-log.mjs";
import { PluginCore } from "./plugin-core.mjs";

const INJECTION_END = "</habitus-memory-context>";

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

export function createContextInjection(value, nonce = randomBytes(16).toString("hex")) {
  if (!/^[0-9a-f]{32}$/.test(nonce)) throw new Error("injection nonce is invalid");
  const injection = `<habitus-memory-context receipt="${nonce}">\n${encodeContextPayload(value)}\n${INJECTION_END}`;
  return Object.freeze({
    injection,
    receipt: Object.freeze({ nonce, digest: createHash("sha256").update(injection).digest("hex") }),
  });
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
    const core = new PluginCore(config);
    if (action === "user-prompt") {
      const context = await core.recall(adapter, input);
      const prepared = context ? createContextInjection(context) : null;
      const injection = prepared?.injection || "";
      if (prepared) await core.recordInjection(adapter, input, prepared.receipt);
      write(
        context
          ? adapter.contextOutput(injection)
          : adapter.successOutput(action),
      );
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
