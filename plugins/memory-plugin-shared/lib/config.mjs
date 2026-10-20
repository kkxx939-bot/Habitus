import { lstatSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

function booleanValue(value, fallback) {
  if (value == null || value === "") return fallback;
  return !["0", "false", "no", "off"].includes(String(value).trim().toLowerCase());
}

function integerValue(value, fallback, minimum, maximum) {
  // 没设、空串、不是正整数都走默认值：Number("") 是 0，原先会被收成下限，比如宽限期从 24 小时静默降到 1 小时（L10.12）。
  // 正数照旧收进 [下限, 上限]
  if (value == null || String(value).trim() === "") return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}

function requireLoopbackURL(value) {
  const parsed = new URL(value);
  const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  const loopback = host === "localhost" || host === "::1" || /^127(?:\.\d{1,3}){3}$/.test(host);
  if (
    !loopback
    || !["http:", "https:"].includes(parsed.protocol)
    || parsed.username
    || parsed.password
    || parsed.search
    || parsed.hash
  ) {
    throw new Error("HABITUS_URL must be an unauthenticated loopback HTTP URL");
  }
  if (!/^\/*$/.test(parsed.pathname)) throw new Error("HABITUS_URL must not contain a path");
  parsed.pathname = "";
  return parsed.toString().replace(/\/$/, "");
}

function connectionURL(stateRoot) {
  const path = join(stateRoot, "connection.json");
  try {
    const metadata = lstatSync(path);
    if (!metadata.isFile() || metadata.isSymbolicLink() || metadata.size > 4096) {
      throw new Error("plugin connection config must be a bounded regular file");
    }
    if ((metadata.mode & 0o022) !== 0) {
      throw new Error("plugin connection config must have private, non-writable permissions");
    }
    if (typeof process.getuid === "function" && metadata.uid !== process.getuid()) {
      throw new Error("plugin connection config must be owned by the current user");
    }
    const value = JSON.parse(readFileSync(path, "utf8"));
    const keys = Object.keys(value || {}).sort();
    if (
      value?.schema_version !== 1
      || typeof value?.base_url !== "string"
      || keys.join(",") !== "base_url,schema_version"
    ) {
      throw new Error("plugin connection config has an invalid schema");
    }
    return value.base_url;
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

export function loadPluginConfig(env = process.env) {
  const selectedConfig = env.HABITUS_CONFIG_FILE
    ? join(dirname(resolve(env.HABITUS_CONFIG_FILE)), "agent-plugin")
    : join(homedir(), ".habitus", "agent-plugin");
  const stateRoot = resolve(env.HABITUS_PLUGIN_STATE_DIR || selectedConfig);
  return Object.freeze({
    enabled: booleanValue(env.HABITUS_MEMORY_ENABLED, true),
    baseUrl: requireLoopbackURL(env.HABITUS_URL || connectionURL(stateRoot) || "http://127.0.0.1:8787"),
    stateRoot,
    timeoutMs: integerValue(env.HABITUS_PLUGIN_TIMEOUT_MS, 15_000, 250, 120_000),
    maxTranscriptBytes: integerValue(env.HABITUS_PLUGIN_MAX_TRANSCRIPT_BYTES, 64 * 1024 * 1024, 1024, 256 * 1024 * 1024),
    debug: booleanValue(env.HABITUS_PLUGIN_DEBUG, false),
    // 操作日志按大小轮转（26 号清单 7.2）：超过这么大就换一份，旧的留这么多份
    operationLogMaxBytes: integerValue(env.HABITUS_PLUGIN_LOG_MAX_BYTES, 5 * 1024 * 1024, 4096, 256 * 1024 * 1024),
    operationLogKeep: integerValue(env.HABITUS_PLUGIN_LOG_KEEP, 3, 1, 50),
    // 会话状态的退场（26 号清单 7.4）：transcript 不在了、outbox 空了，状态又这么久没动过，才删
    sessionRetireGraceMs: integerValue(env.HABITUS_PLUGIN_SESSION_GRACE_HOURS, 24, 1, 720) * 3_600_000,
    // 提醒出口（方案 32 第五节）：一次最多取几条（服务端另有上限）、取的单独超时（在 UserPromptSubmit 的预算之内，失败就当这一轮没有提醒）
    reminderClaimLimit: integerValue(env.HABITUS_PLUGIN_REMINDER_LIMIT, 3, 1, 20),
    reminderTimeoutMs: integerValue(env.HABITUS_PLUGIN_REMINDER_TIMEOUT_MS, 2_000, 250, 30_000),
  });
}
