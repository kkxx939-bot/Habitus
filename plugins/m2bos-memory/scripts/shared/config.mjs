// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
import { homedir } from "node:os";
import { join, resolve } from "node:path";

function booleanValue(value, fallback) {
  if (value == null || value === "") return fallback;
  return !["0", "false", "no", "off"].includes(String(value).trim().toLowerCase());
}

function integerValue(value, fallback, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) return fallback;
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
    throw new Error("M2BOS_URL must be an unauthenticated loopback HTTP URL");
  }
  if (!/^\/*$/.test(parsed.pathname)) throw new Error("M2BOS_URL must not contain a path");
  parsed.pathname = "";
  return parsed.toString().replace(/\/$/, "");
}

export function loadPluginConfig(env = process.env) {
  const stateRoot = resolve(env.M2BOS_PLUGIN_STATE_DIR || join(homedir(), ".m2bos", "agent-plugin"));
  return Object.freeze({
    enabled: booleanValue(env.M2BOS_MEMORY_ENABLED, true),
    baseUrl: requireLoopbackURL(env.M2BOS_URL || "http://127.0.0.1:8787"),
    stateRoot,
    timeoutMs: integerValue(env.M2BOS_PLUGIN_TIMEOUT_MS, 15_000, 250, 120_000),
    maxTranscriptBytes: integerValue(env.M2BOS_PLUGIN_MAX_TRANSCRIPT_BYTES, 64 * 1024 * 1024, 1024, 256 * 1024 * 1024),
    debug: booleanValue(env.M2BOS_PLUGIN_DEBUG, false),
  });
}
