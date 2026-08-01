import { createHash } from "node:crypto";
import { chmod, mkdir, open } from "node:fs/promises";
import { join } from "node:path";

function sessionIdentity(adapter, input) {
  try {
    const value = adapter.nativeSessionId(input);
    return createHash("sha256").update(`${adapter.host}\n${value}`).digest("hex").slice(0, 24);
  } catch {
    return "unavailable";
  }
}

export async function writeOperationLog(config, adapter, input, event) {
  try {
    const directory = join(config.stateRoot, "logs");
    const target = join(directory, "operations.jsonl");
    await mkdir(directory, { recursive: true, mode: 0o700 });
    const record = {
      schemaVersion: 1,
      timestamp: new Date().toISOString(),
      host: adapter?.host || "unknown",
      hook: String(event.hook || "unknown"),
      session: sessionIdentity(adapter, input),
      delivery: event.delivery == null ? null : String(event.delivery),
      stage: String(event.stage || "unknown"),
      status: String(event.status || "error"),
      retryable: event.retryable === true,
      errorType: event.error == null ? null : String(event.error?.name || "Error").slice(0, 120),
      detail: event.error == null ? null : String(event.error?.message || event.error).slice(0, 512),
    };
    const handle = await open(target, "a", 0o600);
    try {
      await handle.writeFile(`${JSON.stringify(record)}\n`, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
    await chmod(target, 0o600);
  } catch {
    // 日志永远不能改变宿主 hook 的 fail-open 行为。
  }
}
