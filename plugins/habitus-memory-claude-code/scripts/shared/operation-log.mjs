// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
import { createHash, randomBytes } from "node:crypto";
import { chmod, mkdir, open, readdir, rename, stat, unlink } from "node:fs/promises";
import { join } from "node:path";

const ROTATED = /^operations-[0-9TZ:.-]+-[0-9]+-[0-9a-f]+\.jsonl$/;

// 超过上限就把当前那份改成一个唯一的名字（时刻 + 进程 + 随机），再按名字清掉多出来的旧份（26 号清单 7.2）。
// 几个进程同时轮转：改名都是唯一名，谁也不会盖掉谁；当前那份已经被别人改走了（ENOENT）就不用再改。
async function rotateIfLarge(directory, target, config) {
  let size;
  try {
    size = (await stat(target)).size;
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  if (size < config.operationLogMaxBytes) return;
  const stamp = new Date().toISOString().replaceAll(":", "-");
  const rotated = join(directory, `operations-${stamp}-${process.pid}-${randomBytes(4).toString("hex")}.jsonl`);
  try {
    await rename(target, rotated);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const old = (await readdir(directory)).filter((name) => ROTATED.test(name)).sort();
  for (const name of old.slice(0, Math.max(0, old.length - config.operationLogKeep))) {
    try {
      await unlink(join(directory, name));
    } catch {
      // 删不掉的那一份（已被别人删、权限不对）跳过，接着清后面的：停下来会让份数无限增长（L10.12）
    }
  }
}

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
    try {
      await rotateIfLarge(directory, target, config);
    } catch {
      // 轮转失败不能连这一条记录也丢掉：照常追加，下一次再轮转
    }
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
