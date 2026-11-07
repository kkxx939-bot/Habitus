// GENERATED FROM plugins/memory-plugin-shared/lib. DO NOT EDIT.
import { createHash, randomBytes } from "node:crypto";
import { chmod, open, readFile, readdir, rename, stat, unlink, writeFile } from "node:fs/promises";
import { makePrivateDirectory } from "./private-dir.mjs";
import { join } from "node:path";

const ROTATED = /^operations-[0-9TZ:.-]+-[0-9]+-[0-9a-f]+\.jsonl$/;
// 接连出现的同一原因记到哪儿了（logs/repeated-failures.json）：一个会话一项，最多留这么多个会话（最早记的先丢）
const REPEATS_FILE = "repeated-failures.json";
const REPEATS_KEPT = 64;
export const REPEAT_RELOG_MS = 60 * 60 * 1000;

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
    await makePrivateDirectory(directory);
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
      reason: event.reason == null ? null : String(event.reason).slice(0, 64),
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

async function readRepeats(directory) {
  try {
    const value = JSON.parse(await readFile(join(directory, REPEATS_FILE), "utf8"));
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

async function writeRepeats(directory, repeats) {
  const kept = Object.entries(repeats)
    .sort(([, a], [, b]) => String(a.touchedAt).localeCompare(String(b.touchedAt)))
    .slice(-REPEATS_KEPT);
  await makePrivateDirectory(directory);
  await writeFile(join(directory, REPEATS_FILE), JSON.stringify(Object.fromEntries(kept)), { mode: 0o600 });
}

// 每轮都会走到的环节（UserPromptSubmit 的召回、取提醒）没成时记的日志：同一会话、同一环节接连是同一个原因只记第一次
// （服务停着时他每说一句就多两条，一样的话刷满日志、把别的记录轮转掉），过了 REPEAT_RELOG_MS 还是这个原因再记一次，看得出它一直没好；
// settled 里的环节这一轮成了，清掉它的记号，下次再出错照常记。记号文件读写不加锁：几个会话同时写最多多记一条，日志照样不丢
export async function writeRepeatedFailureLogs(config, adapter, input, { hook, failures, settled }, now = Date.now()) {
  try {
    const directory = join(config.stateRoot, "logs");
    const session = sessionIdentity(adapter, input);
    const repeats = await readRepeats(directory);
    const previous = repeats[session]?.stages || {};
    const stages = Object.fromEntries(Object.entries(previous).filter(([stage]) => !settled.includes(stage)));
    for (const failure of failures) {
      const last = stages[failure.stage];
      if (last?.reason === failure.reason && now - Date.parse(last.loggedAt) < REPEAT_RELOG_MS) continue;
      stages[failure.stage] = { reason: failure.reason, loggedAt: new Date(now).toISOString() };
      await writeOperationLog(config, adapter, input, {
        hook, stage: failure.stage, status: "failed", retryable: failure.retryable !== false, reason: failure.reason, error: new Error(failure.detail),
      });
    }
    if (JSON.stringify(stages) === JSON.stringify(previous)) return;
    if (Object.keys(stages).length === 0) delete repeats[session];
    else repeats[session] = { stages, touchedAt: new Date(now).toISOString() };
    await writeRepeats(directory, repeats);
  } catch {
    // 日志永远不能改变宿主 hook 的 fail-open 行为。
  }
}
