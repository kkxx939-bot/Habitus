import { createHash } from "node:crypto";
import { appendFile, mkdir, open, readFile, rm, truncate } from "node:fs/promises";
import { join } from "node:path";
import { completedTurns } from "./host-adapter.mjs";
import { stripInjectionReceipts } from "./injection-receipt.mjs";
import { stripMedia } from "./media.mjs";

// "消息 + 片段"型宿主（OpenCode、ZCode：一条消息 {info, parts}，会话不落成文件、要从宿主的 API 或库里拿）的转写镜像：
// 把已完成的消息按顺序镜像成本地 JSONL，之后与 Codex / Claude Code 同一条路——按游标读增量、进 outbox、送回服务。
// 只追加不改写：共享库的游标按"文件前缀不变"认增量。不镜像的：压缩写的摘要和压缩后重放的原话（不是他和 agent 说的话）；
// 附件、读图的 data: base64 换成占位（media.mjs；服务端用不上，留着会把镜像撑大）。

export function transcriptPath(stateRoot, sessionId, host) {
  const key = createHash("sha256").update(`${host}\n${sessionId}`).digest("hex");
  return join(stateRoot, "transcripts", host, `${key}.jsonl`);
}

// 一轮读到最后一条完成了的助手消息为止（info.time.completed 有了才算完成）；只切在一轮结束处（裁定 69）：OpenCode 每调一次
// 模型是一条助手消息，要接着调工具的那条 finish = "tool-calls"，不切——压缩前送的那一批也不把一轮拆开
export function selectCompletedMessages(records, injectionReceipts) {
  const accepted = [];
  const boundaries = [];
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record == null) continue;
    accepted.push(stripInjectionReceipts(record, injectionReceipts).value);
    if (
      record?.info?.role === "assistant"
      && Number.isFinite(record?.info?.time?.completed)
      && record?.info?.finish !== "tool-calls"
    ) {
      boundaries.push({ consumedCount: index + 1, recordCount: accepted.length });
    }
  }
  return completedTurns(accepted, boundaries);
}

export class TranscriptMirror {
  constructor(stateRoot, host) {
    this.stateRoot = stateRoot;
    this.host = host;
    this.mirrored = new Map();
    this.queues = new Map();
  }

  // 同一会话的镜像排队做：空闲、压缩、删除几路事件可能同时到，并发追加会把同一批写两遍
  mirror(sessionId, messages) {
    const previous = this.queues.get(sessionId) ?? Promise.resolve();
    const next = previous.catch(() => {}).then(() => this.#mirror(sessionId, messages));
    this.queues.set(sessionId, next);
    next.finally(() => { if (this.queues.get(sessionId) === next) this.queues.delete(sessionId); }).catch(() => {});
    return next;
  }

  // 会话删掉、最后一批送完之后，镜像文件没用了
  async forget(sessionId) {
    await (this.queues.get(sessionId) ?? Promise.resolve()).catch(() => {});
    this.mirrored.delete(sessionId);
    await rm(transcriptPath(this.stateRoot, sessionId, this.host), { force: true });
  }

  async #mirror(sessionId, messages) {
    const path = transcriptPath(this.stateRoot, sessionId, this.host);
    const seen = await this.#seen(sessionId, path);
    const lines = [];
    const ids = [];
    const list = Array.isArray(messages) ? messages : [];
    const skipped = compactionArtifacts(list);
    for (const [index, message] of list.entries()) {
      const id = message?.info?.id;
      if (typeof id !== "string" || seen.has(id) || ids.includes(id) || skipped.has(id)) continue;
      // 还在写的助手消息先不镜像，后面的也不越过它；它后面已经有消息了，说明那一轮早断了（进程崩溃不会补写完成时刻），照常往下
      if (message.info.role === "assistant" && !Number.isFinite(message.info?.time?.completed) && index === list.length - 1) break;
      lines.push(JSON.stringify(slim(message)));
      ids.push(id);
    }
    if (lines.length > 0) {
      await mkdir(join(this.stateRoot, "transcripts", this.host), { recursive: true, mode: 0o700 });
      await dropTornTail(path);
      await appendFile(path, `${lines.join("\n")}\n`, { mode: 0o600 });
      for (const id of ids) seen.add(id); // 写成功才算镜像过；写失败下次重来
    }
    return { path, appended: lines.length };
  }


  async #seen(sessionId, path) {
    // 只在冷缓存时读一次文件；同一会话已经排队，不会两路同时读
    if (this.mirrored.has(sessionId)) return this.mirrored.get(sessionId);
    const seen = new Set();
    try {
      for (const line of (await readFile(path, "utf8")).split("\n")) {
        if (!line.trim()) continue;
        try { const id = JSON.parse(line)?.info?.id; if (typeof id === "string") seen.add(id); } catch {}
      }
    } catch (error) { if (error?.code !== "ENOENT") throw error; }
    this.mirrored.set(sessionId, seen);
    return seen;
  }
}

// 上次追加写到一半（进程被杀、盘满）会留一行残缺的 JSON：截到最后一个换行，否则新内容拼在坏行上、之后每次读都报错。
// 往本地 JSONL 镜像追加前都先过这一步（DeepSeek Harness 的事件镜像也用它）
export async function dropTornTail(path) {
  let handle;
  try { handle = await open(path, "r"); }
  catch (error) { if (error?.code === "ENOENT") return; throw error; }
  try {
    const { size } = await handle.stat();
    if (size === 0) return;
    const tail = Buffer.alloc(Math.min(size, 64 * 1024));
    let end = size;
    while (end > 0) {
      const start = Math.max(0, end - tail.length);
      const { bytesRead } = await handle.read(tail, 0, end - start, start);
      const index = tail.subarray(0, bytesRead).lastIndexOf(0x0a);
      if (end === size && index === bytesRead - 1) return;
      if (index >= 0) { await handle.close(); handle = null; await truncate(path, start + index + 1); return; }
      end = start;
    }
    await handle.close(); handle = null;
    await truncate(path, 0);
  } finally {
    await handle?.close();
  }
}

// 压缩（OpenCode compaction.ts）：一条只有 compaction 段的 user 消息 → summary 为真的助手摘要；只有自动压缩且是上下文溢出触发的
// （compaction 段 auto && overflow），才把压缩前最后那句人话复制成一条新的 user 消息重放，复制时图片 / PDF 这类 file 段换成
// "[Attached <mime>: <文件名>]" 文字段。摘要和重放的那句都不是这一轮新说的，不进转写；手动压缩之后他说的话即使与压缩前一字不差也是新说的。
// ZCode 的压缩不写 summary 为真的助手消息、compaction 段不带 overflow、也不重放原话，这里认不出任何东西，照常镜像
function compactionArtifacts(messages) {
  const skipped = new Set();
  let lastSaid = null;
  let replays = false;
  let pendingReplay = null;
  for (const message of messages) {
    const info = message?.info;
    if (typeof info?.id !== "string") continue;
    const parts = Array.isArray(message.parts) ? message.parts : [];
    if (info.role === "assistant" && (info.summary === true || info.mode === "compaction")) {
      skipped.add(info.id);
      pendingReplay = replays ? lastSaid : null;
      continue;
    }
    if (info.role !== "user") { pendingReplay = null; continue; }
    const compaction = parts.find((part) => part?.type === "compaction");
    if (compaction) { replays = compaction.auto === true && compaction.overflow === true; continue; }
    const said = spokenText(parts.map(asReplayed));
    if (pendingReplay != null && said === pendingReplay) skipped.add(info.id);
    pendingReplay = null;
    if (said) lastSaid = said;
  }
  return skipped;
}

// 重放时 OpenCode 对 file 段的改写（compaction.ts，MessageV2.isMedia：image/* 与 application/pdf）
function asReplayed(part) {
  if (part?.type !== "file" || typeof part.mime !== "string") return part;
  if (!part.mime.startsWith("image/") && part.mime !== "application/pdf") return part;
  return { type: "text", text: `[Attached ${part.mime}: ${part.filename ?? "file"}]` };
}

function spokenText(parts) {
  return parts
    .filter((part) => part?.type === "text" && !part.synthetic && !part.ignored && typeof part.text === "string")
    .map((part) => part.text)
    .join("\n")
    .trim();
}

// 镜像前就把附件、读图的 data: base64 换成占位（media.mjs，与入队时同一份规则）：服务端用不上，留着会把镜像撑大
function slim(message) {
  return { info: message.info, parts: stripMedia(message.parts || []) };
}
