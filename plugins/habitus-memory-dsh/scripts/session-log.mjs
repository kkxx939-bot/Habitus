import { createHash } from "node:crypto";
import { appendFile, mkdir, readFile, readdir, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import * as zlib from "node:zlib";

// DeepSeek Harness（deepseek-ai/deepseek-harness，MIT）的会话日志（dsh-session-persistence-jsonl，0.2.1-rc.2 源码）：
//   $DSH_HOME/sessions/--<规范化 cwd>--/<转义后的会话 id>/session.v<N>.jsonl[.zstd]
// 取版本号最高的那一代；默认是首尾相接、带校验和的 Zstandard 帧，`compression: 'none'` 时是纯文本。第一行是会话头，
// 之后一行一个事件 {type, seq, time, data}。hook 桥给 hook 的 transcript_path 永远是空串，所以插件自己找、自己解。
// 待实测：当前一代行格式里"无损来源序列"的写法与文档里的事件信封是否一字不差。

export function sessionsRoot(env = process.env) {
  return join(env.DSH_HOME || join(env.HOME || homedir(), ".dsh"), "sessions");
}

// 会话 id → 一个安全路径段（与 dsh 的 encodeSegment 同一规则）
export function encodeSegment(raw) {
  if (raw === ".") return "~002E";
  if (raw === "..") return "~002E~002E";
  let out = "";
  for (let index = 0; index < raw.length; index += 1) {
    const code = raw.charCodeAt(index);
    const ch = String.fromCharCode(code);
    out += ch !== "~" && /^[A-Za-z0-9._-]$/.test(ch) ? ch : `~${code.toString(16).toUpperCase().padStart(4, "0")}`;
  }
  return out;
}

// 不按 cwd 去算项目目录（cwd 中途可能换过）：在各项目目录下找这个会话的目录
export async function findSessionLog(sessionId, root) {
  const segment = encodeSegment(sessionId);
  let projects;
  try { projects = await readdir(root, { withFileTypes: true }); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
  for (const project of projects.filter((entry) => entry.isDirectory())) {
    const directory = join(root, project.name, segment);
    let names;
    try { names = await readdir(directory); }
    catch (error) { if (error?.code === "ENOENT" || error?.code === "ENOTDIR") continue; throw error; }
    const generations = names
      .map((name) => ({ name, match: name.match(/^session(?:\.v(\d+))?\.jsonl(\.zstd)?$/) }))
      .filter((item) => item.match)
      .map((item) => ({ path: join(directory, item.name), version: Number(item.match[1] ?? 0), zstd: Boolean(item.match[2]) }))
      .sort((left, right) => right.version - left.version || Number(right.zstd) - Number(left.zstd));
    if (generations.length > 0) return generations[0];
  }
  return null;
}

// 压缩日志是一帧接一帧（每批事件一帧），Node 的 zstdDecompressSync 只解第一帧：按 zstd 帧格式（RFC 8878 第 3.1 节）
// 自己切出每一帧的边界再逐帧解。最后一帧写到一半（崩溃）就不要它——dsh 下次写入前会把它修好，下一轮再读。
export function zstdFrames(buffer) {
  const frames = [];
  let offset = 0;
  while (offset < buffer.length) {
    const start = offset;
    if (buffer.length - offset < 5) break;
    if (buffer.readUInt32LE(offset) !== 0xfd2fb528) throw new Error(`not a Zstandard frame at byte ${offset}`);
    const descriptor = buffer.readUInt8(offset + 4);
    offset += 5;
    const contentSizeFlag = descriptor >>> 6;
    const singleSegment = (descriptor & 0x20) !== 0;
    const checksum = (descriptor & 0x04) !== 0;
    const dictionaryBytes = [0, 1, 2, 4][descriptor & 0x03];
    const contentSizeBytes = contentSizeFlag === 0 ? (singleSegment ? 1 : 0) : 1 << contentSizeFlag;
    offset += (singleSegment ? 0 : 1) + dictionaryBytes + contentSizeBytes;
    let complete = false;
    while (offset + 3 <= buffer.length) {
      const header = buffer.readUIntLE(offset, 3);
      offset += 3;
      const blockType = (header >>> 1) & 0x03;
      offset += blockType === 0x01 ? 1 : header >>> 3; // RLE 块只带一个字节
      if (offset > buffer.length) break;
      if ((header & 1) !== 0) { complete = true; break; }
    }
    if (complete && checksum) offset += 4;
    if (!complete || offset > buffer.length) break;
    frames.push(buffer.subarray(start, offset));
  }
  return frames;
}

// 解出完整的行
export function decodeLog(buffer, zstd) {
  // zstd 要 Node ≥ 22.15：按名字导入在更老的 Node 上整个模块都加载不了，所以用到时才查
  if (zstd && typeof zlib.zstdDecompressSync !== "function") {
    throw new Error(`dsh session log is zstd-compressed; Node ${process.version} has no zstd (needs >= 22.15)`);
  }
  const text = zstd
    ? zstdFrames(buffer).map((frame) => zlib.zstdDecompressSync(frame).toString("utf8")).join("")
    : buffer.toString("utf8");
  const lines = text.split("\n");
  if (!text.endsWith("\n")) lines.pop();
  const records = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    try { records.push(JSON.parse(line)); } catch { /* 坏行：解不出的不送 */ }
  }
  return records;
}

export async function readSessionLog(sessionId, env = process.env) {
  const log = await findSessionLog(sessionId, sessionsRoot(env));
  if (!log) return { header: null, events: [] };
  const records = decodeLog(await readFile(log.path), log.zstd);
  const header = records.find((record) => record && !Number.isFinite(record.seq)) ?? null;
  return { header, events: records.filter((record) => Number.isFinite(record?.seq) && typeof record?.type === "string") };
}

export function mirrorPath(stateRoot, sessionId) {
  const key = createHash("sha256").update(`dsh\n${sessionId}`).digest("hex");
  return join(stateRoot, "transcripts", "dsh", `${key}.jsonl`);
}

// 事件按 seq 单调：本地镜像只追加 seq 比已镜像的大的
export async function mirrorEvents(stateRoot, sessionId, events) {
  const path = mirrorPath(stateRoot, sessionId);
  let last = -Infinity;
  try {
    for (const line of (await readFile(path, "utf8")).split("\n")) {
      if (!line.trim()) continue;
      try { const seq = JSON.parse(line)?.seq; if (Number.isFinite(seq) && seq > last) last = seq; } catch {}
    }
  } catch (error) { if (error?.code !== "ENOENT") throw error; }
  const fresh = events.filter((event) => event.seq > last).sort((left, right) => left.seq - right.seq);
  if (fresh.length > 0) {
    await mkdir(join(stateRoot, "transcripts", "dsh"), { recursive: true, mode: 0o700 });
    const existing = await stat(path).then((value) => value.size, () => 0);
    const prefix = existing > 0 && !(await readFile(path, "utf8")).endsWith("\n") ? "\n" : "";
    await appendFile(path, `${prefix}${fresh.map((event) => JSON.stringify(event)).join("\n")}\n`, { mode: 0o600 });
  }
  return { path, appended: fresh.length };
}
