import { homedir } from "node:os";
import { join } from "node:path";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { safeIdentifier } from "./shared/host-adapter.mjs";
import { TranscriptMirror, selectCompletedMessages, transcriptPath } from "./shared/parts-transcript.mjs";

// ZCode（zai-org/ZCode，Apache-2.0）认 Claude Code 的插件与 hooks.json，同一份插件在 ZCode 里跑时用这个适配器
// （hook 进程带 ZCODE_SESSION_ID，见 host-adapter.mjs）。和 Claude Code 不一样的两处（ZCode v3.15.1 源码）：
// - hook 给的 transcript_path 是这一次 hook 临时写的一行、跑完就删；真会话在 ~/.zcode/cli/db/db.sqlite 的 message / part 表，
//   一条消息 {info, parts} 与 OpenCode 同形。所以每次 hook 先把这个会话已完成的消息从库里镜像成本地 JSONL，再按游标读增量。
// - 只有 SessionStart / UserPromptSubmit / Stop 等七个事件：没有 PreCompact、SessionEnd、Subagent*（hooks.json 里写了也只记一条
//   警告跳过）；子代理里不跑任何 hook。
// 待实测：注入的上下文在库里存成什么样（ZCode 前面加一行 "UserPromptSubmit hook additional context:"）；后台送转写的进程会不会
// 被 ZCode 连同进程组杀掉。

const SESSION_DB = join(homedir(), ".zcode", "cli", "db", "db.sqlite");
const HOST = "zcode";

// 一个会话的消息与片段，按写入顺序；data 里不含身份字段，按列补回来（ZCode saveMessage / savePart 存的时候拆掉了）
export async function readZCodeSession(sessionId, dbPath = SESSION_DB) {
  // node:sqlite 要 Node ≥ 22.5；更老的 Node 上这里报清楚，hook 照常放行、错误记进插件日志
  const { DatabaseSync } = await import("node:sqlite").catch(() => {
    throw new Error(`reading the ZCode session database needs node:sqlite (Node >= 22.5); found Node ${process.version}`);
  });
  const db = new DatabaseSync(dbPath, { readOnly: true });
  try {
    const messages = db.prepare(
      "select id, data from message where session_id = ? order by sequence, time_created, id",
    ).all(sessionId);
    const parts = db.prepare(
      "select id, message_id, data from part where session_id = ? order by message_id, sequence, id",
    ).all(sessionId);
    const byMessage = new Map();
    for (const row of parts) {
      const list = byMessage.get(row.message_id) ?? [];
      list.push({ ...JSON.parse(String(row.data)), id: row.id, sessionID: sessionId, messageID: row.message_id });
      byMessage.set(row.message_id, list);
    }
    return messages.map((row) => ({
      info: { ...JSON.parse(String(row.data)), id: row.id, sessionID: sessionId },
      parts: byMessage.get(row.id) ?? [],
    }));
  } finally {
    db.close();
  }
}

export function createZCodeAdapter({ dbPath = SESSION_DB, readSession = readZCodeSession } = {}) {
  return Object.freeze({
    host: HOST,
    protocol: "zcode",
    questionTool: "AskUserQuestion",
    nativeSessionId(input) { return safeIdentifier(input?.session_id); },
    conversationId(input) { return `zcode-${this.nativeSessionId(input)}`; },
    prompt(input) { return typeof input?.prompt === "string" ? input.prompt.trim() : ""; },
    successOutput() { return {}; },
    contextOutput(context) {
      return { hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context } };
    },
    // 每个 hook 都把 transcript_path 换成本地镜像（ZCode 给的那个是临时文件，状态里记它会被当成"转写没了"提前退场）；
    // 只有要送转写的那几步才去库里读、追加镜像
    async prepareInput(input, config, action) {
      const sessionId = typeof input?.session_id === "string" ? input.session_id : "";
      if (!sessionId) return input;
      const path = transcriptPath(config.stateRoot, sessionId, HOST);
      if (action !== "user-prompt") {
        await new TranscriptMirror(config.stateRoot, HOST).mirror(sessionId, await readSession(sessionId, dbPath));
      }
      return { ...input, transcript_path: path };
    },
    transcriptPath(input) {
      return typeof input?.transcript_path === "string" && input.transcript_path ? input.transcript_path : null;
    },
    readTranscriptDelta(input, cursor, config, injectionReceipts) {
      return readJsonlDelta(this.transcriptPath(input), cursor, {
        maxBytes: config.maxTranscriptBytes,
        selectCompleted: (records) => selectCompletedMessages(records, injectionReceipts),
      });
    },
  });
}

export const zcodeAdapter = createZCodeAdapter();
