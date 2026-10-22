import { stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { safeIdentifier } from "./shared/host-adapter.mjs";
import { mirrorEvents, mirrorPath, readSessionLog } from "./session-log.mjs";

// DeepSeek Harness 经它的 Claude Code hook 桥（@deepseek-ai/dsh-hooks-claude-code）跑这几个 hook：人说话时注入
// （hookSpecificOutput.additionalContext，与 Claude Code 同形）；SessionStart / Stop 时把会话日志镜像成本地 JSONL、按游标送回
// （协议 dsh）。桥不支持 PreCompact / SessionEnd（压缩只在表层替换，原始事件仍留在日志里，不需要压缩前抢救）。
// 子代理是独立的子会话（会话头带 parentSession）：记成 <父>.subagent.<子>，不召回、不取提醒，会话 lane 照旧跳过。
// 待实测：桥的 agent/pre-step 不分消息来源，注入的上下文、steer 消息可能再触发一次 UserPromptSubmit。

function selectCompleted(records, injectionReceipts) {
  const accepted = [];
  let safe = null;
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record == null) continue;
    accepted.push(stripInjectionReceipts(record, injectionReceipts).value);
    if (record.type === "turn/end") safe = { consumedCount: index + 1, recordCount: accepted.length };
  }
  if (safe == null) return null;
  return { consumedCount: safe.consumedCount, records: accepted.slice(0, safe.recordCount), afterTurn: true };
}

export function createDshAdapter({ readLog = readSessionLog, env = process.env } = {}) {
  return Object.freeze({
    host: "dsh",
    protocol: "dsh",
    questionTool: "ask_user_question",
    nativeSessionId(input) {
      const id = safeIdentifier(input?.session_id);
      return input?.parent_session ? safeIdentifier(`${input.parent_session}.subagent.${id}`) : id;
    },
    conversationId(input) { return `dsh-${this.nativeSessionId(input)}`; },
    prompt(input) { return typeof input?.prompt === "string" ? input.prompt.trim() : ""; },
    isSubagent(input) { return Boolean(input?.parent_session); },
    successOutput() { return {}; },
    contextOutput(context) {
      return { hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context } };
    },
    // 每一步都读会话头（认子代理）、把 transcript_path 换成本地镜像；要送转写的那几步才追加镜像
    async prepareInput(input, config, action) {
      const sessionId = typeof input?.session_id === "string" ? input.session_id : "";
      if (!sessionId) return input;
      const { header, events } = await readLog(sessionId, env);
      if (action !== "user-prompt") await mirrorEvents(config.stateRoot, sessionId, events);
      const parent = typeof header?.parentSession === "string" && header.parentSession ? header.parentSession : undefined;
      return { ...input, transcript_path: mirrorPath(config.stateRoot, sessionId), parent_session: parent };
    },
    transcriptPath(input) {
      return typeof input?.transcript_path === "string" && input.transcript_path ? input.transcript_path : null;
    },
    readTranscriptDelta(input, cursor, config, injectionReceipts) {
      return readJsonlDelta(this.transcriptPath(input), cursor, {
        maxBytes: config.maxTranscriptBytes,
        selectCompleted: (records) => selectCompleted(records, injectionReceipts),
      });
    },
  });
}

export const hostAdapter = createDshAdapter();
