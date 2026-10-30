import { stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { completedTurns, safeIdentifier } from "./shared/host-adapter.mjs";
import { findSessionLog, mirrorEvents, mirrorPath, readSessionLog, sessionsRoot } from "./session-log.mjs";

// DeepSeek Harness 经它的 Claude Code hook 桥（@deepseek-ai/dsh-hooks-claude-code）跑这几个 hook：人说话时注入
// （hookSpecificOutput.additionalContext，与 Claude Code 同形）；SessionStart / Stop 时把会话日志镜像成本地 JSONL、按游标送回
// （协议 dsh）。桥不支持 PreCompact / SessionEnd（压缩只在表层替换，原始事件仍留在日志里，不需要压缩前抢救）：会话状态与镜像的退场
// 放在 SessionStart 的后台（retiresOnSessionStart），"会话不在了"看 dsh 的会话日志还在不在（镜像是插件自己的文件，不能拿它判）。
// 子代理是独立的子会话（会话头带 parentSession）：不进记忆——不镜像、不送、不召回、不取提醒。
// 待实测：桥的 agent/pre-step 不分消息来源，注入的上下文、steer 消息可能再触发一次 UserPromptSubmit。

function selectCompleted(records, injectionReceipts) {
  const accepted = [];
  const boundaries = [];
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record == null) continue;
    accepted.push(stripInjectionReceipts(record, injectionReceipts).value);
    if (record.type === "turn/end") boundaries.push({ consumedCount: index + 1, recordCount: accepted.length });
  }
  return completedTurns(accepted, boundaries);
}

async function sessionLogExists(sessionId, env) {
  return (await findSessionLog(sessionId, sessionsRoot(env))) != null;
}

export function createDshAdapter({ readLog = readSessionLog, logExists = sessionLogExists, env = process.env } = {}) {
  return Object.freeze({
    host: "dsh",
    protocol: "dsh",
    questionTool: "ask_user_question",
    retiresOnSessionStart: true,
    nativeSessionId(input) { return safeIdentifier(input?.session_id); },
    hostSessionId(input) { return typeof input?.session_id === "string" ? input.session_id : null; },
    // 没记下宿主会话 id 的老状态当它还在（不退场）
    async sessionExists(state) {
      if (typeof state?.hostSessionId !== "string" || !state.hostSessionId) return true;
      return logExists(state.hostSessionId, env);
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
      const parent = typeof header?.parentSession === "string" && header.parentSession ? header.parentSession : undefined;
      if (action !== "user-prompt" && !parent) await mirrorEvents(config.stateRoot, sessionId, events);
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
