import { stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { safeIdentifier } from "./shared/host-adapter.mjs";
import { QUESTION_TOOL } from "./question-tool.mjs";

// pi 自己把会话落成 JSONL（`~/.pi/agent/sessions/.../<时刻>_<会话 id>.jsonl`，只追加；迁移、分叉时整份重写，游标按前缀认出来）：
// 直接读它，与 Codex / Claude Code 同一条路——按游标读增量、进 outbox、送回服务（协议 pi）。
// 一轮读到最后一条"说完了"的助手消息为止：stopReason 是 toolUse（还要跑工具）或 pending 的不算说完。

function settledAssistant(record) {
  return record?.type === "message"
    && record.message?.role === "assistant"
    && !["toolUse", "pending"].includes(record.message?.stopReason);
}

function selectCompleted(records, injectionReceipts) {
  const accepted = [];
  let safe = null;
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record == null) continue;
    accepted.push(stripInjectionReceipts(record, injectionReceipts).value);
    if (settledAssistant(record)) safe = { consumedCount: index + 1, recordCount: accepted.length };
  }
  if (safe == null) return null;
  return { consumedCount: safe.consumedCount, records: accepted.slice(0, safe.recordCount), afterTurn: true };
}

export const hostAdapter = Object.freeze({
  host: "pi",
  protocol: "pi",
  // pi 没有内置的提问工具：插件注册的那一个（scripts/question-tool.mjs）
  questionTool: QUESTION_TOOL,
  nativeSessionId(input) { return safeIdentifier(input?.session_id); },
  conversationId(input) { return `pi-${this.nativeSessionId(input)}`; },
  prompt(input) { return typeof input?.prompt === "string" ? input.prompt.trim() : ""; },
  successOutput() { return {}; },
  contextOutput(context) { return { context }; },
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
