import { stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { completedTurns, safeIdentifier } from "./shared/host-adapter.mjs";
import { QUESTION_TOOL } from "./question-tool.mjs";

// pi 自己把会话落成 JSONL（`~/.pi/agent/sessions/.../<时刻>_<会话 id>.jsonl`，只追加；迁移、分叉时整份重写，游标按前缀认出来）：
// 直接读它，与 Codex / Claude Code 同一条路——按游标读增量、进 outbox、送回服务（协议 pi）。
// 一轮读到最后一条"说完了"的助手消息为止，只切在一轮结束处（裁定 69）：stopReason 是 toolUse（还要跑工具）或 pending 的不算说完；
// error / length 的答复若之后有一条把它从上下文里拿掉的 context_edit（targetId 指它、replacement 为 null），也不算——
// pi 要接着做同一轮：可重试的错误等一会儿重试，上下文溢出或截断先自动压缩再 agent.continue()，两条路都先写这条 context_edit
// （agent-session.ts _omitRecoveryAttempt）；不再重试的 error / length 不写它，那一轮就到此结束。

function retriedAttempts(records) {
  const omitted = new Set();
  for (const record of records) {
    if (record?.type === "context_edit" && record.replacement === null && typeof record.targetId === "string") {
      omitted.add(record.targetId);
    }
  }
  return omitted;
}

function settledAssistant(record, retried) {
  if (record?.type !== "message" || record.message?.role !== "assistant") return false;
  const stopReason = record.message?.stopReason;
  if (["toolUse", "pending"].includes(stopReason)) return false;
  return !(["error", "length"].includes(stopReason) && retried.has(record.id));
}

function selectCompleted(records, injectionReceipts) {
  const retried = retriedAttempts(records);
  const accepted = [];
  const boundaries = [];
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record == null) continue;
    accepted.push(stripInjectionReceipts(record, injectionReceipts).value);
    if (settledAssistant(record, retried)) boundaries.push({ consumedCount: index + 1, recordCount: accepted.length });
  }
  return completedTurns(accepted, boundaries);
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
