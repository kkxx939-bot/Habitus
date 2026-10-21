import { stripInjectionReceipts } from "./shared/injection-receipt.mjs";
import { readJsonlDelta } from "./shared/jsonl.mjs";
import { safeIdentifier } from "./shared/host-adapter.mjs";

// OpenCode 没有转写文件：插件在会话空闲时把已完成的消息原样镜像成一条条 {info, parts}（scripts/mirror.mjs），
// 之后与 Codex / Claude Code 同一条路：按游标读增量、剥掉我们注入的块、进 outbox、送回服务（协议 opencode）。

function completedAssistant(record) {
  return record?.info?.role === "assistant" && Number.isFinite(record?.info?.time?.completed);
}

function selectCompleted(records, injectionReceipts) {
  const accepted = [];
  let safe = null;
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record == null) continue;
    accepted.push(stripInjectionReceipts(record, injectionReceipts).value);
    if (completedAssistant(record)) safe = { consumedCount: index + 1, recordCount: accepted.length };
  }
  if (safe == null) return null;
  return { consumedCount: safe.consumedCount, records: accepted.slice(0, safe.recordCount), afterTurn: true };
}

export const hostAdapter = Object.freeze({
  host: "opencode",
  protocol: "opencode",
  // agent 向人提问、等人作答的工具（方案 32 第 4.2 条；OpenCode 的 question 工具）
  questionTool: "question",
  nativeSessionId(input) {
    const id = safeIdentifier(input?.session_id);
    return input?.parent_id ? safeIdentifier(`${input.parent_id}.subagent.${id}`) : id;
  },
  conversationId(input) { return `opencode-${this.nativeSessionId(input)}`; },
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
